import os
import re
import random
import json
import warnings
import tgt
import librosa
import numpy as np
import pyworld as pw
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import soundfile as sf

import audio as Audio


class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.in_dir = config["path"]["raw_path"]
        self.out_dir = config["path"]["preprocessed_path"]
        self.val_size = config["preprocessing"]["val_size"]
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.hop_length = config["preprocessing"]["stft"]["hop_length"]
        
        self.fail_count = 0
        
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        assert config["preprocessing"]["pitch"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]
        assert config["preprocessing"]["energy"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]
        self.pitch_phoneme_averaging = (
            config["preprocessing"]["pitch"]["feature"] == "phoneme_level"
        )
        self.energy_phoneme_averaging = (
            config["preprocessing"]["energy"]["feature"] == "phoneme_level"
        )

        self.pitch_normalization = config["preprocessing"]["pitch"]["normalization"]
        self.energy_normalization = config["preprocessing"]["energy"]["normalization"]

        self.STFT = Audio.stft.TacotronSTFT(
            config["preprocessing"]["stft"]["filter_length"],
            config["preprocessing"]["stft"]["hop_length"],
            config["preprocessing"]["stft"]["win_length"],
            config["preprocessing"]["mel"]["n_mel_channels"],
            config["preprocessing"]["audio"]["sampling_rate"],
            config["preprocessing"]["mel"]["mel_fmin"],
            config["preprocessing"]["mel"]["mel_fmax"],
        )
    
    @staticmethod
    def check_folder_or_file(name):
        if os.path.exists(name):
            if os.path.isdir(name):
                return "folder"
            elif os.path.isfile(name):
                return "file"
            else:
                return "neither"
        else:
            return "not_exist"

    def build_from_path(self):
        os.makedirs((os.path.join(self.out_dir, "mel")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "pitch")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "energy")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "duration")), exist_ok=True)
        

        print("Processing Data ...")
        out = list()
        n_frames = 0
        pitch_scaler = StandardScaler()
        energy_scaler = StandardScaler()
        
        total_wavs = 0
        for speaker in os.listdir(self.in_dir):
            for emotion in os.listdir(os.path.join(self.in_dir, speaker)):
                for wav_name in os.listdir(os.path.join(self.in_dir, speaker, emotion)):
                    if ".wav" not in wav_name:
                        continue
                    total_wavs += 1  
                    
        pbar = tqdm(total=total_wavs, desc="Overall Progress")
        
        #temporary code, needs to be removed later
#         import pandas as pd
#         filter_ = pd.read_csv('/workspace/nemo/vol/vol5/filtered_nova.csv') 
#         filelist = filter_.filepath.to_list()

        # Compute pitch, energy, duration, and mel-spectrogram
        speakers = {}
        emotions = set()
        for i, speaker in enumerate(os.listdir(self.in_dir)):
            speakers[speaker] = i
            
            speaker_path = os.path.join(self.in_dir, speaker)
            if not Preprocessor.check_folder_or_file(speaker_path) == "folder":
                continue
            
            for j, emotion in enumerate(os.listdir(os.path.join(self.in_dir, speaker))):
                emotions.add(emotion)
                
                emotion_path = os.path.join(self.in_dir, speaker, emotion)
                if not Preprocessor.check_folder_or_file(emotion_path) == "folder":
                    continue
                
                for wav_name in os.listdir(os.path.join(self.in_dir, speaker, emotion)):
                    if ".wav" not in wav_name:
                        continue
                    pbar.update(1)
                    
                    #temporary code. Needs to be removed later
#                     if speaker == 'NovaConvai':
#                         if wav_name not in filelist:
#                             continue
                    

                    basename = wav_name.split(".")[0]
                    tg_path = os.path.join(
                        self.out_dir, "TextGrid", speaker, emotion, "{}.TextGrid".format(basename)
                    )
                    #print("tg_path:", tg_path)
                    if os.path.exists(tg_path):
                        ret = self.process_utterance(speaker, emotion, basename)
                        if ret is None:
                            continue
                        else:
                            info, pitch, energy, n = ret
                        out.append(info)
                        
                    else:
                        self.fail_count += 1
                        print(tg_path)
                        print("********************" + str(self.fail_count) + "**********************")
                        continue

                    if len(pitch) > 0:
                        pitch_scaler.partial_fit(pitch.reshape((-1, 1)))
                    if len(energy) > 0:
                        energy_scaler.partial_fit(energy.reshape((-1, 1)))

                    n_frames += n
        pbar.close()


        print("Computing statistic quantities ...")
        # Perform normalization if necessary
        if self.pitch_normalization:
            pitch_mean = pitch_scaler.mean_[0]
            pitch_std = pitch_scaler.scale_[0]
        else:
            # A numerical trick to avoid normalization...
            pitch_mean = 0
            pitch_std = 1
        if self.energy_normalization:
            energy_mean = energy_scaler.mean_[0]
            energy_std = energy_scaler.scale_[0]
        else:
            energy_mean = 0
            energy_std = 1

        pitch_min, pitch_max = self.normalize(
            os.path.join(self.out_dir, "pitch"), pitch_mean, pitch_std
        )
        energy_min, energy_max = self.normalize(
            os.path.join(self.out_dir, "energy"), energy_mean, energy_std
        )

        # Save files
        with open(os.path.join(self.out_dir, "speakers.json"), "w") as f:
            f.write(json.dumps(speakers))

        emotions_dict = {emotion: i for i, emotion in enumerate(emotions)}
        with open(os.path.join(self.out_dir, "emotions.json"), "w") as f:
            f.write(json.dumps(emotions_dict))

        with open(os.path.join(self.out_dir, "stats.json"), "w") as f:
            stats = {
                "pitch": [
                    float(pitch_min),
                    float(pitch_max),
                    float(pitch_mean),
                    float(pitch_std),
                ],
                "energy": [
                    float(energy_min),
                    float(energy_max),
                    float(energy_mean),
                    float(energy_std),
                ],
            }
            f.write(json.dumps(stats))

        print(
            "Total time: {} hours".format(
                n_frames * self.hop_length / self.sampling_rate / 3600
            )
        )

        random.shuffle(out)
        out = [r for r in out if r is not None]

        # Write metadata
        with open(os.path.join(self.out_dir, "train.txt"), "w", encoding="utf-8") as f:
            for m in out[self.val_size :]:
                f.write(m + "\n")
        with open(os.path.join(self.out_dir, "val.txt"), "w", encoding="utf-8") as f:
            for m in out[: self.val_size]:
                f.write(m + "\n")

        return out

    def process_utterance(self, speaker, emotion, basename):
        wav_path = os.path.join(self.in_dir, speaker, emotion, "{}.wav".format(basename))
        if not os.path.exists(os.path.join(self.in_dir + "_modified_wav", speaker, emotion)):
            os.makedirs((os.path.join(self.in_dir + "_modified_wav", speaker, emotion)), exist_ok=True)
        modified_wav_path = os.path.join(self.in_dir + "_modified_wav", speaker, emotion, "{}.wav".format(basename))
        
        text_path = os.path.join(self.in_dir, speaker, emotion, "{}.lab".format(basename))
        tg_path = os.path.join(
            self.out_dir, "TextGrid", speaker, emotion, "{}.TextGrid".format(basename)
        )
        
        # Read raw text
        with open(text_path, "r") as f:
            raw_text = f.readline().strip("\n")
            
        # Read and trim wav files
        wav, sr = librosa.load(wav_path)
        duration = librosa.get_duration(y=wav)
        
#         if duration < 2:
#             #print("********************" + str(self.fail_count) + "**********************")
#             return None

        # Get alignments
        textgrid = tgt.io.read_textgrid(tg_path)
#         import pdb
#         pdb.set_trace()
        phone, duration, start, end, modified_wav = self.get_alignment(
            textgrid, raw_text, wav, sr
        )
        
        
        try:
            sf.write(modified_wav_path, modified_wav, sr)
            if not phone:
                return None
            text = "{" + " ".join(phone) + "}"
            if start >= end:
                return None


            wav = modified_wav[
                int(self.sampling_rate * start) : int(self.sampling_rate * end)
            ].astype(np.float32)

            # Compute fundamental frequency
            pitch, t = pw.dio(
                wav.astype(np.float64),
                self.sampling_rate,
                frame_period=self.hop_length / self.sampling_rate * 1000,
            )
            pitch = pw.stonemask(wav.astype(np.float64), pitch, t, self.sampling_rate)

            pitch = pitch[: sum(duration)]
            if np.sum(pitch != 0) <= 1:
                return None

            # Compute mel-scale spectrogram and energy
            mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, self.STFT)
            mel_spectrogram = mel_spectrogram[:, : sum(duration)]
            energy = energy[: sum(duration)]

            if self.pitch_phoneme_averaging:
                # perform linear interpolation
                nonzero_ids = np.where(pitch != 0)[0]
                interp_fn = interp1d(
                    nonzero_ids,
                    pitch[nonzero_ids],
                    fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
                    bounds_error=False,
                )
                pitch = interp_fn(np.arange(0, len(pitch)))

                # Phoneme-level average
                pos = 0
                for i, d in enumerate(duration):
                    if d > 0:
                        pitch[i] = np.mean(pitch[pos : pos + d])
                    else:
                        pitch[i] = 0
                    pos += d
                pitch = pitch[: len(duration)]

            if self.energy_phoneme_averaging:
                # Phoneme-level average
                pos = 0
                for i, d in enumerate(duration):
                    if d > 0:
                        energy[i] = np.mean(energy[pos : pos + d])
                    else:
                        energy[i] = 0
                    pos += d
                energy = energy[: len(duration)]
                
#             import pdb
#             pdb.set_trace()

            if np.isnan(pitch).any() or np.isnan(energy).any() or np.isnan(mel_spectrogram).any():
                self.fail_count += 1
                print("********************" + str(self.fail_count) + "**********************")
                return None

            # Save files
            dur_filename = "{}-{}-duration-{}.npy".format(speaker, emotion, basename)
            np.save(os.path.join(self.out_dir, "duration", dur_filename), duration)

            pitch_filename = "{}-{}-pitch-{}.npy".format(speaker, emotion, basename)
            np.save(os.path.join(self.out_dir, "pitch", pitch_filename), pitch)

            energy_filename = "{}-{}-energy-{}.npy".format(speaker, emotion, basename)
            np.save(os.path.join(self.out_dir, "energy", energy_filename), energy)

            mel_filename = "{}-{}-mel-{}.npy".format(speaker, emotion, basename)
            np.save(
                os.path.join(self.out_dir, "mel", mel_filename),
                mel_spectrogram.T,
            )

            return (
                "|".join([basename, speaker, emotion, text, raw_text]),
                self.remove_outlier(pitch),
                self.remove_outlier(energy),
                mel_spectrogram.shape[1],
            )
        except:
            print("failed data writing for: " + modified_wav_path)
            self.fail_count += 1
            print("********************" + str(self.fail_count) + "**********************")
            return None
        
        

    def get_alignment(self, textgrid, text, wav, sample_rate):
        sil_phones = ["sil", "sp", "spn", "fstop", "ques", "excl", "comma", "semic"]
        sil_dict = {'.': "fstop", '?': "ques", '!': "excl", ',': "comma", ';': "semic"}
        
        new_tier_words = []
        tier = textgrid.get_tier_by_name("words")
        try:
            for t1, t2 in zip(tier._objects[:-1], tier._objects[1:]):
                new_tier_words.append(t1)
                if (t2.start_time - t1.end_time) != 0:

                    #warnings.warn('Find a slice with empty text.')
                    new_tier_words.append(tgt.Interval(start_time=t1.end_time, end_time=t2.start_time, text=''))
            new_tier_words.append(t2)
            new_tier_words.append(tgt.Interval(start_time=t2.end_time, end_time=t2.end_time + 0.5, text=''))
        except:
            print()
            return None, None, None, None, None
        
        
        new_tier_ph = []
        tier = textgrid.get_tier_by_name("phones")
        for t1, t2 in zip(tier._objects[:-1], tier._objects[1:]):
            new_tier_ph.append(t1)
            if (t2.start_time - t1.end_time) != 0:

                #warnings.warn('Find a slice with empty text.')
                new_tier_ph.append(tgt.Interval(start_time=t1.end_time, end_time=t2.start_time, text=''))
        new_tier_ph.append(t2)
        new_tier_ph.append(tgt.Interval(start_time=t2.end_time, end_time=t2.end_time + 0.5, text=''))
        
        words = re.split(r"([,;.\-\?\!\s+])", text)
        trim_words = [word for word in words if word not in ['', ' ']]
        
        for interval, word in zip(new_tier_words, trim_words):
            if interval.text == "":
                interval.text = word
                
        for interval_ph in new_tier_ph:
            if interval_ph.text == "":
                start = interval_ph.start_time
                end = interval_ph.end_time
                interval = [i for i in new_tier_words if i.start_time <= start and i.end_time >= end]
                interval_ph.text = sil_dict.get(interval[0].text, 'sp')
                
        # match wav and ph end times by padding zeros
        last_ph_end = new_tier_ph[-1].end_time
        padded_ph_length = int(last_ph_end *sample_rate)
        zero_pad_length = (padded_ph_length - wav.shape[0]) + 1
        if zero_pad_length > 0:
            zero_pad = np.zeros(zero_pad_length)
            wav = np.concatenate((wav, zero_pad))
        
        for interval_ph in new_tier_ph:
            start, end, text = interval_ph.start_time, interval_ph.end_time, interval_ph.text
            if text in ["fstop", "ques", "excl", "comma", "semic"]:
#                 import pdb
#                 pdb.set_trace()
                
                epsilon = 1e-10
                start_sample = int(start * sample_rate)
                end_sample = int(end * sample_rate)

                # Duration of silence to add in samples
                silence_duration_samples = end_sample - start_sample
                
                fade_in_duration = min(int(0.1 * sample_rate), int(0.3 * silence_duration_samples))  
                fade_out_duration = min(int(0.15 * sample_rate), int(0.7 * silence_duration_samples))
                
                fade_out = np.linspace(wav[start_sample], epsilon, fade_out_duration)
                fade_in = np.linspace(epsilon, wav[end_sample], fade_in_duration)
                
                
                
                silence_length = silence_duration_samples - (fade_in_duration + fade_out_duration)
                #if silence_length > 0:
                silence = np.concatenate((fade_out, np.full(silence_length, epsilon), fade_in))
#                 else:
#                     silence = np.full(silence_duration_samples, epsilon)
                wav[start_sample: end_sample] = silence
            

        phones = []
        durations = []
        start_time = 0
        end_time = 0
        end_idx = 0
        
#         import pdb
#         pdb.set_trace()
        
#         try:
#             new_tier = [] # MFA2.0(default config) may yield slices with no token which leads a mismatch in length.
#             for t1, t2 in zip(tier._objects[:-1], tier._objects[1:]):
#                 new_tier.append(t1)
#                 if (t2.start_time - t1.end_time) != 0:

#                     #warnings.warn('Find a slice with empty text.')
#                     new_tier.append(tgt.Interval(start_time=t1.end_time, end_time=t2.start_time, text='spn'))
#             new_tier.append(t2)
#         except:
# #             print()
#             return None, None, None, None
        
        for t in new_tier_ph:
            s, e, p = t.start_time, t.end_time, t.text

            # Trim leading silences
            if phones == []:
                if p in sil_phones:
                    continue
                else:
                    start_time = s

#             if p not in sil_phones:
                # For ordinary phones
            phones.append(p)
            end_time = e
            end_idx = len(phones)
#             else:
#                 # For silent phones
#                 phones.append(p)

            durations.append(
                int(
                    np.round(e * self.sampling_rate / self.hop_length)
                    - np.round(s * self.sampling_rate / self.hop_length)
                )
            )

        # Trim tailing silences
#         phones = phones[:end_idx]
#         durations = durations[:end_idx]

        return phones, durations, start_time, end_time, wav

    def remove_outlier(self, values):
        values = np.array(values)
        p25 = np.percentile(values, 25)
        p75 = np.percentile(values, 75)
        lower = p25 - 1.5 * (p75 - p25)
        upper = p75 + 1.5 * (p75 - p25)
        normal_indices = np.logical_and(values > lower, values < upper)

        return values[normal_indices]

    def normalize(self, in_dir, mean, std):
        max_value = np.finfo(np.float64).min
        min_value = np.finfo(np.float64).max
        for filename in os.listdir(in_dir):
            filename = os.path.join(in_dir, filename)
            values = (np.load(filename) - mean) / std
            np.save(filename, values)

            max_value = max(max_value, max(values))
            min_value = min(min_value, min(values))

        return min_value, max_value
