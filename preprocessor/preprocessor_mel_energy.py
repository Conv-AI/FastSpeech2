import os
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

import audio as Audio


class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.in_dir = config["path"]["raw_path"]
        self.out_dir = config["path"]["preprocessed_path"]
        self.val_size = config["preprocessing"]["val_size"]
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.hop_length = config["preprocessing"]["stft"]["hop_length"]
        
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
#         os.makedirs((os.path.join(self.out_dir, "mel")), exist_ok=True)
#         os.makedirs((os.path.join(self.out_dir, "pitch")), exist_ok=True)
#         os.makedirs((os.path.join(self.out_dir, "energy")), exist_ok=True)
#         os.makedirs((os.path.join(self.out_dir, "duration")), exist_ok=True)

#         print("Processing Data ...")
#         out = list()
#         n_frames = 0
#         pitch_scaler = StandardScaler()
#         energy_scaler = StandardScaler()
        
        total_wavs = 0
        for speaker in os.listdir(self.in_dir):
            for emotion in os.listdir(os.path.join(self.in_dir, speaker)):
                for wav_name in os.listdir(os.path.join(self.in_dir, speaker, emotion)):
                    if ".wav" not in wav_name:
                        continue
                    total_wavs += 1  
                    
        pbar = tqdm(total=total_wavs, desc="Overall Progress")

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

                    basename = wav_name.split(".")[0]
                    tg_path = os.path.join(
                        self.out_dir, "TextGrid", speaker, emotion, "{}.TextGrid".format(basename)
                    )
                    modified_wav_filename = "{}-{}-wav-{}.npy".format(speaker, emotion, basename)
                    modified_wav_path = os.path.join(self.out_dir, 'wav_modified', modified_wav_filename)
        
#                     dur_filename = "{}-{}-duration-{}.npy".format(speaker, emotion, basename)
#                     duration_path = os.path.join(self.out_dir, 'duration', dur_filename)
                    
#                     print("tg_path:", tg_path)
                    if os.path.exists(tg_path) and os.path.exists(modified_wav_path):
                        self.process_utterance(speaker, emotion, basename)
#                         if ret is None:
#                             continue
#                         else:
#                             info, pitch, energy, n = ret
#                         out.append(info)

#                     if len(pitch) > 0:
#                         pitch_scaler.partial_fit(pitch.reshape((-1, 1)))
#                     if len(energy) > 0:
#                         energy_scaler.partial_fit(energy.reshape((-1, 1)))

#                     n_frames += n
        pbar.close()


#         print("Computing statistic quantities ...")
#         # Perform normalization if necessary
#         if self.pitch_normalization:
#             pitch_mean = pitch_scaler.mean_[0]
#             pitch_std = pitch_scaler.scale_[0]
#         else:
#             # A numerical trick to avoid normalization...
#             pitch_mean = 0
#             pitch_std = 1
#         if self.energy_normalization:
#             energy_mean = energy_scaler.mean_[0]
#             energy_std = energy_scaler.scale_[0]
#         else:
#             energy_mean = 0
#             energy_std = 1

#         pitch_min, pitch_max = self.normalize(
#             os.path.join(self.out_dir, "pitch"), pitch_mean, pitch_std
#         )
#         energy_min, energy_max = self.normalize(
#             os.path.join(self.out_dir, "energy"), energy_mean, energy_std
#         )

#         # Save files
#         with open(os.path.join(self.out_dir, "speakers.json"), "w") as f:
#             f.write(json.dumps(speakers))

#         emotions_dict = {emotion: i for i, emotion in enumerate(emotions)}
#         with open(os.path.join(self.out_dir, "emotions.json"), "w") as f:
#             f.write(json.dumps(emotions_dict))

#         with open(os.path.join(self.out_dir, "stats.json"), "w") as f:
#             stats = {
#                 "pitch": [
#                     float(pitch_min),
#                     float(pitch_max),
#                     float(pitch_mean),
#                     float(pitch_std),
#                 ],
#                 "energy": [
#                     float(energy_min),
#                     float(energy_max),
#                     float(energy_mean),
#                     float(energy_std),
#                 ],
#             }
#             f.write(json.dumps(stats))

#         print(
#             "Total time: {} hours".format(
#                 n_frames * self.hop_length / self.sampling_rate / 3600
#             )
#         )

#         random.shuffle(out)
#         out = [r for r in out if r is not None]

#         # Write metadata
#         with open(os.path.join(self.out_dir, "train.txt"), "w", encoding="utf-8") as f:
#             for m in out[self.val_size :]:
#                 f.write(m + "\n")
#         with open(os.path.join(self.out_dir, "val.txt"), "w", encoding="utf-8") as f:
#             for m in out[: self.val_size]:
#                 f.write(m + "\n")

#         return out

    def process_utterance(self, speaker, emotion, basename):
#         wav_path = os.path.join(self.in_dir, speaker, emotion, "{}.wav".format(basename))
#         text_path = os.path.join(self.in_dir, speaker, emotion, "{}.lab".format(basename))
#         tg_path = os.path.join(
#             self.out_dir, "TextGrid", speaker, emotion, "{}.TextGrid".format(basename)
#         )
        modified_wav_filename = "{}-{}-wav-{}.npy".format(speaker, emotion, basename)
        modified_wav_path = os.path.join(self.out_dir, 'wav_modified', modified_wav_filename)
        
        dur_filename = "{}-{}-duration-{}.npy".format(speaker, emotion, basename)
        duration_path = os.path.join(self.out_dir, 'duration', dur_filename)
        
#         pitch_filename = "{}-{}-pitch-{}.npy".format(speaker, emotion, basename)
#         pitch_path = os.path.join(self.out_dir, 'pitch', pitch_filename)
        

#         # Get alignments
#         textgrid = tgt.io.read_textgrid(tg_path)
#         phone, duration, start, end = self.get_alignment(
#             textgrid.get_tier_by_name("phones")
#         )
#         if not phone:
#             return None
#         text = "{" + " ".join(phone) + "}"
#         if start >= end:
#             return None

#         # Read and trim wav files
#         wav, _ = librosa.load(wav_path)
#         wav = wav[
#             int(self.sampling_rate * start) : int(self.sampling_rate * end)
#         ].astype(np.float32)

#         # Read raw text
#         with open(text_path, "r") as f:
#             raw_text = f.readline().strip("\n")

        # Compute fundamental frequency
#         pitch, t = pw.dio(
#             wav.astype(np.float64),
#             self.sampling_rate,
#             frame_period=self.hop_length / self.sampling_rate * 1000,
#         )
#         pitch = pw.stonemask(wav.astype(np.float64), pitch, t, self.sampling_rate)

#         pitch = pitch[: sum(duration)]
#         if np.sum(pitch != 0) <= 1:
#             return None

        # Compute mel-scale spectrogram and energy
    
        wav = np.load(modified_wav_path)
        duration = np.load(duration_path)
        
        mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, self.STFT)
        mel_spectrogram = mel_spectrogram[:, : sum(duration)]
        energy = energy[: sum(duration)]

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

#         # Save files
#         dur_filename = "{}-{}-duration-{}.npy".format(speaker, emotion, basename)
#         np.save(os.path.join(self.out_dir, "duration", dur_filename), duration)

#         pitch_filename = "{}-{}-pitch-{}.npy".format(speaker, emotion, basename)
#         np.save(os.path.join(self.out_dir, "pitch", pitch_filename), pitch)

        energy_filename = "{}-{}-energy-{}.npy".format(speaker, emotion, basename)
        np.save(os.path.join(self.out_dir, "energy", energy_filename), energy)

        mel_filename = "{}-{}-mel-{}.npy".format(speaker, emotion, basename)
        np.save(
            os.path.join(self.out_dir, "mel", mel_filename),
            mel_spectrogram.T,
        )

#         return (
#             "|".join([basename, speaker, emotion, text, raw_text]),
#             self.remove_outlier(pitch),
#             self.remove_outlier(energy),
#             mel_spectrogram.shape[1],
#         )

    def get_alignment(self, tier):
        sil_phones = ["sil", "sp", "spn"]

        phones = []
        durations = []
        start_time = 0
        end_time = 0
        end_idx = 0
        
        try:
            new_tier = [] # MFA2.0(default config) may yield slices with no token which leads a mismatch in length.
            for t1, t2 in zip(tier._objects[:-1], tier._objects[1:]):
                new_tier.append(t1)
                if (t2.start_time - t1.end_time) != 0:

                    #warnings.warn('Find a slice with empty text.')
                    new_tier.append(tgt.Interval(start_time=t1.end_time, end_time=t2.start_time, text='spn'))
            new_tier.append(t2)
        except:
#             print()
            return None, None, None, None
        
        for t in new_tier:
            s, e, p = t.start_time, t.end_time, t.text

            # Trim leading silences
            if phones == []:
                if p in sil_phones:
                    continue
                else:
                    start_time = s

            if p not in sil_phones:
                # For ordinary phones
                phones.append(p)
                end_time = e
                end_idx = len(phones)
            else:
                # For silent phones
                phones.append(p)

            durations.append(
                int(
                    np.round(e * self.sampling_rate / self.hop_length)
                    - np.round(s * self.sampling_rate / self.hop_length)
                )
            )

        # Trim tailing silences
        phones = phones[:end_idx]
        durations = durations[:end_idx]

        return phones, durations, start_time, end_time

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
