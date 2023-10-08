import os
import torch
import random
import json
import warnings
import tgt
import librosa
import numpy as np
import pandas as pd
import pyworld as pw
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import multiprocessing

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
        
    def collect_all_paths(self):
        # Compute pitch, energy, duration, and mel-spectrogram
        speakers = {}
        emotions = set()
        
        tg_paths, wav_paths, text_paths = [], [], []
        for i, speaker in enumerate(tqdm(os.listdir(self.in_dir))):
            speakers[speaker] = i
            
            speaker_path = os.path.join(self.in_dir, speaker)
            if not Preprocessor.check_folder_or_file(speaker_path) == "folder":
                continue
            
            for j, emotion in enumerate(tqdm(os.listdir(os.path.join(self.in_dir, speaker)))):
                emotions.add(emotion)
                
                emotion_path = os.path.join(self.in_dir, speaker, emotion)
                if not Preprocessor.check_folder_or_file(emotion_path) == "folder":
                    continue
                
                for wav_name in os.listdir(os.path.join(self.in_dir, speaker, emotion)):
                    if ".wav" not in wav_name:
                        continue

                    basename = wav_name.split(".")[0]
                    wav_path = os.path.join(self.in_dir, speaker, emotion, "{}.wav".format(basename))
                    text_path = os.path.join(self.in_dir, speaker, emotion, "{}.lab".format(basename))
                    tg_path = os.path.join(
                        self.out_dir, "TextGrid", speaker, emotion, "{}.TextGrid".format(basename)
                    )
                    tg_paths.append(tg_path)
                    wav_paths.append(wav_path)
                    text_paths.append(text_path)
                    
        # Save files
        with open(os.path.join(self.out_dir, "speakers.json"), "w") as f:
            f.write(json.dumps(speakers))

        emotions_dict = {emotion: i for i, emotion in enumerate(emotions)}
        with open(os.path.join(self.out_dir, "emotions.json"), "w") as f:
            f.write(json.dumps(emotions_dict))
            
        paths_data = pd.DataFrame({'tg_paths': tg_paths, 'text_paths': text_paths, 'wav_paths': wav_paths})
        paths_data.to_csv(self.out_dir + '/' + 'paths.csv', index=False)
            

    
    def calculate_stats(pitches, energies, mels):
        n_frames = 0
        
        for pitch, energy, mel in zip(pitches, energies, mels):
            if len(pitch) > 0:
                pitch_scaler.partial_fit(pitch.reshape((-1, 1)))
            if len(energy) > 0:
                energy_scaler.partial_fit(energy.reshape((-1, 1)))
            if len(mel) > 0:
                n = mel.shape[1]
                n_frames += n


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
        
    def process_duration_and_pitch_mp(self, basenames, speakers, emotions):
        out = []
        num_processes = multiprocessing.cpu_count()  # You can adjust this as needed
        pool = multiprocessing.Pool(processes=num_processes)

        # Use the pool to apply the function to each pair of tg and wav paths
        out.extend(pool.starmap(self.process_duration_and_pitch, zip(basenames, speakers, emotions)))

        # Close the pool and wait for all processes to complete
        pool.close()
        pool.join()
        
        random.shuffle(out)
        out = [r for r in out if r is not None]
        
        #print(out)

        # Write metadata
        with open(os.path.join(self.out_dir, "train.txt"), "w", encoding="utf-8") as f:
            for m in out[self.val_size :]:
                f.write(m + "\n")
        with open(os.path.join(self.out_dir, "val.txt"), "w", encoding="utf-8") as f:
            for m in out[: self.val_size]:
                f.write(m + "\n")
        
    
    def process_duration_and_pitch(self, basename, speaker, emotion):
        # Get alignments
        wav_path = os.path.join(self.in_dir, speaker, emotion, "{}.wav".format(basename))
        text_path = os.path.join(self.in_dir, speaker, emotion, "{}.lab".format(basename))
        tg_path = os.path.join(
            self.out_dir, "TextGrid", speaker, emotion, "{}.TextGrid".format(basename)
        )
        
        if os.path.exists(tg_path) and os.path.exists(wav_path) and os.path.exists(text_path):
            
            textgrid = tgt.io.read_textgrid(tg_path)
            phone, duration, start, end = self.get_alignment(
                textgrid.get_tier_by_name("phones")
            )
            if not phone:
                return None
            text = "{" + " ".join(phone) + "}"
            if start >= end:
                return None

            # Read and trim wav files
            wav, _ = librosa.load(wav_path)
            wav = wav[
                int(self.sampling_rate * start) : int(self.sampling_rate * end)
            ].astype(np.float32)

            # Read raw text
            with open(text_path, "r") as f:
                raw_text = f.readline().strip("\n")

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

            # Save files
            dur_filename = "{}-{}-duration-{}.npy".format(speaker, emotion, basename)
            np.save(os.path.join(self.out_dir, "duration", dur_filename), duration)

            pitch_filename = "{}-{}-pitch-{}.npy".format(speaker, emotion, basename)
            np.save(os.path.join(self.out_dir, "pitch", pitch_filename), pitch)

            wav_filename = "{}-{}-wav-{}.npy".format(speaker, emotion, basename)
            np.save(os.path.join(self.out_dir, "wav_modified", wav_filename), wav)

            return "|".join([basename, speaker, emotion, text, raw_text])
    
    def read_npy(self, speaker, emotion, basename):
        dur, wav = None, None
        dur_filename = "{}-{}-duration-{}.npy".format(speaker, emotion, basename)
        wav_filename = "{}-{}-wav-{}.npy".format(speaker, emotion, basename)
        
        dur_path = os.path.join(self.out_dir, "duration", dur_filename)
        wav_path = os.path.join(self.out_dir, "wav_modified", wav_filename)
        if os.path.exists(dur_path) and os.path.exists(wav_path):
            dur = np.load(dur_path)
            wav = np.load(wav_path)
        
        return dur, wav
    
    def read_npy_mp(self, speakers, emotions, basenames):
        num_processes = multiprocessing.cpu_count()  # You can adjust this as needed
        pool = multiprocessing.Pool(processes=num_processes)
        
        #pool.starmap(self.process_duration_and_pitch, zip(tg_paths, wav_paths, basenames, speakers, emotions))
        
        res = pool.starmap(self.read_npy, zip(speakers, emotions, basenames))
                                   
        pool.close()
        pool.join()
        
        return res
        
        
    @staticmethod    
    def pad_batch_wavs(batch_wavs):
        """
        Pad a batch of audio waveforms with zeros to match the length of the longest waveform.

        Parameters:
            batch_wavs (np.ndarray): A 2D NumPy array where each row represents an audio waveform.

        Returns:
            np.ndarray: A 2D NumPy array containing the padded audio waveforms.
        """
        max_length = max(len(wav) for wav in batch_wavs)

        # Initialize an empty array for padded waveforms
        padded_batch = np.zeros((len(batch_wavs), max_length), dtype=np.float32)

        for i, wav in enumerate(batch_wavs):
            current_length = len(wav)
            padded_batch[i, :current_length] = wav
            
        tensor_batch = torch.Tensor(padded_batch)
        
        batch_size, max_len = tensor_batch.shape
        tensor_batch = tensor_batch.view(batch_size, -1)

        return tensor_batch
    
    def save_energy_and_mel(self, mel_spectrogram, energy, duration, speaker, basename, emotion):
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

        energy_filename = "{}-{}-energy-{}.npy".format(speaker, emotion, basename)
        np.save(os.path.join(self.out_dir, "energy", energy_filename), energy)

        mel_filename = "{}-{}-mel-{}.npy".format(speaker, emotion, basename)
        np.save(
            os.path.join(self.out_dir, "mel", mel_filename),
            mel_spectrogram.T,
        ) 
        
        
        
    def process_mels_and_energies(self, basenames, speakers, emotions, batch_size):
        # Compute mel-scale spectrogram and energy
        n = int(len(speakers)/batch_size) + 1
        
        for i in range(n):
            basenames_ = basenames[batch_size*i: batch_size*(i+1)]
            speakers_ = speakers[batch_size*i: batch_size*(i+1)]
            emotions_ = emotions[batch_size*i: batch_size*(i+1)]
            res = self.read_npy_mp(speakers_, emotions_, basenames_)
            wavs = [item[1] for item in res if isinstance(item[1], np.ndarray)]
            durations = [item[0] for item in res if isinstance(item[0], np.ndarray)]
            
            wavs = Preprocessor.pad_batch_wavs(wavs)
    
            #print(wavs)
            print(i)
            
            mel_spectrograms, energies = Audio.tools.get_mel_from_wav(wavs, self.STFT)
            
            num_processes = multiprocessing.cpu_count()  # You can adjust this as needed
            pool = multiprocessing.Pool(processes=num_processes)
            
            pool.starmap(self.save_energy_and_mel, zip(mel_spectrograms, energies, durations, speakers_, basenames_, emotions_))

            # Close the pool and wait for all processes to complete
            pool.close()
            pool.join()
            
    def read_pitch_energy_mel(self, speaker, emotion, basename):
        pitch, energy, mel = None, None, None
        pitch_filename = "{}-{}-pitch-{}.npy".format(speaker, emotion, basename)
        energy_filename = "{}-{}-energy-{}.npy".format(speaker, emotion, basename)
        mel_filename = "{}-{}-mel-{}.npy".format(speaker, emotion, basename)
        
        pitch_path = os.path.join(self.out_dir, "pitch", pitch_filename)
        energy_path = os.path.join(self.out_dir, "energy", energy_filename)
        mel_path = os.path.join(self.out_dir, "mel", mel_filename)
        
        #print(pitch_path, energy_path, mel_path)
        
        if os.path.exists(pitch_path) and os.path.exists(energy_path) and os.path.exists(mel_path):
            pitch = np.load(pitch_path)
            energy = np.load(energy_path)
            mel = np.load(mel_path)
        
        return pitch, energy, mel
        
        
    def build_from_path(self, speakers, emotions, basenames):
        out = list()
        n_frames = 0
        pitch_scaler = StandardScaler()
        energy_scaler = StandardScaler()
        
        num_processes = multiprocessing.cpu_count()  # You can adjust this as needed
        pool = multiprocessing.Pool(processes=num_processes)
        
        res = pool.starmap(self.read_pitch_energy_mel, zip(speakers, emotions, basenames))
        
        pool.close()
        pool.join()
        
        for item in res:
            #print(item)
            pitch = item[0]
            energy = item[1]
            if not isinstance(pitch, np.ndarray) or not isinstance(energy, np.ndarray):
                continue
            pitch = self.remove_outlier(pitch)
            energy = self.remove_outlier(energy)
            mel = item[2]
            n = mel.shape[1]
            n_frames += n
            
            if len(pitch) > 0:
                pitch_scaler.partial_fit(pitch.reshape((-1, 1)))
                
            if len(energy) > 0:
                energy_scaler.partial_fit(energy.reshape((-1, 1)))
                
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

                    warnings.warn('Find a slice with empty text.')
                    new_tier.append(tgt.Interval(start_time=t1.end_time, end_time=t2.start_time, text='spn'))
            new_tier.append(t2)
        except:
            print()
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

        