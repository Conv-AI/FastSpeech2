import os

import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

from text import _clean_text


def prepare_align(config):
    in_dir = config["path"]["corpus_path"]
    out_dir = config["path"]["raw_path"]
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    cleaners = config["preprocessing"]["text"]["text_cleaners"]

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    filelist_fixed = open(f'{out_dir}/filelist.txt', 'w', encoding='utf-8')
    
    c = 0
    c1 = 0
    c2 = 0
    count = 0
    log_file = open(os.path.join(out_dir, 'log.txt'), "w", encoding='utf-8')
    for speaker in tqdm(os.listdir(in_dir)):
        for emotion in os.listdir(os.path.join(in_dir, speaker)):
            for file_name in os.listdir(os.path.join(in_dir, speaker, emotion)):
                
                if file_name[-4:] != ".wav":
                    continue
                base_name = file_name[:-4]
                count += 1
                print(count, end='\r')
                
                if os.path.exists(os.path.join(out_dir, speaker, emotion, "{}.wav".format(base_name))) and os.path.exists(os.path.join(out_dir, speaker, emotion, "{}.lab".format(base_name))):
                    continue 
                
                text_path = os.path.join(
                    in_dir, speaker, emotion, "{}.lab".format(base_name)
                )
                wav_path = os.path.join(
                    in_dir, speaker, emotion, "{}.wav".format(base_name)
                )
                
                               
                
                try:
                    with open(text_path) as f:
                        text = f.readline().strip("\n")
                except Exception as e:
                    log_file.write("Lab File Not readable::::"+text_path+"\n")
                    log_file.flush()
                    c += 1
#                     print(c, end='\r')
                    c1 += 1
                    continue
                text = _clean_text(text, cleaners)

                os.makedirs(os.path.join(out_dir, speaker, emotion), exist_ok=True)
                try:
                    wav, _ = librosa.load(wav_path)
                except Exception as e:
                    log_file.write("Wav File Not readable::::"+text_path+"\n")
                    log_file.flush()
                    c += 1
#                     print(c, end='\r')
                    c2 += 1
                    continue
                try:
                    wav = wav / max(abs(wav)) * max_wav_value
                except Exception as e:
                    log_file.write("Wav file Empty::::"+wav_path+"\n")
                    log_file.flush()
                    c += 1
#                     print(c, end='\r')
                    continue
#                     os.remove(wav_path)
#                     os.remove(text_path)
                wavfile.write(
                    os.path.join(out_dir, speaker, emotion, "{}.wav".format(base_name)),
                    sampling_rate,
                    wav.astype(np.int16),
                )
                with open(
                    os.path.join(out_dir, speaker, emotion, "{}.lab".format(base_name)),
                    "w",
                ) as f1:
                    f1.write(text)
                filelist_fixed.write("|".join([base_name, text, speaker, emotion]) + "\n")
                filelist_fixed.flush()
                
    filelist_fixed.close()
    log_file.close()
    print("Total empty wavs:::::::::::::::::"+str(c-c1-c2))
    print("Total unreadable wavs:::::::::::::::::"+str(c2))
    print("Total unreadable labs:::::::::::::::::"+str(c1))
