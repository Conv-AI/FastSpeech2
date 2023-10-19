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
    
    speakers, emotions, basenames = [], [], []

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    filelist_fixed = open(f'{out_dir}/filelist.txt', 'w', encoding='utf-8')
    
    c = 0
    c1 = 0
    c2 = 0
    log_file = open(os.path.join(out_dir, 'log.txt'), "w", encoding='utf-8')
    for speaker in tqdm(os.listdir(in_dir)):
        for emotion in os.listdir(os.path.join(in_dir, speaker)):
            for file_name in os.listdir(os.path.join(in_dir, speaker, emotion)):
                if file_name[-4:] != ".wav":
                    continue
                base_name = file_name[:-4]
                
                if os.path.exists(os.path.join(out_dir, speaker, emotion, "{}.wav".format(base_name))) and os.path.exists(os.path.join(out_dir, speaker, emotion, "{}.lab".format(base_name))):
                    continue 
                
                speakers.append(speaker)
                emotions.append(emotion)
                basenames.append(basename)
                
    run_alignment_mp(speakers, emotions, basenames)
    

    
def run_alignment(speaker, emotion, basename):
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
        
        continue
    text = _clean_text(text, cleaners)

    os.makedirs(os.path.join(out_dir, speaker, emotion), exist_ok=True)
    try:
        wav, _ = librosa.load(wav_path)
    except Exception as e:
        
        return
    try:
        wav = wav / max(abs(wav)) * max_wav_value
    except Exception as e:
        
        return
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
    
    
def run_alignment_mp(speakers, emotions, basenames):
    num_processes = multiprocessing.cpu_count()  # You can adjust this as needed
    pool = multiprocessing.Pool(processes=num_processes)

    pool.starmap(run_alignment, zip(speakers, emotions, basenames))
    
    pool.close()
    pool.join()
    
    