import os
import re
import yaml
import json
import torch
import torch.nn as nn
import numpy as np
from g2p_en import G2p
from model import FastSpeech2
from utils.tools import pad_1D
from text import text_to_sequence
from synthesize import read_lexicon

import IPython.display as ipd
from string import punctuation

from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int)
args = parser.parse_args()
device = args.device

torch.cuda.set_device(f'cuda:{device}')

ckpt_file_path = '/workspace/nemo/vol/extvol1/emotional_transference_2324_speakers.pth'
checkpoint = torch.load(ckpt_file_path, map_location=torch.device('cpu'))

model_config = yaml.load(open('/workspace/nemo/vol/FastSpeech2/config/emo_exp/model.yaml',
                              "r"), Loader=yaml.FullLoader)
preprocess_config = yaml.load(open('/workspace/nemo/vol/FastSpeech2/config/emo_exp/preprocess.yaml',
                              "r"), Loader=yaml.FullLoader)

fp = FastSpeech2(model_config=model_config, preprocess_config=preprocess_config)
fp.load_state_dict(checkpoint['model'])
# fp = nn.DataParallel(fp)
fp = fp.to(f'cuda:{device}')

g2p = G2p()
lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

def preprocess_english(text, preprocess_config, g2p = g2p, lexicon = lexicon):
    text = text.rstrip(punctuation)
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")
    
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)
    
def get_inference(text_list, speaker_list, emotion_list):
    speakers = torch.tensor(speaker_list).to(f'cuda:{device}')
    emotion = torch.tensor(emotion_list).to(f'cuda:{device}')
    text = [preprocess_english(i, preprocess_config) for i in text_list]
    text = pad_1D(text)
    texts = torch.tensor(text).to(f'cuda:{device}')

    text_lens = torch.tensor([len(i) for i in texts]).to(f'cuda:{device}')
    max_len = max(text_lens)
    
    batchs = [(speakers, emotion, texts, text_lens,max_len )]
    #     print(batchs)
                
    predictions = fp(*(batchs[0]))
    mel_predictions = predictions[1].permute(0, 2, 1)
    
    return mel_predictions
    
def save_predicted_mels(mel_predictions, emotion_list, speaker_list, basename_list, target_dir = '/workspace/nemo/vol/extvol2/mels' ):
    save_paths = []
    for speaker, emotion, basename, mel in zip(speaker_list, emotion_list, basename_list, mel_predictions):
        save_path = f"{target_dir}/mel_{speaker}_{emotion}_{basename}.npy"
        np.save(save_path, mel.detach().to('cpu').numpy())
        save_paths.append(save_path)
    return save_paths
    
# function to read in the emotion and speaker json
emotion_json_path = "/workspace/nemo/vol/extvol1/emotions.json"
EMOTION_DICT = {}

with open(emotion_json_path) as f:
    EMOTION_DICT = json.loads(f.read())
    
speakers_json_path = "/workspace/nemo/vol/extvol1/speakers.json"
SPEAKER_ID_DICT = {}

with open(speakers_json_path) as f:
    SPEAKER_ID_DICT = json.loads(f.read())
    
def get_speaker_id(x):
    return SPEAKER_ID_DICT.get(x,'--NA--')

def get_emotion_id(x):
    return EMOTION_DICT.get(x,'--NA--')
    
from torch.utils.data import Dataset

class Dataset(Dataset):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.basename, self.speaker, self.emotions, self.text = self.process_meta()
        self.drop_last = False

    def __len__(self):
        return len(self.text)
    
    def process_meta(self):
        number_of_rows_to_process = 180353
        start_index = number_of_rows_to_process*device
        end_index = start_index + number_of_rows_to_process + 1
        
        data = pd.read_csv('/workspace/nemo/vol/extvol1/data-points-rec-2.csv')
        
        data = data.iloc[start_index:end_index]
        
        print("Total :  ",len(data))
        data['_speaker_id'] = data['speaker_id'].apply(get_speaker_id)
        data['_emotion_id'] = data['emotion'].apply(get_emotion_id)
        
        data = data[data['_speaker_id']!='--NA--']
        data = data[data['_emotion_id']!='--NA--']
        
        name = []
        speaker = []
        text = []
        emotions = []
        
        for index, row in data.iterrows():
            n = os.path.splitext(os.path.basename(row['basename']))[0]
            s = row['_speaker_id']
            e = row['_emotion_id']
            t = row['text']
            
            if os.path.exists(f"/workspace/nemo/vol/extvol2/mels/mel_{s}_{e}_{n}.npy"):
                # skipping the files that were already processed
                continue
            
            name.append(n)
            speaker.append(s)
            text.append(t)
            emotions.append(e)
        print("After filtering :  ",len(text))
        return name, speaker, emotions, text

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker_id = self.speaker[idx]
        emotion_id = self.emotions[idx]
        text = self.text[idx]
        sample = {
            "id": basename,
            "speaker": speaker_id,
            "emotion": emotion_id,
            "text": text
        }

        return sample
    
    def reprocess(self, data, idxs):
        ids = [data[idx]["id"] for idx in idxs]
        speakers = [data[idx]["speaker"] for idx in idxs]
        texts = [data[idx]["text"] for idx in idxs]
        emotions = [data[idx]["emotion"] for idx in idxs]

        speakers = np.array(speakers)
        emotions = np.array(emotions)

        return (
            ids,
            speakers,
            emotions,
            texts
        )

    def collate_fn(self, data):
        data_size = len(data)

        idx_arr = np.arange(data_size)

        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size):]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        output = list()
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))

        return output
        
batch_size = 8

dataset = Dataset(batch_size)
loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )
    
UNIVERSAL_DU_MEL_LIST = []

for batches in tqdm(loader):
    for batch in batches:
        try:
            basename_list, speaker_list, emotion_list, text_list = batch
            speaker_list = speaker_list.tolist()
            emotion_list = emotion_list.tolist()
            mel_predictions = get_inference(text_list, speaker_list, emotion_list)
            save_paths = save_predicted_mels(mel_predictions, emotion_list, speaker_list, basename_list)

            for basename, mel_path in zip(basename_list, save_paths):
                du = {
                    'basename' : basename,
                    'mel_path' : mel_path
                }
                UNIVERSAL_DU_MEL_LIST.append(du)
        except Exception as e:
            print(e)
            continue