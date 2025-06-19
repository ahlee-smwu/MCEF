import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle, pandas as pd
import numpy as np

class IEMOCAPDataset(Dataset):
    def __init__(self, train=True):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
        self.roberta2, self.roberta3, self.roberta4, \
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        self.testVid = pickle.load(open('data/iemocap_multimodal_features.pkl', 'rb'), encoding='latin1')
        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)

    # org ver.
    def __getitem__(self, index):
        vid = self.keys[index]
        
        return torch.FloatTensor(self.videoText[vid]),\
               torch.FloatTensor(self.videoVisual[vid]),\
               torch.FloatTensor(self.videoAudio[vid]),\
               torch.FloatTensor([[1,0] if x=='M' else [0,1] for x in\
                                  self.videoSpeakers[vid]]),\
               torch.FloatTensor([1]*len(self.videoLabels[vid])),\
               torch.LongTensor(self.videoLabels[vid]),\
               vid

    # fixed ver. Nan
    '''def __getitem__(self, index):
        vid = self.keys[index]

        # safer conversion
        text = np.array(self.videoText[vid])
        visual = np.array(self.videoVisual[vid])
        audio = np.array(self.videoAudio[vid])

        # NaN check
        if np.isnan(text).any():
            print(f"❌ NaN in videoText[{vid}]")
        if np.isnan(visual).any():
            print(f"❌ NaN in videoVisual[{vid}]")
        if np.isnan(audio).any():
            print(f"❌ NaN in videoAudio[{vid}]")

        return torch.from_numpy(text).float(), \
            torch.from_numpy(visual).float(), \
            torch.from_numpy(audio).float(), \
            torch.FloatTensor([[1, 0] if x == 'M' else [0, 1] for x in self.videoSpeakers[vid]]), \
            torch.FloatTensor([1] * len(self.videoLabels[vid])), \
            torch.LongTensor(self.videoLabels[vid]), \
            vid'''

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<4 else pad_sequence(dat[i], True) if i<6 else dat[i].tolist() for i in dat]


class MELDDataset(Dataset):
    def __init__(self, path, train=True):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText, \
        self.roberta2, self.roberta3, self.roberta4, \
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        self.testVid, _ = pickle.load(open(path, 'rb'))

        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.videoText[vid]),\
               torch.FloatTensor(self.videoVisual[vid]),\
               torch.FloatTensor(self.videoAudio[vid]),\
               torch.FloatTensor(self.videoSpeakers[vid]),\
               torch.FloatTensor([1]*len(self.videoLabels[vid])),\
               torch.LongTensor(self.videoLabels[vid]),\
               vid

    def __len__(self):
        return self.len

    def return_labels(self):
        return_label = []
        for key in self.keys:
            return_label+=self.videoLabels[key]
        return return_label

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<4 else pad_sequence(dat[i], True) if i<6 else dat[i].tolist() for i in dat]