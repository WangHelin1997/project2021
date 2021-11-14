import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import random
import numpy as np
import os
import librosa
noisy_ls = os.listdir('/home/pkusz/home/PKU_team/guangchang/audio')

class Datasets(Dataset):
    '''
       Load audio data
       mix_scp: file path of mix audio (type: str)
       ref_scp: file path of ground truth audio (type: list[spk1,spk2])
    '''

    def __init__(self, data_path=None, sr=16000, length=5):
        super(Datasets, self).__init__()
        self.data_list = []
        self.sr = sr
        self.length = length
        for root, dirs, files in os.walk(data_path):
            for name in files:
                file = os.path.join(root, name)
                self.data_list.append(file)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        # nosiy_audio = random.sample(noisy_ls,5)
        # nosiy_audio = '/home/pkusz/home/PKU_team/guangchang/audio/' + nosiy_audio[0]
        # (audio_n, _n) = librosa.core.load(nosiy_audio, sr=self.sr, mono=True)
        (audio, _) = librosa.core.load(self.data_list[index], sr=self.sr, mono=True)
        audio = self.pad_or_truncate(audio)
        # shape1 = audio.shape[0]
        # print(shape1)
        # assert 1==2
        # audio = audio + audio_n[:shape1]

        label = int(self.data_list[index][-5])

        # target = np.zeros(2)
        # if label < 0.5:
        #     target[0] = 1.
        # else:
        #     target[1] = 1.
        target = np.ones(1)
        if label < 0.5:
            target[0] = 0.

        return audio, target

    def pad_or_truncate(self, x):
        """Pad all audio to specific length."""
        audio_length = round(self.sr * self.length)
        if len(x) <= audio_length:
            return np.concatenate((x, np.zeros(audio_length - len(x))), axis=0)
        else:
            return x[0: audio_length]

class Datasets_donchao(Dataset):
    '''
       Load audio data
       mix_scp: file path of mix audio (type: str)
       ref_scp: file path of ground truth audio (type: list[spk1,spk2])
    '''

    def __init__(self, data_path=None, sr=16000, length=5): # the data_path is txt file path, it store the real path of audio
        super(Datasets, self).__init__()
        self.data_list = []
        self.sr = sr
        self.length = length
        file_obj = open(data_path,'r')
        for line in file_obj:
            self.data_list.append(line)
        # for root, dirs, files in os.walk(data_path):
        #     for name in files:
        #         file = os.path.join(root, name)
        #         self.data_list.append(file)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        (audio, _) = librosa.core.load(self.data_list[index], sr=self.sr, mono=True)
        audio = self.pad_or_truncate(audio)

        label = int(self.data_list[index][-5])
        print(self.data_list[index])
        print(label)
        assert 1==2

        # target = np.zeros(2)
        # if label < 0.5:
        #     target[0] = 1.
        # else:
        #     target[1] = 1.
        target = np.zeros(1)
        if label < 0.5:
            target[0] = 1.

        return audio, target

    def pad_or_truncate(self, x):
        """Pad all audio to specific length."""
        audio_length = round(self.sr * self.length)
        if len(x) <= audio_length:
            return np.concatenate((x, np.zeros(audio_length - len(x))), axis=0)
        else:
            return x[0: audio_length]

if __name__ == "__main__":
    data_path='/home/pkusz/home/PKU_team/guangchang/data_splits'
    datasets = Datasets(data_path, 16000, 5)
    print(datasets.data_list[:5])
    print(len(datasets.data_list))


