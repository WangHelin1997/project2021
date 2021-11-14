import os
import torch
import argparse
from net import *
import librosa
import numpy as np

class Detection():
    def __init__(self, model_pth, gpuid):
        super(Detection, self).__init__()
        model = net(16000, 1024, 320, 64, 50, 8000, 2, False)
        dicts = torch.load(model_pth, map_location='cpu')
        model.load_state_dict(dicts["model_state_dict"])
        self.gpuid = tuple(gpuid)
        self.device = torch.device('cuda:{}'.format(gpuid[0]) if len(gpuid) > 0 else 'cpu')
        self.model = model.to(self.device)

    def inference(self, file_path): #
        self.model.eval()
        with torch.no_grad():
            (audio, _) = librosa.core.load(file_path, sr=16000, mono=True)
            print("Compute on utterance {}...".format(file_path))
            audio = torch.from_numpy(audio).to(self.device)
            if audio.dim() == 1:
                audio = torch.unsqueeze(audio, 0)
            out = self.model(audio)
            if out[0, 0] > 0.5:
                return True
            else:
                return False

    def test(self, file_path, threshold=0.5):
        self.model.eval()
        data_list = []
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for root, dirs, files in os.walk(file_path):
            for name in files:
                file = os.path.join(root, name)
                data_list.append(file)
        for file in data_list:
            with torch.no_grad():
                label = int(file[-5])
                (audio, _) = librosa.core.load(file, sr=16000, mono=True)
                audio = torch.from_numpy(audio).to(self.device)
                if audio.dim() == 1:
                    audio = torch.unsqueeze(audio, 0)
                out = self.model(audio)
                # print(out,label)
                # assert 1==2
                if out[0, 0] > threshold: # 大于预测为正类
                    print("Compute on utterance {}: True".format(file))
                else:
                    print("Compute on utterance {}: False".format(file))
                if out[0, 0] > threshold and label < threshold:# label 0, predict 1
                    FP += 1.0
                elif out[0, 0] > threshold and label > threshold:# label 1, predict 1
                    TP += 1.0
                elif out[0, 0] < threshold and label > threshold:# label 1, predict 0
                    FN += 1.0
                elif out[0, 0] > threshold and label > threshold:# label 0, predict 0
                    TN += 1.0
        Precision = TP/(TP + FP) 
        Recall = TP/(TP+FN)
        
        # FRR = FR/(TR + FR)
        ACC = (TP + TN)/(FP+TP+FN+TN)
        print('Precision ',Precision)
        print('Recall ',Recall)
        print('ACC: {}'.format(ACC))
        print('True Acceptance: {}'.format(TP))
        print('False Acceptance: {}'.format(FP))
        print('True Rejection: {}'.format(TN))
        print('False Rejection: {}'.format(FN))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-model_pth', type=str, default='/home/pkusz/home/PKU_team/pku_code/checkpoint/net/best.pt', help="Path to model file.")
    parser.add_argument(
        '-gpuid', type=str, default='0', help='Enter GPU id number')
    parser.add_argument(
        '-file_path', type=str,
        default='/home/pkusz/home/PKU_team/guangchang/data_splits/2021-02-25-20-17-52-2.4G/2021-02-25-20-17-52-2.4G.wav_260_0.wav',
        help='test file path')
    parser.add_argument(
        '-test_path', type=str,
        default='/home/pkusz/home/PKU_team/new_data/test',
        help='test files path')
    parser.add_argument(
        '-test_single_file', type=bool,
        default=False,
        help='whether test single file')

    args = parser.parse_args()
    gpuid = [int(i) for i in args.gpuid.split(',')]
    separation = Detection(args.model_pth, gpuid)
    # if args.test_single_file:
    #     print(separation.inference(args.file_path))
    # else:
    #     data_list = []
    #     for root, dirs, files in os.walk(args.test_path):
    #         for name in files:
    #             file = os.path.join(root, name)
    #             data_list.append(file)
    #     for file in data_list:
    #         print(separation.inference(file))
    separation.test(args.test_path)

if __name__ == "__main__":
    main()

