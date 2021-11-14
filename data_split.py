"""
@authors: Zhongjie Ye, Dongchao Yang, Helin Wang
@Introduction: Split data according to the annotation.
"""

import librosa
import os
import pickle
import soundfile as sf
import argparse

IMG_EXTENSIONS = ['.wav', '.mp3']

def is_wav_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def main(args):
    names = [args.names]
    data_split_root = os.path.join(args.data_split_root, 'train')
    if not os.path.isdir(data_split_root):
        os.mkdir(os.path.join(data_split_root))
    segment_label = pickle.load(open("gcw.p", "rb"))
    sample_rate = args.sample_rate
    split_lens = args.audio_length * args.sample_rate
    data_list = []
    for roots in names:
        for root, _, fnames in sorted(os.walk(roots , followlinks=True)):
            for fname in fnames:
                if is_wav_file(fname):
                    wav_dir = os.path.join(root, fname)
                    data_save = data_split_root
                    segment = segment_label[fname]
                    data, sampling_rate= librosa.load(wav_dir,sample_rate)
                    print(fname)
                    count = 0
                    for start_time, end_time, label in segment:
                        start_time = start_time * sample_rate
                        end_time = end_time * sample_rate
                        print(start_time, end_time, label)
                        save_nums = int((end_time - start_time) / split_lens)
                        start_time = int(start_time)
                        for i in range(save_nums):
                            name = fname[:-4] + "_{}_{}.wav".format(str(count),str(label))
                            save_name = os.path.join(data_save, name)
                            split_data = data[start_time+i*split_lens:start_time+(i+1)*split_lens]
                            if split_data.shape[0] != split_lens:
                                print("no match split lens ", split_data.shape[0])
                                continue
                            sf.write(save_name, split_data, sample_rate)
                            data_list.append(save_name)
                            count += 1
    print(len(data_list))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split data according to the annotation.')
    parser.add_argument('--names', type=str, default='/home/pkusz/home/PKU_team/gcw')
    parser.add_argument('--data_split_root', type=str)  # /home/pkusz/home/PKU_team/new_data/
    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument('--audio_length', type=int, default=5)
    args = parser.parse_args()
    main(args)
