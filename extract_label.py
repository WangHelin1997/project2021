"""
@authors: Zhongjie Ye, Dongchao Yang, Helin Wang
@Introduction: Get annotation.
"""

import os
import re
import pickle
import argparse

IMG_EXTENSIONS = ['.wav', '.mp3']

def is_wav_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def main(args):
    names = [args.names]
    save_dict = {}
    for roots in names:
        for root, _, fnames in sorted(os.walk(roots , followlinks=True)):
            for fname in fnames:
                if is_wav_file(fname):
                    full_path = os.path.join(root,fname[:-4]+".txt")
                    print(full_path)

                    with open(full_path, encoding='utf-8') as f:
                        filepaths_and_text = [line.strip()for line in f]
                    times = []
                    labels = []
                    flag = False
                    end_time = 1800.0
                    for line in filepaths_and_text:
                        if line[:5] == "<Turn":
                            end_time = re.findall(r"\d+\.?\d*",line)[-1]
                            end_time = float(end_time)
                        if flag:
                            label = int(line)
                            labels.append(label)
                            flag = False
                        if line[:5] == "<Sync":
                            time = re.findall(r"\d+\.?\d*",line)[0]
                            time = float(time)
                            times.append(time)
                            flag = True
                    times.append(end_time)
                    lens = len(labels)
                    save_tuple = []
                    for i in range(lens):
                        start_time = times[i]
                        end_time = times[i+1]
                        label = labels[i]
                        save_tuple.append((start_time,end_time,label))
                    save_dict[fname] = save_tuple
    with open("gcw.p","wb") as f:
        pickle.dump(save_dict, f)
    print(save_dict)
    print(len(save_dict.keys()))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get annotation.')
    parser.add_argument('--names', type=str, default='/home/pkusz/home/PKU_team/gcw')
    args = parser.parse_args()
    main(args)

