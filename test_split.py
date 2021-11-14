"""
@authors: Zhongjie Ye, Dongchao Yang, Helin Wang
@Introduction: Randomly Select Test data from Training data.
"""

import os
import random
import shutil
import argparse

def main(args):
    data_path = args.data_path
    ls = os.listdir(os.path.join(data_path, 'train'))
    if not os.path.isdir(os.path.join(data_path, 'test')):
        os.mkdir(os.path.join(data_path, 'test'))
    one_num = args.test_sample_num // 2
    zero_num = args.test_sample_num - one_num
    one_ls = []
    zero_ls = []
    for a in ls:
        label = int(a[-5])
        if label == 0:
            zero_num += 1
            zero_ls.append(a)
        else:
            one_num += 1
            one_ls.append(a)

    print(len(zero_ls))
    print(len(one_ls))

    tmp_one = random.sample(one_ls, one_num)
    tmp_zero = random.sample(zero_ls, zero_num)

    for a in tmp_one:
        full_path = os.path.join(data_path, 'train', a)
        despath = os.path.join(data_path, 'test', a)
        shutil.move(full_path, despath)

    for a in tmp_zero:
        full_path = os.path.join(data_path, 'train', a)
        despath = os.path.join(data_path, 'test', a)
        shutil.move(full_path, despath)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Randomly Select Test data from Training data.')
    parser.add_argument('--data_path', type=str) # /home/pkusz/home/PKU_team/new_data/
    parser.add_argument('--test_sample_num', type=int, default=800)
    args = parser.parse_args()
    main(args)
