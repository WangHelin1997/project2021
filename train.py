"""
@authors: Helin Wang
@Introduction: Train
"""

from torch.utils.data import DataLoader as Loader
from dataset import Datasets
from net import *
import argparse
import torch
import trainer
import torch.optim.lr_scheduler as lr_scheduler


def make_dataloader(args):
    # make train's dataloader
    train_dataset = Datasets(args.data_path)
    train_dataloader = Loader(train_dataset,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              shuffle=True)
    test_dataset = Datasets(args.data_path_test)
    test_dataloader = Loader(test_dataset,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              shuffle=True)
    return train_dataloader, test_dataloader

def make_optimizer(params, args):
    optimizer = getattr(torch.optim, 'Adam')
    optimizer = optimizer(params, lr=args.lr, weight_decay=args.wd)
    return optimizer


def train():
    parser = argparse.ArgumentParser(
        description='Parameters for training.')
    parser.add_argument('--data_path', type=str, default='/home/pkusz/home/PKU_team/new_data/train', help='Path of wave file.')
    parser.add_argument('--data_path_test', type=str, default='/home/pkusz/home/PKU_team/new_data/test',
                        help='Path of test wave file.')
    parser.add_argument('--epoch', type=int, default=60, help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers.')
    parser.add_argument('--print_freq', type=int, default=50, help='Number of print frequency.')
    parser.add_argument('--clip_norm', type=int, default=200, help='Number of clip norm.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning Rate.')
    parser.add_argument('--wd', type=float, default=1e-5, help='Weight decay.')
    parser.add_argument('--checkpoint', type=str, default='/home/pkusz/home/PKU_team/pku_code/checkpoint', help='Path to save model.')
    parser.add_argument('--cuda', type=bool, default=True, help='Using gpu.')
    parser.add_argument('--pretrain', type=bool, default=True, help='Using pretrained model.')
    parser.add_argument('--pretrained_checkpoint_path', type=str, default='/home/pkusz/home/PKU_team/guangchang/Cnn10_mAP=0.380.pth',
                        help='Path of pretrained checkpoint.')
    args = parser.parse_args()

    # build model
    print("Building the model")
    model = net(16000, 1024, 320, 64, 50, 8000, 2, False)
    # Load pretrained model
    if args.pretrain:
        print('Load pretrained model from {}'.format(args.pretrained_checkpoint_path))
        model.load_from_pretrain(args.pretrained_checkpoint_path)
    # build optimizer
    print("Building the optimizer")
    optimizer = make_optimizer(model.parameters(), args)
    # build dataloader
    print('Building the dataloader')
    train_dataloader, test_dataloader = make_dataloader(args)
    print('Train Datasets Length: {}'.format(len(train_dataloader)))
    print('Test Datasets Length: {}'.format(len(test_dataloader)))
    # build scheduler
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.epoch)
    # build trainer
    print('Building the Trainer')
    Trainer = trainer.Trainer(train_dataloader, test_dataloader, model, optimizer, scheduler, args)
    Trainer.run()


if __name__ == "__main__":
    train()


