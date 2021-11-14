"""
@authors: Helin Wang
@Introduction: Trainer
"""

import time
from loss import get_loss,focal_loss
import torch
import os
import matplotlib.pyplot as plt
from torch.nn.parallel import data_parallel
from sklearn import metrics
import numpy as np

def check_parameters(net):
    '''
        Returns module parameters. Mb
    '''
    parameters = sum(param.numel() for param in net.parameters())
    return parameters / 10**6

class Trainer(object):
    def __init__(self, train_dataloader, test_dataloader, model, optimizer, scheduler, args):
        super(Trainer).__init__()
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.scheduler = scheduler
        self.cur_epoch = 0
        self.total_epoch = args.epoch
        self.print_freq = args.print_freq
        self.checkpoint = args.checkpoint
        self.name = 'net'
        self.cuda = args.cuda

        if args.cuda:
            print('Load Nvida GPU .....')
            self.device = torch.device('cuda:0')
            self.model = model.to(self.device)
            print('Loading model parameters: {:.3f} Mb'.format(check_parameters(self.model)))
        else:
            print('Load CPU ...........')
            self.device = torch.device('cpu')
            self.model = model.to(self.device)
            print('Loading model parameters: {:.3f} Mb'.format(check_parameters(self.model)))

        self.optimizer = optimizer
        self.clip_norm = args.clip_norm
        print("Gradient clipping by {}, default L2".format(self.clip_norm))

    def train(self, epoch):
        print('Start training from epoch: {:d}, iter: {:d}'.format(epoch, 0))
        self.model.train()
        num_batchs = len(self.train_dataloader)
        total_loss = 0.0
        total_acc = 0.0
        num_index = 1
        start_time = time.time()
        for audios, labels in self.train_dataloader:
            audios = audios.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)
            audios = audios.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()

            out = self.model(audios)

            epoch_loss = get_loss(out, labels)
            # print(epoch_loss)
            # total_loss += epoch_loss
            total_loss += epoch_loss.item()

            prob = out.cpu().detach()
            # Evaluate
            count_nums = prob.shape[0]
            count = 0.
            for i in range(count_nums):
                if prob[i, 0] < 0.5 and labels[i, 0] < 0.5:
                    count += 1.
                elif prob[i, 0] > 0.5 and labels[i, 0] > 0.5:
                    count += 1.

            classwise_accuracy = count / count_nums
            total_acc += classwise_accuracy
            epoch_loss.backward()

            if self.clip_norm:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.clip_norm)

            self.optimizer.step()
            if num_index % self.print_freq == 0:
                message = '<epoch:{:d}, iter:{:d}, lr:{:.3e}, total_loss:{:.3f}, total_acc:{:.3f}>'.format(
                    epoch, num_index, self.optimizer.param_groups[0]['lr'], total_loss / num_index, total_acc / num_index)
                print(message)
            num_index += 1
        end_time = time.time()
        total_loss = total_loss / num_index
        total_acc = total_acc / num_index
        message = 'Finished *** <epoch:{:d}, iter:{:d}, lr:{:.3e}, loss:{:.3f}, acc:{:.3f}, Total time:{:.3f} min> '.format(
            epoch, num_index, self.optimizer.param_groups[0]['lr'], total_loss, total_acc, (end_time - start_time) / 60)
        print(message)
        return total_loss, total_acc


    def test(self, epoch):
        print('Start testing from epoch: {:d}, iter: {:d}'.format(epoch, 0))
        self.model.eval()
        num_batchs = len(self.test_dataloader)
        total_loss = 0.0
        total_acc = 0.0
        num_index = 1
        start_time = time.time()
        for audios, labels in self.test_dataloader:
            audios = audios.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)
            audios = audios.to(self.device)
            labels = labels.to(self.device)

            out = self.model(audios)

            epoch_loss = get_loss(out, labels)
            total_loss += epoch_loss.item()
            # total_loss += epoch_loss

            prob = out.cpu().detach()
            # Evaluate
            count_nums = prob.shape[0]
            count = 0.
            for i in range(count_nums):
                if prob[i, 0] < 0.5 and labels[i, 0] < 0.5:
                    count += 1.
                elif prob[i, 0] > 0.5 and labels[i, 0] > 0.5:
                    count += 1.

            classwise_accuracy = count / count_nums
            total_acc += classwise_accuracy

            if num_index % self.print_freq == 0:
                message = '<epoch:{:d}, iter:{:d}, lr:{:.3e}, total_loss:{:.3f}, total_acc:{:.3f}>'.format(
                    epoch, num_index, self.optimizer.param_groups[0]['lr'], total_loss / num_index, total_acc / num_index)
                print(message)
            num_index += 1
        end_time = time.time()
        total_loss = total_loss / num_index
        total_acc = total_acc / num_index
        message = 'Finished *** <epoch:{:d}, iter:{:d}, lr:{:.3e}, loss:{:.3f}, acc:{:.3f}, Total time:{:.3f} min> '.format(
            epoch, num_index, self.optimizer.param_groups[0]['lr'], total_loss, total_acc, (end_time - start_time) / 60)
        print(message)
        return total_loss, total_acc


    def run(self):
        train_loss = []
        train_acc = []
        test_loss = []
        test_acc = []

        self.save_checkpoint(self.cur_epoch, best=False)
        best_loss = 100.
        best_acc = 0.
        # starting training part
        while self.cur_epoch < self.total_epoch:
            self.cur_epoch += 1
            t_loss, t_acc = self.train(self.cur_epoch)
            tt_loss, tt_acc = self.test(self.cur_epoch)
            train_loss.append(t_loss)
            train_acc.append(t_acc)
            test_loss.append(tt_loss)
            test_acc.append(tt_acc)
            # schedule here
            self.scheduler.step()

            if tt_acc <= best_acc:
                print('No improvement, Best Test Loss: {:.4f}, Best Test Acc: {:.4f}'.format(best_loss, best_acc))
            else:
                best_acc = tt_acc
                best_loss = tt_loss
                self.save_checkpoint(self.cur_epoch, best=True)
                print('Epoch: {:d}, Now Best Loss Change: {:.4f}, Now Best Acc: {:.4f}'.format(self.cur_epoch, tt_loss, tt_acc))

        self.save_checkpoint(self.cur_epoch, best=False)
        print("Training for {:d}/{:d} epoches done!".format(
            self.cur_epoch, self.total_epoch))

        # draw loss image
        plt.title("Loss and Acc")
        x = [i for i in range(self.cur_epoch)]
        plt.plot(x, train_loss, 'b-', label=u'train_loss', linewidth=0.8)
        plt.plot(x, train_acc, 'b', label=u'train_acc', linewidth=0.8)
        plt.plot(x, test_loss, 'y', label=u'test_loss', linewidth=0.8)
        plt.plot(x, test_acc, 'r', label=u'test_acc', linewidth=0.8)
        plt.legend()
        plt.ylabel('acc')
        plt.xlabel('epoch')
        plt.savefig('loss.png')



    def save_checkpoint(self, epoch, best=True):
        '''
           save model
           best: the best model
        '''
        os.makedirs(os.path.join(self.checkpoint, self.name), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optim_state_dict': self.optimizer.state_dict()
        },
            os.path.join(self.checkpoint, self.name, '{0}.pt'.format('best' if best else 'last')))


