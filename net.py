"""
@authors: Helin Wang
"""

from models import *
import torch
import torch.nn as nn
import torch.nn.functional as F

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)

class net(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin,
                 fmax, classes_num, freeze_base):
        """Classifier for a new task using pretrained Cnn10 as a sub module.
        """
        super(net, self).__init__()
        audioset_classes_num = 527

        self.base = Cnn10(sample_rate, window_size, hop_size, mel_bins, fmin,
                          fmax, audioset_classes_num)

        # Transfer to another task layer
        self.fc_transfer = nn.Linear(512, 256, bias=True)
        self.fc_out = nn.Linear(256,classes_num,bias=True)
        self.relu = nn.ReLU()

        if freeze_base:
            # Freeze AudioSet pretrained layers
            for param in self.base.parameters():
                param.requires_grad = False

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_transfer)

    def load_from_pretrain(self, pretrained_checkpoint_path):
        checkpoint = torch.load(pretrained_checkpoint_path)
        self.base.load_state_dict(checkpoint['model'])

    def forward(self, input, mixup_lambda=None):
        """Input: (batch_size, data_length)
        """
        output_dict = self.base(input, mixup_lambda)
        embedding = output_dict['embedding']
        embedding2 = self.relu(self.fc_transfer(embedding))

        output = torch.softmax(self.fc_out(embedding2), dim=-1)
        # print(output)
        output = output[:, 0]
        return output[:, None]
        #return output

