import os
import datetime
import torch

from parameters import Params

params = Params()


class AddGaussianNoise(torch.nn.Module):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

"""from PIL import Image
import numpy as np
import cv2
from torchvision import transforms
import torch.nn as nn
aug = AddGaussianNoise(0, .025)
img = Image.open('data/top_5/train/plants/2a7d62b731a04f5fa54b9afa882a89ed/0_2a7d62b731a04f5fa54b9afa882a89ed_RGB.png')
img = np.array(img)
img = img / 255
cv2.imshow('file', img)
cv2.waitKey(0)
img = torch.tensor(img)
img = torch.moveaxis(img, -1, 0)
random_transforms = transforms.RandomApply(nn.ModuleList([AddGaussianNoise(0, .1)]), p=0.25)
trans = transforms.Compose([
    transforms.RandomResizedCrop((128, 128), scale=(.5, .6), ratio=(1, 1))
])
img = trans(img)
img = torch.moveaxis(img, 0, -1)
img = img.numpy()
print(img.shape)
img = np.clip(img, 0, 1)
print(np.max(img), np.min(img))
cv2.imshow('file', img)
cv2.waitKey(0)"""

def log_writer(network_name, epoch, step, loss, train=True):
    """Writes training losses into a log.txt file."""
    filename = '%s_log.txt' % network_name
    if filename not in os.listdir(params.LOG_PATH):
        with open(os.path.join(params.LOG_PATH, filename), 'w') as f:
            initial_message = '%s Training Log at %s' % (network_name, datetime.datetime.now())
            f.write(initial_message + '\n')
    else:
        with open(os.path.join(params.LOG_PATH, filename), 'a') as f:
            if train:
                log_message = 'train-epoch-step: %s-%s -- Loss: %s' % (epoch, step, loss)
            else:
                log_message = 'val-epoch-step: %s-%s -- Loss: %s' % (epoch, step, loss)
            f.write('\n')
            f.write(log_message)


def epoch_logger(network_name, epoch, train_loss, val_loss, test_loss, train_acc, val_acc, test_acc):
    filename = '%s_log.txt' % network_name
    with open(os.path.join(params.LOG_PATH, filename), 'a') as f:
        train_loss_mean = sum(train_loss) / len(train_loss)
        val_loss_mean = sum(val_loss) / len(val_loss)
        log_message = 'Epoch: %s -- Train Loss: %s -- Train Acc: %s -- Val Loss: %s -- Val Acc: %s\n \
                        Test Loss: %s -- Test Acc: %s' \
                        % (epoch,
                           round(train_loss_mean, 4), train_acc,
                           round(val_loss_mean, 4), val_acc,
                           test_loss, test_acc)
        f.write('\n')
        f.write(log_message)


def get_correct_preds(output, label):
    """Return number of correct predictions out of number of instances."""
    _, prediction = torch.max(output.data, 1)
    _, ground_truth = torch.max(label.data, 1)
    total = label.size(0)
    correct = (prediction == ground_truth).sum().item()
    return correct, total


def get_acc(correct, total):
    """Return accuracy given number of correct predictions and total 
    number of predictions."""
    return round(100 * correct / total, 2)


def model_test(model, test_path):
    """Return testing results of model using data in <test_path>."""
    pass
