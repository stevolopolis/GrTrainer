import os
import datetime
import torch

from parameters import Params

params = Params()


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


def epoch_logger(network_name, epoch, train_loss, val_loss, train_acc, val_acc):
    filename = '%s_log.txt' % network_name
    with open(os.path.join(params.LOG_PATH, filename), 'a') as f:
        train_loss_mean = sum(train_loss) / len(train_loss)
        val_loss_mean = sum(val_loss) / len(val_loss)
        log_message = 'Epoch: %s -- Train Loss: %s -- Train Acc: %s -- Val Loss: %s -- Val Acc: %s' \
                        % (epoch,
                           round(train_loss_mean, 4), train_acc,
                           round(val_loss_mean, 4), val_acc)
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
