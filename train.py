import os
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD

from tqdm import tqdm

from paths import Path
from parameters import Params
from data_loader import DataLoader
from inference.models.alexnet import AlexNet, PretrainedAlexnet
from inference.models.grconvnet_cls import GrCLS
from utils import epoch_logger, log_writer, get_correct_preds, get_acc
from grasp_utils import get_correct_grasp_preds
from evaluation import get_test_acc
from loss import BCEL1Loss, CLSLoss, WeightedL2Loss

params = Params() 
paths = Path()

# Create <trained-models> directory
paths.create_model_path()
# Create directory for training logs
paths.create_log_path()
# Create subdirectory in <logs> for current model
paths.create_model_log_path()

# Load model
#model = GrCLS().to(params.DEVICE)
#model = AlexNet().to(params.DEVICE)
model = PretrainedAlexnet(n_cls=params.NUM_CLASS).to(params.DEVICE)
#checkpoint_path = os.path.join(params.MODEL_PATH, 'alexnetCls_raw_v1', 'alexnetCls_raw_v1_epoch80.pth')
#model.load_state_dict(torch.load(checkpoint_path))
# Create DataLoader class
data_loader = DataLoader(params.TRAIN_PATH, params.BATCH_SIZE, params.TRAIN_VAL_SPLIT)
# Get number of training/validation steps
n_train, n_val = data_loader.get_train_val()

# Training utils
optim = Adam(model.parameters(), lr=params.LR)
scheduler = torch.optim.lr_scheduler.StepLR(optim, 25, 0.1)
criterion = nn.CrossEntropyLoss()
grasp_criterion = nn.MSELoss()

for epoch in tqdm(range(1, params.EPOCHS + 1)):
    train_history = []
    val_history = []
    train_total = 0
    train_correct = 0
    val_total = 0
    val_correct = 0
    #for step, (img, label) in enumerate(data_loader.load_batch()):
    for step, (img, label, candidates) in enumerate(data_loader.load_grasp_batch()):
        optim.zero_grad()
        output = model(img)
        #output = F.sigmoid(output)
       
        #loss = criterion(output, label)
        loss = grasp_criterion(output, label)
        #label = F.one_hot(label, num_classes=params.NUM_CLASS)
        #label = label.type(torch.float32)

        if step < n_train:
            loss.backward()
            optim.step()

            log_writer(params.MODEL_NAME, epoch, step, loss.item(), train=True)
            train_history.append(loss.item())
            #correct, total = get_correct_preds(output, label)
            correct, total = get_correct_grasp_preds(output, candidates)
            train_correct += correct
            train_total += total
        else:
            log_writer(params.MODEL_NAME, epoch, step, loss.item(), train=False)
            val_history.append(loss.item())
            #correct, total = get_correct_preds(output, label)
            correct, total = get_correct_grasp_preds(output, candidates)
            val_correct += correct
            val_total += total
    
    test_acc, test_loss = 0, 0  # get_test_acc(model)
    scheduler.step()

    # Get training and validation accuracies
    train_acc = get_acc(train_correct, train_total)
    val_acc = get_acc(val_correct, val_total)
    epoch_logger(params.MODEL_NAME, epoch, train_history, val_history, test_loss, train_acc, val_acc, test_acc)
    torch.save(model.state_dict(), os.path.join(params.MODEL_LOG_PATH, f"{params.MODEL_NAME}_epoch{epoch}.pth"))

torch.save(model.state_dict(), os.path.join(params.MODEL_LOG_PATH, f"{params.MODEL_NAME}_final.pth"))