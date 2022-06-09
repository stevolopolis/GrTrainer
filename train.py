import os
import torch 
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.nn import BCELoss, L1Loss

from tqdm import tqdm

from paths import Path
from parameters import Params
from data_loader import DataLoader
from inference.models.alexnet import AlexNet
from inference.models.grconvnet_cls import GrCLS
from utils import epoch_logger, log_writer, get_correct_preds, get_acc

params = Params() 
paths = Path()

# Create <trained-models> directory
paths.create_model_path()
# Create directory for training logs
paths.create_log_path()
# Create subdirectory in <logs> for current model
paths.create_model_log_path()

# Load model
model = GrCLS().to(params.DEVICE)
#model = AlexNet().to(params.DEVICE)
#checkpoint_path = os.path.join(params.MODEL_PATH, 'alexnetCLS_v4', 'alexnetCLS_v4_epoch49.pth')
#model.load_state_dict(torch.load(checkpoint_path))
# Create DataLoader class
data_loader = DataLoader(params.TRAIN_PATH, params.BATCH_SIZE, params.TRAIN_VAL_SPLIT)
# Get number of training/validation steps
n_train, n_val = data_loader.get_train_val()

# Training utils
optim = Adam(model.parameters(), lr=params.LR)
criterion = BCELoss()

for epoch in tqdm(range(1, params.EPOCHS + 1)):
    train_history = []
    val_history = []
    train_total = 0
    train_correct = 0
    val_total = 0
    val_correct = 0
    for step, (img, label) in enumerate(data_loader.load_batch()):
        optim.zero_grad()
        output = model(img)
        output = F.softmax(output, dim=1)
        
        label = F.one_hot(label, num_classes=params.NUM_CLASS).to(params.DEVICE)
        label = label.type(torch.float32)
        loss = criterion(output, label)

        if step < n_train:
            loss.backward()
            optim.step()

            log_writer(params.MODEL_NAME, epoch, step, loss.item(), train=True)
            train_history.append(loss.item())
            correct, total = get_correct_preds(output, label)
            train_correct += correct
            train_total += total
        else:
            log_writer(params.MODEL_NAME, epoch, step, loss.item(), train=False)
            val_history.append(loss.item())
            correct, total = get_correct_preds(output, label)
            val_correct += correct
            val_total += total
    
    # Get training and validation accuracies
    train_acc = get_acc(train_correct, train_total)
    val_acc = get_acc(val_correct, val_total)
    epoch_logger(params.MODEL_NAME, epoch, train_history, val_history, train_acc, val_acc)
    torch.save(model.state_dict(), os.path.join(params.MODEL_LOG_PATH, f"{params.MODEL_NAME}_epoch{epoch}.pth"))

torch.save(model.state_dict(), os.path.join(params.MODEL_LOG_PATH, f"{params.MODEL_NAME}_final.pth"))