import torch
import os

import torch.nn.functional as F

from parameters import Params
from inference.models.grconvnet_cls import GrCLS
from data_loader import DataLoader
from utils import get_correct_preds, get_acc

params = Params()

model_name = params.MODEL_NAME
weights_dir = params.MODEL_PATH
weights_path = os.path.join(weights_dir, model_name + '_epoch14.pth')

model = GrCLS().to(params.DEVICE)
model.load_state_dict(torch.load(weights_path))
model.eval()

data_loader = DataLoader(params.TEST_PATH, 2, params.TRAIN_VAL_SPLIT)

correct = 0
total = 0
for (img, label) in data_loader.load_batch():
    output = F.softmax(model(img), dim=1)
    label = F.one_hot(label, num_classes=30).to(params.DEVICE)
    batch_correct, batch_total = get_correct_preds(output, label)
    correct += batch_correct
    total += batch_total

    accuracy = get_acc(correct, total)

accuracy = get_acc(correct, total)
print(accuracy)
