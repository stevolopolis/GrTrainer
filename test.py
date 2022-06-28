import torch
import os

from parameters import Params
from inference.models.grconvnet_cls import GrCLS
from inference.models.alexnet import AlexNet, PretrainedAlexnet, myAlexNet
from evaluation import get_test_acc, get_grasp_acc, visualize_grasp

params = Params()

model_name = params.MODEL_NAME
weights_dir = params.MODEL_PATH
for epoch in range(46, 47):
    weights_path = os.path.join(weights_dir, model_name, model_name + '_epoch%s.pth' % epoch)

    #model = GrCLS().to(params.DEVICE)
    #model = AlexNet().to(params.DEVICE)
    #model = myAlexNet().to(params.DEVICE)
    model = PretrainedAlexnet(n_cls=params.NUM_CLASS).to(params.DEVICE)
    model.load_state_dict(torch.load(weights_path))
    model.eval()

    #accuracy, loss = get_test_acc(model)
    #accuracy, loss = get_grasp_acc(model)
    #print('Epoch: %s' % epoch, accuracy, loss)
    visualize_grasp(model)
