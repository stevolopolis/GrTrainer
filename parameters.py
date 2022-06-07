import torch

class Params:
    """
    Parameters for AM visualization on gr-convnet
    """
    def __init__(self):
        # Model name
        self.MODEL_NAME = 'grconvCLS'

        # device: cpu / gpu
        self.DEVICE = torch.device('cpu') if torch.cuda.is_available \
                                      else torch.device('cpu')
        # Training params
        self.OUTPUT_SIZE = 64
        self.IMG_SIZE = (4, self.OUTPUT_SIZE, self.OUTPUT_SIZE) 
        self.EPOCHS = 50
        self.LR = 1e-4
        self.BATCH_SIZE = 64
        self.TRAIN_VAL_SPLIT = 0.2

        # Paths
        self.DATA_PATH = 'data'
        self.TRAIN_PATH = 'data/train'
        self.TEST_PATH = 'data/test'
        self.MODEL_PATH = 'trained-models'
        self.LOG_PATH = 'logs'
