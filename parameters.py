import torch

class Params:
    """
    Parameters for AM visualization on gr-convnet
    """
    def __init__(self):
        # Model name
        self.MODEL_NAME = 'grconvCLS_top5'

        # device: cpu / gpu
        self.DEVICE = torch.device('cuda') if torch.cuda.is_available \
                                      else torch.device('cpu')
        # Training params
        self.NUM_CLASS = 5
        self.OUTPUT_SIZE = 128  # 128 for training grCLS
        self.IMG_SIZE = (4, self.OUTPUT_SIZE, self.OUTPUT_SIZE) 
        self.EPOCHS = 50
        self.LR = 1e-4
        self.BATCH_SIZE = 16
        self.TRAIN_VAL_SPLIT = 0.2

        # Paths
        self.DATA_PATH = 'data'
        self.TRAIN_PATH = 'data/top_5/train'
        self.TEST_PATH = 'data/top_5/test'
        self.LABEL_FILE = 'cls_top_5.txt'
        self.MODEL_PATH = 'trained-models'
        self.MODEL_LOG_PATH = 'trained-models/%s' % self.MODEL_NAME
        self.LOG_PATH = 'logs'