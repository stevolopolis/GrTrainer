import glob
import torch
import os
import random
import math
import numpy as np

from PIL import Image
from torchvision import transforms
from parameters import Params

# Params class containing parameters for all visualization.
params = Params()

class DataLoader:
    def __init__(self, path, batch_size, train_val_split=0.2):
        self.path = path
        self.batch_size = batch_size
        self.train_val_split = train_val_split

        # Get class names list
        self.img_cls_list = self.get_cls_id()
        # Get all ids in dataset
        self.img_id_map = self.scan_img_id()
        self.n_data = len(self.img_id_map.keys())
        self.img_id_list = list(self.img_id_map.keys())
        # Shuffle ids for training
        random.shuffle(self.img_id_list)

        # Data augmentation
        self.transformation = transforms.Compose([
            transforms.RandomRotation(90),
            transforms.GaussianBlur((5, 5))
            #transforms.RandomPerspective()
        ])
        self.transformation_rgb = transforms.Compose([
            transforms.ColorJitter()
        ])
        

    def load_batch(self):
        for i, (img, label) in enumerate(self.load()):
            img = self.transformation(img)
            if i % self.batch_size == 0:
                img_batch = img
                label_batch = label
            elif (i+1) % self.batch_size == 0:
                yield (img_batch, label_batch)

            img_batch = torch.cat((img_batch, img), dim=0)
            label_batch = torch.cat((label_batch, label), dim=0)

        yield (img_batch, label_batch)
    
    def load(self):
        for img_id_with_var in self.img_id_list:
            img_id = img_id_with_var.split('_')[-1]
            img_cls = self.img_id_map[img_id_with_var]
            img_cls_idx = self.img_cls_list.index(img_cls)
            img_cls_idx = torch.tensor([img_cls_idx])

            img_path = os.path.join(self.path, img_cls, img_id)

            # Open RGB image with PIL
            img_rgb = Image.open(os.path.join(img_path, img_id_with_var + '_RGB.png'))
            img_rgb = img_rgb.resize((params.OUTPUT_SIZE, params.OUTPUT_SIZE))
            img_rgb = np.array(img_rgb)
            # Open depth image with PIL
            img_d = Image.open(os.path.join(img_path, img_id_with_var + '_perfect_depth.tiff'))
            img_d = img_d.resize((params.OUTPUT_SIZE, params.OUTPUT_SIZE))
            img_d = np.array(img_d)

            yield (self.process(img_rgb, img_d), img_cls_idx)

    def process(self, rgb, d):
        """Returns rgbd image with correct format for inputing to model."""
        d = np.expand_dims(d, 2)
        img = np.concatenate((rgb, d), axis=2)
        img = np.moveaxis(img, -1, 0)
        img = np.expand_dims(img, 0)
        img = torch.tensor(img, dtype=torch.float32).to(params.DEVICE)

        return img

    def scan_img_id(self):
        """Returns all the ids of images present in the 'data' folder."""
        img_id_dict = {}
        for img_path in glob.iglob('%s/*/*/*' % self.path):
            if not img_path.endswith('RGB.png'):
                continue
            
            img_cls = img_path.split('\\')[-3]
            # E.g. '<img_idx>_<img_id>_<img_type>.png'
            img_name = img_path.split('\\')[-1]
            img_var = img_name.split('_')[0]
            img_id = img_name.split('_')[1]
            img_id_with_var = img_var + '_' + img_id
            img_id_dict[img_id_with_var] = img_cls

        n_data = len(img_id_dict.keys())
        n_train, n_val = self.get_train_val(n_data)
        print('Dataset size: %s' % n_data)
        print('Training steps: %s -- Val steps: %s' % (n_train, n_val))
        return img_id_dict

    def get_cls_id(self):
        """Returns a list of class names in fixed order (according to the txt file)."""
        cls_list = []
        with open(os.path.join(params.DATA_PATH, params.LABEL_FILE), 'r') as f:
            file = f.readlines()
            for cls in file:
                # remove '\n' from string
                cls = cls[:-1]
                cls_list.append(cls)

        return cls_list

    def get_train_val(self, n_data=None):
        """Returns the number of training/validation steps."""
        if n_data is not None:
            n_steps = math.ceil(n_data / self.batch_size)
        else:
            n_steps = math.ceil(self.n_data / self.batch_size)
        n_val = round(n_steps * self.train_val_split)
        n_train = n_steps - n_val
        return n_train, n_val

    def augment(self, img):
        """Return augmented images."""




def jacquard_sin_loader(img_id):
    """Returns 'sin' ground truth map for images take from the Jaquard Dataset.
    
    Code referenced from @author: Sulabh Kumra - https://github.com/skumra/robotic-grasping"""
    # Get ground-truth path from img_id
    path = glob.glob('%s/*/%s_grasps.txt' % (params.DATA_PATH, img_id))[0]
    # Load all grasp rectangles from .txt file
    bbs = grasp.GraspRectangles.load_from_jacquard_file(path, scale=params.OUTPUT_SIZE / 1024.0)
    # Convert grasp rectangles into one single 'cos' map
    _, ang_img, _ = bbs.draw((params.OUTPUT_SIZE, params.OUTPUT_SIZE))
    sin = numpy_to_torch(np.sin(2 * ang_img))

    return sin


def numpy_to_torch(s):
    if len(s.shape) == 2:
        return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
    else:
        return torch.from_numpy(s.astype(np.float32))