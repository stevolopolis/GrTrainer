"""
This file contains the DataLoader class which is responsible for
loading both CLS data and Grasping data.

DataLoader also includes all the necessary function for data augmentation
such as a color and noise augmentation pipeline for CLS and a
rotation+translation pipeline for Grasping.

"""
import glob
import torch
import os
import random
import math
import torch.nn as nn
import numpy as np

from PIL import Image
from torchvision import transforms
from parameters import Params
from utils import AddGaussianNoise

params = Params()

class MyRotationTransform:
    """Rotate by one of the given angles."""
    def __init__(self, angle):
        self.angles = angle

    def __call__(self, x):
        return transforms.functional.rotate(x, self.angle)


class DataLoader:
    """
    DataLoader class. Loads both CLS data and Grasping data.

    CLS data:
        - self.load_batch() and self.load()
    Grasp data:
        - self.load_grasp_batch() and self.load_grasp()
    Image processing:
        - self.process()
    CLS labels:
        - self.scan_img_id() and self.get_cls_id()
    Grasp labels:
        - self.load_grasp_label() and self.get_grasp_label()
    """
    def __init__(self, path, batch_size, train_val_split=0.2):
        self.path = path
        self.batch_size = batch_size
        self.train_val_split = train_val_split

        # Get list of class names
        self.img_cls_list = self.get_cls_id()
        # Get dictionary of image-id to classes
        self.img_id_map = self.scan_img_id()
        self.n_data = len(self.img_id_map.keys())
        self.img_id_list = list(self.img_id_map.keys())
        # Shuffle ids for training
        random.shuffle(self.img_id_list)

        # Custom data augmentations
        # Add gaussian noise with 25% probability
        random_transforms = transforms.RandomApply(nn.ModuleList([AddGaussianNoise(0, .02)]), p=0.25)
        # Geometric data augmentations
        self.transformation = transforms.Compose([
            transforms.RandomResizedCrop(params.OUTPUT_SIZE, scale=(.75, .85), ratio=(1, 1)),
            transforms.RandomRotation(90)
        ])
        # Color data augmentations
        self.transformation_rgb = transforms.Compose([
            #transforms.ColorJitter(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            #random_transforms
            #transforms.Grayscale(num_output_channels=3)
        ])
        

    def load_batch(self):
        """Yields a batch of CLS training data -- (img, label)."""
        for i, (img, label) in enumerate(self.load()):
            img = self.transformation(img)
            if i % self.batch_size == 0:
                img_batch = img
                label_batch = label
            elif (i+1) % self.batch_size == 0:
                img_batch = torch.cat((img_batch, img), dim=0)
                label_batch = torch.cat((label_batch, label), dim=0)
                yield (img_batch, label_batch)
            else:
                img_batch = torch.cat((img_batch, img), dim=0)
                label_batch = torch.cat((label_batch, label), dim=0)

        # This line catches the final few instances (less than batch_size)
        if (i + 1) % self.batch_size != 0:
            yield (img_batch, label_batch)
    
    def load(self):
        """Yields a single instance of CLS training data -- (img, label)."""
        for img_id_with_var in self.img_id_list:
            img_id = img_id_with_var.split('_')[-1]
            img_cls = self.img_id_map[img_id_with_var]
            img_cls_idx = self.img_cls_list.index(img_cls)
            img_cls_idx = torch.tensor([img_cls_idx]).to(params.DEVICE)

            img_path = os.path.join(self.path, img_cls, img_id)

            # Open RGB image with PIL
            img_rgb = Image.open(os.path.join(img_path, img_id_with_var + '_RGB.png'))
            img_rgb = img_rgb.resize((params.OUTPUT_SIZE, params.OUTPUT_SIZE))
            img_rgb = torch.tensor(np.array(img_rgb), dtype=torch.float32)
            # Open depth image with PIL
            img_d = Image.open(os.path.join(img_path, img_id_with_var + '_perfect_depth.tiff'))
 
            img_d = img_d.resize((params.OUTPUT_SIZE, params.OUTPUT_SIZE))
            img_d = torch.tensor(np.array(img_d), dtype=torch.float32)

            yield (self.process(img_rgb, img_d), img_cls_idx)

    def load_grasp_batch(self):
        """Yields a batch of Grasp training data -- (img, grasp-label, grasp-candidates)."""
        for i, (img, label, candidates) in enumerate(self.load_grasp()):
            if i % self.batch_size == 0:
                img_batch = img
                label_batch = label
                candidate_batch = [candidates]
            elif (i+1) % self.batch_size == 0:
                img_batch = torch.cat((img_batch, img), dim=0)
                label_batch = torch.cat((label_batch, label), dim=0)
                candidate_batch.append(candidates)
                yield (img_batch, label_batch, candidate_batch)
            else:
                img_batch = torch.cat((img_batch, img), dim=0)
                label_batch = torch.cat((label_batch, label), dim=0)
                candidate_batch.append(candidates)

        # This line of code catches the final few instances (less that batch_size)
        if (i + 1) % self.batch_size != 0:
            yield (img_batch, label_batch, candidate_batch)

    def load_grasp(self):
        """Yields a single instance of Grasp training data -- (img, grasp-label, grasp-candidates)."""
        for img_id_with_var in self.img_id_list:
            img_id = img_id_with_var.split('_')[-1]
            img_cls = self.img_id_map[img_id_with_var]

            img_path = os.path.join(self.path, img_cls, img_id)

            # Open RGB image with PIL
            img_rgb = Image.open(os.path.join(img_path, img_id_with_var + '_RGB.png'))
            img_rgb = img_rgb.resize((params.OUTPUT_SIZE, params.OUTPUT_SIZE))
            img_rgb = torch.tensor(np.array(img_rgb), dtype=torch.float32)
            # Open Depth image with PIL
            img_d = Image.open(os.path.join(img_path, img_id_with_var + '_perfect_depth.tiff'))
            img_d = img_d.resize((params.OUTPUT_SIZE, params.OUTPUT_SIZE))
            img_d = torch.tensor(np.array(img_d), dtype=torch.float32)

            # Get Grasp label candidates and training label from '_grasps.txt' file
            grasp_file_path = os.path.join(img_path, img_id_with_var + '_grasps.txt')
            # List of Grasp candidates
            grasp_list = self.load_grasp_label(grasp_file_path)
            # Single Grasp label
            grasp_label = self.get_grasp_label(grasp_list, metric='widest')
            grasp_label = np.array([grasp_label])
            grasp_list = np.array(grasp_list)
            
            # Normalize and combine rgb with depth channel
            img_rgbd = self.process(img_rgb, img_d)

            # Manual augmentaion random parameters
            degree = random.choice([0, 90, -90, 180])
            ratio = random.uniform(0.75, 0.85)
            jitter_x = random.uniform(0.075, 0.075)
            jitter_y = random.uniform(-0.075, 0.075)
            
            # Augmentation on image -- random resized crops with jitters
            img_rgbd = crop_jitter_resize_img(img_rgbd, ratio, jitter_x, jitter_y)
            # Augmentation on image -- random rotations (can only do 1/2 pi rotations for label accuracy)
            img_rgbd = transforms.functional.rotate(img_rgbd, degree)
            # Augmentation on labels -- random resized crop with jitters
            grasp_label = crop_jitter_resize_label(grasp_label, ratio, jitter_x, jitter_y)
            # Augmentation on labels -- random rotations
            grasp_label = rotate_grasp_label(grasp_label, degree)
            # Augmentation on label candidates -- random resized crop with jitter
            grasp_list = crop_jitter_resize_label(grasp_list, ratio, jitter_x, jitter_y)
            # Augmentation on label candidates -- random rotations
            grasp_list = rotate_grasp_label(grasp_list, degree)
            
            yield (img_rgbd,
                   torch.tensor(grasp_label, dtype=torch.float32).to(params.DEVICE),
                   torch.tensor(grasp_list).to(params.DEVICE))
        
    def process(self, rgb, d):
        """
        Returns rgbd image with correct format for inputing to model:
            - Imagenet normalization
            - Concat depth channel to image
        """
        rgb = rgb / 255.0
        rgb = torch.moveaxis(rgb, -1, 0)
        rgb = self.transformation_rgb(rgb)
        if d is None:
            img = rgb
        elif params.NUM_CHANNEL == 3:
            # Input channels -- (gray, gray, depth)
            #rgb = transforms.Grayscale(num_output_channels=1)(rgb)
            #rgb = torch.cat((rgb, rgb), axis=0)
            # Input channels -- (red, green, depth)
            d = torch.unsqueeze(d, 2)
            d = d - torch.mean(d)
            d = torch.clip(d, -1, 1)
            d = torch.moveaxis(d, -1, 0)
            img = torch.cat((rgb[:2], d), axis=0)
        else:
            # Input channels -- (red, green, blue, depth)
            d = torch.unsqueeze(d, 2)
            d = d - torch.mean(d)
            d = torch.clip(d, -1, 1)
            d = torch.moveaxis(d, -1, 0)
            img = torch.cat((rgb, d), axis=0)

        img = torch.unsqueeze(img, 0)
        img = img.to(params.DEVICE)

        return img

    def load_grasp_label(self, file_path):
        """Returns a list of grasp labels from <file_path>."""
        grasp_list = []
        with open(file_path, 'r') as f:
            file = f.readlines()
            # dat format in each line: 'x;y;theta;w;h'
            for grasp in file:
                # remove '\n' from string
                grasp = grasp[:-1]
                label = grasp.split(';')
                label = self.noramlize_grasp(label)
                grasp_list.append(label)

        return grasp_list

    def get_grasp_label(self, grasp_list, metric='random'):
        """
        Returns the selected grasp label for training.
        
        Selection metrics:
            - random -- random label
            - median -- random label between 35-percentile and 65-percentile of grasp dimension sum (w+h)
            - widest -- random label between 70-percentile and 80-percentile of grasp dimension sum (w+h)
            |   - empirically the best choice (sufficiently big without outliers)
            - smallest -- random label between 0-percentile and 20-percentile of grasp dimension sum (w+h)
        
        """
        # Selection method: 'w+h' median
        if metric == 'median':
            grasp_list.sort(key=lambda x: x[3] + x[4])
            mid_idx = random.randint(int(len(grasp_list) * 0.35), int(len(grasp_list) * 0.65))
            return grasp_list[mid_idx]
        # Selection method: random
        if metric == 'random':
            idx = random.randint(0, len(grasp_list) - 1)
            return grasp_list[idx]
        # Selection method: widest grasp
        if metric == 'widest':
            grasp_list.sort(key=lambda x: x[3] + x[4])
            top_10 = random.randint(int(len(grasp_list) * 0.2), int(len(grasp_list) * 0.3))
            return grasp_list[-top_10]
        # Selection method: smallest grasp
        if metric == 'smallest':
            grasp_list.sort(key=lambda x: x[3] + x[4])
            top_10 = int(len(grasp_list) * 0.2)
            return grasp_list[top_10]

    def noramlize_grasp(self, label):
        """Returns normalize grasping labels."""
        norm_label = []
        for i, value in enumerate(label):
            if i == 4:
                # Height
                norm_label.append(float(value) / 100)
            elif i == 2:
                # Theta
                norm_label.append((float(value) + 90) / 180)
            elif i == 3:
                # Width
                norm_label.append(float(value) / 1024)
            else:
                # Coordinates
                norm_label.append(float(value) / 1024)

        return norm_label

    def scan_img_id(self):
        """
        Returns a dictionary mapping the image ids from the 'data' 
        folder to their corresponding classes.
        """
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


# ----------------------------------------------------------------
# Geometric augmentations for Grasp data
# ----------------------------------------------------------------
def rotate_grasp_label(grasp_list, degrees):
    """Returns rotated list of Grasp labels given the degrees."""
    # grasp_list.shape == (n, 5)
    # x, y, theta, w, h
    new_grasp_list =[]
    for grasp in grasp_list:
        x = grasp[0] * 1024
        y = grasp[1] * 1024

        angle = np.deg2rad(-degrees)
        R = np.array([[np.cos(angle), -np.sin(angle)],
                    [np.sin(angle),  np.cos(angle)]])
        o = np.atleast_2d((1024 // 2, 1024 // 2))
        p = np.atleast_2d((x, y))
        
        coords = np.squeeze((R @ (p.T-o.T) + o.T).T)
        if degrees == 0 or degrees == 180:
            theta = grasp[2] * 180 - 90
        elif degrees == 90 or degrees == -90:
            if grasp[2] <= 0.5 :
                theta = grasp[2] * 180
            elif grasp[2] > 0.5:
                theta = grasp[2] * 180 - 180
        
        w = grasp[3]
        h = grasp[4]

        new_grasp_list.append([coords[0] / 1024, coords[1] / 1024, (theta + 90) / 180, w, h])

    return np.array(new_grasp_list)


def crop_jitter_resize_img(img, ratio, jitter_x, jitter_y):
    """Returns an augmented image after crop-jitter-resizing."""
    # img.shape = (1, 3, img_h, img_w)
    img_h = img.shape[2]
    img_w = img.shape[3]
    new_img_h = int(img_h * ratio)
    new_img_w = int(img_w * ratio)

    jitter_coord_y = int(jitter_y * params.OUTPUT_SIZE)
    jitter_coord_x = int(jitter_x * params.OUTPUT_SIZE)

    crop_y = (img_h - new_img_h) // 2
    crop_x = (img_w - new_img_w) // 2
    
    crop_img = img[:, :, crop_y + jitter_coord_y : crop_y + new_img_h + jitter_coord_y,\
                   crop_x + jitter_coord_x : crop_x + new_img_w + jitter_coord_x]

    return transforms.functional.resize(crop_img, (img_h, img_w))


def crop_jitter_resize_label(grasp_label, ratio, jitter_x, jitter_y):
    """Returns augmented labels after crop-jitter-resizing."""
    grasp_label[:, 0] = (grasp_label[:, 0] - ((1-ratio)/2) - jitter_x) / ratio
    grasp_label[:, 1] = (grasp_label[:, 1] - ((1-ratio)/2) - jitter_y) / ratio
    grasp_label[:, 3] = grasp_label[:, 3] / ratio
    grasp_label[:, 4] = grasp_label[:, 4] / ratio

    return grasp_label
