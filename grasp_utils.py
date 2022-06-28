import torch
import numpy as np

from shapely.geometry import Polygon
from math import pi 


def bboxes_to_grasps(bboxes):
    # convert bbox to grasp representation -> tensor([x, y, theta, h, w])
    x = bboxes[:,0] + (bboxes[:,4] - bboxes[:,0])/2
    y = bboxes[:,1] + (bboxes[:,5] - bboxes[:,1])/2 
    theta = torch.atan((bboxes[:,3] -bboxes[:,1]) / (bboxes[:,2] -bboxes[:,0]))
    w = torch.sqrt(torch.pow((bboxes[:,2] -bboxes[:,0]), 2) + torch.pow((bboxes[:,3] -bboxes[:,1]), 2))
    h = torch.sqrt(torch.pow((bboxes[:,6] -bboxes[:,0]), 2) + torch.pow((bboxes[:,7] -bboxes[:,1]), 2))
    grasps = torch.stack((x, y, theta, h, w), 1)
    return grasps


def grasps_to_bboxes(grasps):
    # convert grasp representation to bbox
    x = grasps[:,0] * 1024
    y = grasps[:,1] * 1024
    theta = torch.deg2rad(grasps[:,2] * 180 - 90)
    w = grasps[:,3] * 1024
    h = grasps[:,4] * 100
    
    x1 = x -w/2*torch.cos(theta) +h/2*torch.sin(theta)
    y1 = y -w/2*torch.sin(theta) -h/2*torch.cos(theta)
    x2 = x +w/2*torch.cos(theta) +h/2*torch.sin(theta)
    y2 = y +w/2*torch.sin(theta) -h/2*torch.cos(theta)
    x3 = x +w/2*torch.cos(theta) -h/2*torch.sin(theta)
    y3 = y +w/2*torch.sin(theta) +h/2*torch.cos(theta)
    x4 = x -w/2*torch.cos(theta) -h/2*torch.sin(theta)
    y4 = y -w/2*torch.sin(theta) +h/2*torch.cos(theta)
    bboxes = torch.stack((x1, y1, x2, y2, x3, y3, x4, y4), 1)
    return bboxes


def box_iou(bbox_value, bbox_target):
    p1 = Polygon(bbox_value.view(-1,2).tolist())
    p2 = Polygon(bbox_target.view(-1,2).tolist())
    iou = p1.intersection(p2).area / (p1.area +p2.area -p1.intersection(p2).area) 
    return iou


def get_correct_grasp_preds(output, target):
    bbox_output = grasps_to_bboxes(output)
    correct = 0
    for i in range(len(target)):
        bbox_target = grasps_to_bboxes(target[i])
        #print(output[i], target[i])
        for j in range(len(bbox_target)):
            iou = box_iou(bbox_output[i], bbox_target[j])
            pre_theta = output[i][2] * 180 - 90
            target_theta = target[i][j][2] * 180 - 90
            angle_diff = torch.abs(pre_theta - target_theta)
            
            if angle_diff < 30 and iou > 0.25:
                correct += 1
                break

    return correct, len(target)
