import os 
import sys
import numpy as np
import random

import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.nn import functional as F
from torchvision import datasets,transforms
from torchvision.utils import save_image

from PIL import Image
import matplotlib.pyplot as plt
from skimage import io
import glob
import itertools
import configurations as cfg
import skvideo.io
import cv2
import pickle
import torch

# im size 
IM_SIZE = 112

def HorizontalFlip(imgs, boxes):
    """Horizontally flip the given seq Images randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """
    """
    Args:
        img (seq Images): seq Images to be flipped.
    Returns:
        seq Images: Randomly flipped seq images.
    """
    boxes_flipped = np.zeros_like(boxes)
    for ix,i in enumerate(boxes):
        x,y,w,h = boxes[ix]
        boxes_flipped[ix] = np.array([(112-x-w),y,w,h]) 
        
    return (np.array(torch.flip(torch.tensor(imgs), [2])).copy(),boxes_flipped)
    # return imgs


def sliding_window_label(arr, size, stride):
    num_chunks = int((len(arr) - size) / stride) + 2
    
    result = np.zeros((num_chunks, 16), dtype=np.float32)
    cnt = 0
    for i in range(0,  (num_chunks * stride) - stride, stride):
        result[cnt,:] = arr[i:i + size]
        cnt += 1
    
    return result


def sliding_window_seg(arr, size, stride):
    num_chunks = int((len(arr) - size) / stride) + 2

    result = np.zeros((num_chunks, 16, IM_SIZE, IM_SIZE, 1), dtype=np.float32)
    cnt = 0
    # print(result.shape,arr.shape)
    # exit()
    for i in range(0,  (num_chunks * stride) - stride, stride):
        result[cnt,:,:,:,:] = arr[i:i + size]
        cnt += 1
    
    return result


def sliding_window(arr, size, stride):
    num_chunks = int((len(arr) - size) / stride) + 2
    
    result = np.zeros((num_chunks, 16, IM_SIZE, IM_SIZE, 3), dtype=np.float32)
    cnt = 0

    for i in range(0,  (num_chunks * stride) - stride, stride):
        result[cnt,:,:,:,:] = arr[i:i + size]
        cnt += 1
    
    return (result)

def get_video_clips(video_path, mode, return_ratio = False):
    
    video = np.array(skvideo.io.vread(video_path))
    r = (np.array(video.shape[1:3])/float(IM_SIZE))[::-1]

    video = np.array([cv2.resize(frame, (IM_SIZE, IM_SIZE)) for frame in video], dtype=np.float32)

    clips = sliding_window(video, 16, 16)
    #print(clips.shape)
    

    if return_ratio == True:
        return (video,r)
    else:
        return clips
    

class UCF_101(Dataset):
    def __init__(self, videoPth, mode = None, transform_flip = None, transform_rgb = None,transform_seg = None, annotation_file = 'pyannot.pkl', test_loc=None, flag_seg = False, per_vid_samples = 5):
        self.videoPth = videoPth
        self.per_vid_samples = per_vid_samples
        self.transform_rgb = transform_rgb
        self.transform_seg = transform_seg
        self.transform_flip = transform_flip

        self.mode = mode
        self.test_loc = test_loc
        self.flag_seg = flag_seg
        with open(self.videoPth+'/'+annotation_file,'rb') as f:
            self.annotation = pickle.load(f)
        self.files_anno = [i for i in self.annotation.keys()]
        # self.classes = {x:i for i,x in enumerate(np.unique(np.array([i.split('/')[0] for i in self.files_anno])))}

        video = []
        if mode == 'Train':
            vid_path = self.videoPth+'/train'
        else:
            vid_path = self.videoPth+'/test'

        classes = os.listdir(vid_path)
        for j in range(len(classes)):
            totalvideos = []
            for file in os.listdir(vid_path + '/' + classes[j]  + '/' ):
                if file.endswith(".avi"):
                    totalvideos.append(vid_path + '/' + classes[j]  + '/'  + file)
            video.append(totalvideos)
        video = list(itertools.chain.from_iterable(video))
        self.videoSet = video


        self.classes = {i:x for i,x in enumerate(classes)}
        self.classes_inv = {x:i for i,x in enumerate(classes)}

            
    def __getitem__(self, index):
        frame_len = 16
        file = self.videoSet[index]
        # print('\n\nindex \t %d'%index)
        video,r = get_video_clips(file, self.mode,return_ratio = True)
        
        class_id = file.split('/')[::-1][1]
        file_tag = '/'.join(file.replace('.avi','').split('/')[::-1][:2][::-1])

        start_frame = random.randint(0,video.shape[0]-frame_len)
        end_frame = start_frame + frame_len

        boxes = np.array([np.zeros(4) for i in range(frame_len)]).astype(np.uint8)
        if file_tag in self.files_anno:
            anno = self.annotation[file_tag]['annotations'][0]
            if start_frame <= anno['sf'] <= end_frame or anno['sf'] <= start_frame < end_frame <= anno['ef'] or  start_frame <= anno['ef'] <= end_frame:
                start = max(0,(start_frame-anno['sf']))
                end = min(min(start+frame_len,end_frame-anno['sf']),anno['ef']-anno['sf'])

                if start_frame <= anno['sf'] <= end_frame: 
                    boxes[(anno['sf']-start_frame):] = anno['boxes'][start:end] 
                else:
                    boxes[:(end-start)] = anno['boxes'][start:end]


        if np.count_nonzero(boxes)>0:
            flag_bbox = True
            # using aspect ratio for the boxes based on cropping
            for i in range(len(boxes)):
                boxes[i] = np.array([boxes[i][0]/r[0],boxes[i][1]/r[1],boxes[i][2]/r[0],boxes[i][3]/r[1]])
        else:
            flag_bbox = False



        clips = video[start_frame:end_frame,:,:,:]  
        segmentation_clips = np.zeros(clips.shape[:-1])

        action_class = self.classes_inv[class_id]


        if flag_bbox == True:
            if self.test_loc != None:
                try:
                    fold = self.test_loc + '/'+class_id
                    os.mkdir(fold)
                except:
                    pass
            clips_flipped,boxes_flipped = HorizontalFlip(clips,boxes) 
            for i in range(frame_len):
                # import ipdb; ipdb.set_trace()
                box = boxes[i]
                box_flip = boxes_flipped[i]
                segmentation_clips[i,box[1]:box[1]+box[3],box[0]:box[0]+box[2]] = 1

                if self.test_loc != None:
                    clips[i] = cv2.rectangle(clips[i], (box[0],box[1]),(box[0]+box[2],box[1]+box[3]), (0,0,255), 2)
                    cv2.imwrite(fold+'/{}-{}-{}_{}.jpg'.format(file_tag.split('/')[-1],start_frame,end_frame,i),clips[i])
                    cv2.imwrite(fold+'/{}-{}-{}_{}_seg.jpg'.format(file_tag.split('/')[-1],start_frame,end_frame,i),segmentation_clips[i]*255)
                    # flipping
                    clips_flipped[i] = cv2.rectangle(clips_flipped[i], (box_flip[0],box_flip[1]),(box_flip[0]+box_flip[2],box_flip[1]+box_flip[3]), (0,0,255), 2)
                    cv2.imwrite(fold+'/{}-{}-{}_{}_flipped.jpg'.format(file_tag.split('/')[-1],start_frame,end_frame,i),clips_flipped[i])

        if (self.transform_rgb):
          clips = self.transform_rgb(clips)

        # if (self.transform_flip):
        #   clips_flipped, boxes_flipped = self.transform_flip((clips,boxes))

        # import ipdb; ipdb.set_trace()

        # if (self.transform_seg):
        #   segmentation_clips = self.transform_seg(segmentation_clips)

        
        if self.flag_seg == True:
            return(clips,boxes,action_class,segmentation_clips)
        else:
            return(clips,boxes,action_class)

        
    def __len__(self):
        # return 200#len(self.videoSet*self.per_vid_samples)   
        return len(self.videoSet*self.per_vid_samples)   
        # return len(self.videoSet)   
            


