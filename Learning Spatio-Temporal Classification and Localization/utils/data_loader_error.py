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
    def __init__(self, videoPth, mode = None, transform_flip = None, transform_rgb = None,transform_seg = None, annotation_file = 'pyannot.pkl', test_loc=None, flag_seg = False, per_vid_samples = 2):
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

        self.classes = {0: 'ApplyEyeMakeup', 1: 'ApplyLipstick', 2: 'Archery', 3: 'BabyCrawling', 4: 'BalanceBeam', 5: 'BandMarching', 6: 'BaseballPitch', 7: 'Basketball', 8: 'BasketballDunk', 9: 'BenchPress', 10: 'Biking', 11: 'Billiards', 12: 'BlowDryHair', 13: 'BlowingCandles', 14: 'BodyWeightSquats', 15: 'Bowling', 16: 'BoxingPunchingBag', 17: 'BoxingSpeedBag', 18: 'BreastStroke', 19: 'BrushingTeeth', 20: 'CleanAndJerk', 21: 'CliffDiving', 22: 'CricketBowling', 23: 'CricketShot', 24: 'CuttingInKitchen', 25: 'Diving', 26: 'Drumming', 27: 'Fencing', 28: 'FieldHockeyPenalty', 29: 'FloorGymnastics', 30: 'FrisbeeCatch', 31: 'FrontCrawl', 32: 'GolfSwing', 33: 'Haircut', 34: 'Hammering', 35: 'HammerThrow', 36: 'HandstandPushups', 37: 'HandstandWalking', 38: 'HeadMassage', 39: 'HighJump', 40: 'HorseRace', 41: 'HorseRiding', 42: 'HulaHoop', 43: 'IceDancing', 44: 'JavelinThrow', 45: 'JugglingBalls', 46: 'JumpingJack', 47: 'JumpRope', 48: 'Kayaking', 49: 'Knitting', 50: 'LongJump', 51: 'Lunges', 52: 'MilitaryParade', 53: 'Mixing', 54: 'MoppingFloor', 55: 'Nunchucks', 56: 'ParallelBars', 57: 'PizzaTossing', 58: 'PlayingCello', 59: 'PlayingDaf', 60: 'PlayingDhol', 61: 'PlayingFlute', 62: 'PlayingGuitar', 63: 'PlayingPiano', 64: 'PlayingSitar', 65: 'PlayingTabla', 66: 'PlayingViolin', 67: 'PoleVault', 68: 'PommelHorse', 69: 'PullUps', 70: 'Punch', 71: 'PushUps', 72: 'Rafting', 73: 'RockClimbingIndoor', 74: 'RopeClimbing', 75: 'Rowing', 76: 'SalsaSpin', 77: 'ShavingBeard', 78: 'Shotput', 79: 'SkateBoarding', 80: 'Skiing', 81: 'Skijet', 82: 'SkyDiving', 83: 'SoccerJuggling', 84: 'SoccerPenalty', 85: 'StillRings', 86: 'SumoWrestling', 87: 'Surfing', 88: 'Swing', 89: 'TableTennisShot', 90: 'TaiChi', 91: 'TennisSwing', 92: 'ThrowDiscus', 93: 'TrampolineJumping', 94: 'Typing', 95: 'UnevenBars', 96: 'VolleyballSpiking', 97: 'WalkingWithDog', 98: 'WallPushups', 99: 'WritingOnBoard', 100: 'YoYo'}

        #self.classes = {i:x for i,x in enumerate(classes)}
        self.classes_inv = {x:i for i,x in self.classes.items()}

    def get_boxes(self,anno,boxes,video):
        frame_len = 16
        
        start_frame = random.randint(0,video.shape[0]-frame_len)
        end_frame = start_frame + frame_len

        if start_frame <= anno['sf'] <= end_frame or anno['sf'] <= start_frame < end_frame <= anno['ef'] or  start_frame <= anno['ef'] <= end_frame:
            start = max(0,(start_frame-anno['sf']))
            end = min(min(start+frame_len,end_frame-anno['sf']),anno['ef']-anno['sf'])
            try:    
                if start_frame <= anno['sf'] <= end_frame: 
                    boxes[(anno['sf']-start_frame):] = anno['boxes'][start:end] 
                else:
                    boxes[:(end-start)] = anno['boxes'][start:end]
            except:
                return(False,boxes,start_frame,end_frame)
        return(True,boxes,start_frame,end_frame)
            
    def __getitem__(self, index):
        frame_len = 16
        file = self.videoSet[index]
        # print('\n\nindex \t %d'%index)
        video,r = get_video_clips(file, self.mode,return_ratio = True)
        
        class_id = file.split('/')[::-1][1]
        file_tag = '/'.join(file.replace('.avi','').split('/')[::-1][:2][::-1])

        boxes = np.array([np.zeros(4) for i in range(frame_len)]).astype(np.uint8)
        if file_tag in self.files_anno:
            flag = False
            anno = self.annotation[file_tag]['annotations'][0]
            while flag == False:
                boxes = np.array([np.zeros(4) for i in range(frame_len)]).astype(np.uint8)
                flag,boxes,start_frame,end_frame = self.get_boxes(anno,boxes,video)
        else:

            start_frame = random.randint(0,video.shape[0]-frame_len)
            end_frame = start_frame + frame_len

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
        # return len(self.videoSet*self.per_vid_samples)   
        return len(self.videoSet)   
            


