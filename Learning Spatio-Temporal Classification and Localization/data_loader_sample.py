
import os 
import cv2
import sys
import numpy as np
from torchvision import datasets,transforms
from torch.utils.data import DataLoader


import videotransforms
from videotransforms import HorizontalFlip

from utils.dataLoader import UCF_101


def draw_chunk(boxes,clips,boxes_flipped,clips_flipped,label):
    fold = 'tests'
    for i in range(16):
        box = boxes[i]
        box_flip = boxes_flipped[i]
        # import ipdb;ipdb.set_trace()
        clips[i] = cv2.rectangle(cv2.UMat(clips[i]).get(), (int(box[0]),int(box[1])),(int(box[0]+box[2]),int(box[1]+box[3])), (0,0,255), 2)
        cv2.imwrite(fold+'/{}_{}.jpg'.format(label,i),clips[i])
        # cv2.imwrite(fold+'/{}_{}_seg.jpg'.format(label,i),segmentation_clips[i]*255)
        # flipping
        clips_flipped[i] = cv2.rectangle(cv2.UMat(clips_flipped[i]).get(), (int(box_flip[0]),int(box_flip[1])),(int(box_flip[0]+box_flip[2]),int(box_flip[1]+box_flip[3])), (0,0,255), 2)
        cv2.imwrite(fold+'/{}_{}_flipped.jpg'.format(label,i),clips_flipped[i])

    return(True)


batchSize = 16
epochs = 100
mode = "Train"


transform_rgb = transforms.Compose([videotransforms.ClipToTensor()])
transform_seg = transforms.Compose([videotransforms.ClipToTensor_seg()])

dsetTrain = UCF_101('.', mode, transform_rgb = transform_rgb)

# dsetTrain = UCF_101('.', mode, transform_rgb,transform_seg, test_loc='./test', flag_seg = True)

val = dsetTrain.__getitem__(1)

trainloader = DataLoader(dsetTrain, batch_size = batchSize, shuffle = True)


for i,x in enumerate(trainloader):
    #x = torch.randn(20, 3, 16, 112, 112)
    for j,(clips,boxes,labels)  in enumerate(zip(x[0],x[1],x[2])): 
        if np.count_nonzero(boxes) > 0:
            clips_flipped,boxes_flipped = HorizontalFlip(clips,boxes)
            clips = np.transpose(np.array(clips*255),(1,2,3,0)).astype(np.uint8)
            clips_flipped = np.transpose(np.array(clips_flipped*255),(1,2,3,0)).astype(np.uint8)
            file_name = str(labels)+'_%d_%d'%(i,j)
            draw_chunk(np.array(boxes),clips,np.array(boxes_flipped),clips_flipped,file_name)

    if i > 4: break