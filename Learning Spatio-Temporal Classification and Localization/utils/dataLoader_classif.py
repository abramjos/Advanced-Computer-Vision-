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


classes = {}
classes["Boxing"] = 1
classes["Carrying"] = 2
classes["Clapping"] = 3
classes["Jogging"] = 4
classes["Open_Close_Car_Door"] = 5
classes["Open_Close_Trunk"] = 6
classes["Running"] = 7
classes["Throwing"] = 8
classes["Walking"] = 9
classes["Waving"] = 10


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

	result = np.zeros((num_chunks, 16, 240,320,1), dtype=np.float32)
	cnt = 0
	# print(result.shape,arr.shape)
	# exit()
	for i in range(0,  (num_chunks * stride) - stride, stride):
		result[cnt,:,:,:,:] = arr[i:i + size]
		cnt += 1
	
	return result





def sliding_window(arr, size, stride):
	num_chunks = int((len(arr) - size) / stride) + 2
	
	result = np.zeros((num_chunks, 16,224, 224, 3), dtype=np.float32)
	cnt = 0

	for i in range(0,  (num_chunks * stride) - stride, stride):
		result[cnt,:,:,:,:] = arr[i:i + size]
		cnt += 1
	
	return result



def get_video_clips(video_path, mode):
	
	video = np.array(skvideo.io.vread(video_path))
	video = np.array([cv2.resize(frame, (224, 224)) for frame in video], dtype=np.float32)

	clips = sliding_window(video, 16, 16)
	#print(clips.shape)
	

	if mode == "Train":
		return video
	else:
		return clips
	


class UCF_IR(Dataset):
	def __init__(self, videoPth, mode=None,transform_rgb=None,transform_seg=None):
		self.videoPth = videoPth
		self.transform_rgb = transform_rgb
		self.transform_seg = transform_seg
		self.mode = mode

		Trainvideo = []
		Testvideo = []
		time = ["Day", "Night"]
		location = ["Ground", "Roof"]
		timeptr = 0
		location_ptr = 0
		for i in range(len(time)*len(location)):
			classes = os.listdir(self.videoPth + time[timeptr] + '/' + location[location_ptr])
			for j in range(len(classes)):
				totalvideos = []
				for file in os.listdir(self.videoPth + time[timeptr] + '/' + location[location_ptr] + '/' + classes[j]  + '/' ):
					if file.endswith(".avi"):
						totalvideos.append(self.videoPth + time[timeptr] + '/' + location[location_ptr] + '/' + classes[j]  + '/' + file)
				trainLimit = int(0.8*len(totalvideos))
				
				Trainvideo.append(totalvideos[:trainLimit])
				Testvideo.append(totalvideos[trainLimit:])

			location_ptr += 1
			if(location_ptr > 1):
				location_ptr = 0
				timeptr += 1
			if timeptr > 1:
				Trainvideo = list(itertools.chain.from_iterable(Trainvideo))
				Testvideo = list(itertools.chain.from_iterable(Testvideo))
				if mode == "Train":
					self.videoSet = Trainvideo
				else:
					self.videoSet = Testvideo
					


			
	def __getitem__(self,index):
		
		file = self.videoSet[index]

		lines = []
		anno_file = file[:-4] + '_gt.txt'
		with open(anno_file , 'r') as f:
			lines = f.readlines()

		video = get_video_clips(file, self.mode)
		# print(video.shape[0])
		# exit()

		if self.mode == "Train":
			start_frame = random.randint(0,video.shape[0]-16)
			end_frame = start_frame + 16
			

			clips = video[start_frame:end_frame,:,:,:]
			
			action_class = []
			bboxList = []
			corresponding_anno = lines[start_frame:end_frame]


			cnt = 0
			segmentation_clips = np.zeros((16,240,320,1))
			for i in range(len(corresponding_anno)):
				line = corresponding_anno[i].rstrip().split(',')
				bbox = (line[4],line[5],line[6],line[7])
				bbox = (int(320*int(line[4])/640),int(240*int(line[5])/480), int(320*(int(line[4]) + int(line[6]))/640), int(240*(int(line[5]) + int(line[7]))/480))
				
				segmentation_clips[i,bbox[1]:bbox[3],bbox[0]:bbox[2],:] = 255
				action_class.append(int(line[len(line)-1]) + 1)


			
			if (self.transform_rgb):
				clips = self.transform_rgb(clips)
			if (self.transform_seg):
				segmentation_clips = self.transform_seg(segmentation_clips)
			

			action_class = torch.from_numpy(np.array(action_class)).type(torch.FloatTensor)
			if action_class.shape[0] != 16:
				action = file.rstrip().split('/')
				action = action[len(action)-2]
				action_class = torch.zeros(16)
				action_class[:] = int(classes[action])

			
			
			return clips, action_class, segmentation_clips
		
		else:
			action_class = []
			
			segmentation_clips = np.zeros((len(lines),240,320,1))
			for i in range(len(lines)):
				line = lines[i].rstrip().split(',')
				bbox = (line[4],line[5],line[6],line[7])
				bbox = (int(320*int(line[4])/640),int(240*int(line[5])/480), int(320*(int(line[4]) + int(line[6]))/640), int(240*(int(line[5]) + int(line[7]))/480))

				segmentation_clips[i,bbox[1]:bbox[3],bbox[0]:bbox[2],:] = 255
				# segmentationClips.append(segmentation_clips)

				action_class.append(int(line[len(line)-1]) + 1)

			segmentation_clips = sliding_window_seg(segmentation_clips,16,16)
			
			action_class = torch.from_numpy(sliding_window_label(np.array(action_class),16,16))
			# print(video.shape, segmentation_clips.shape,action_class.shape)
			# exit()

			# return np.transpose(video,(0, 4, 1, 2, 3)), action_class, np.transpose(segmentation_clips,(0, 4, 1, 2, 3)),
			return video, action_class, segmentation_clips, file

	def __len__(self):
		return len(self.videoSet)	
			



