'''

Assignment 1 for Advanced Computer Vision- CAP 6412.
Question 1: CNN based classifier for ImageNet
Abraham Jose
01-30-2020
abraham@knights.ucf.edu 

'''
# from __future__ import print_function
import argparse
import numpy as np
import pickle
import scipy.io 
import torch
import os
import shutil
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from glob import glob
import cv2 
import time
import torchvision.utils as vutils
import torchvision.models as models

from torchsummary import summary as model_summary

from torch.utils.tensorboard import SummaryWriter


# class for training the model
class train_model:
	# initilizing the model params and details
	def __init__(self,log_interval, model, device, optimizer, epochs, log=None, train = True):
		self.log_interval = log_interval
		self.model = model
		self.device = device
		# print(self.device)
		if train == True:
			self.optimizer = optimizer
		self.epochs = epochs
		self.current_epoch = 0
		if log != None:
			self.log = log

	# training code for the model on train_loader dataset generator
	def train(self,train_loader):
		

		# setting model for training
		self.model.train()
		for batch_idx, (data, target) in enumerate(train_loader):
			data_dev, target_dev = data.type(torch.FloatTensor).to(self.device), target.type(torch.FloatTensor).to(self.device)
			self.optimizer.zero_grad()
			output = self.model(data_dev)
			# using cross-entropy as loss function for training
			loss =  F.cross_entropy(output, target_dev.long().view(target.shape[0]))
			# criteria = nn.CrossEntropyLoss().cuda()
			# loss =  criteria(output, target_dev.view_as(target))
			loss.backward()
			self.optimizer.step()

			# _, argmax = torch.max(output, 1)
			# accuracy = (target == argmax.squeeze()).float().mean()

			# for logging at the logging interval during the epoch
			if batch_idx % self.log_interval == 0:
				
				self.current_log_interval+=1

				print('Train Epoch: {} --- {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
					self.current_log_interval, self.current_epoch, batch_idx * len(data_dev), len(train_loader.dataset),
					100. * batch_idx / len(train_loader), loss.item()))
				
				self.log.add_scalar('training_loss',loss.item(),self.current_log_interval)

		return(True)

	def val(self, val_loader):

		# setting model for validation
		self.model.eval()
		val_loss = 0
		correct = 0
		with torch.no_grad():
			for data, target in val_loader:
				data_dev, target_dev = data.type(torch.FloatTensor).to(self.device), target.type(torch.FloatTensor).to(self.device)
				output = self.model(data_dev)
				val_loss += F.cross_entropy(output, target_dev.long().view(target.shape[0]), reduction='sum').item()  # sum up batch loss
				pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
				correct += pred.eq(target_dev.view_as(pred)).sum().item()

		# finding validation loss and accuracy and logging them
		val_loss /= self.val_len
		accuracy = 100. * correct / self.val_len

		self.log.add_scalar('validataion_loss',val_loss,self.current_epoch)
		self.log.add_scalar('validataion_accuracy',accuracy,self.current_epoch)

		print('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
			val_loss, correct, self.val_len, accuracy))
		return(True,accuracy,val_loss)

	def test(self, test_loader):
		self.test_len = len(test_loader.dataset)
		# setting model for testing
		self.model.eval()
		test_loss = 0
		correct = 0
		# creating model for testing and confusion matrix
		cm = np.zeros((200,200))
		with torch.no_grad():
			for data, target in test_loader:
				data_dev, target_dev = data.type(torch.FloatTensor).to(self.device), target.type(torch.FloatTensor).to(self.device)
				output = self.model(data_dev)
				test_loss += F.cross_entropy(output, target_dev.long().view(target.shape[0]), reduction='sum').item()  # sum up batch loss
				pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
				correct += pred.eq(target_dev.view_as(pred)).sum().item()
				for actual,predicted in zip(target.view_as(pred),pred):
					cm[actual,predicted]+=1

		test_loss /= self.test_len
		accuracy = 100. * correct / self.test_len
		return(True,cm,accuracy,test_loss)


	def iterate(self,train_loader, val_loader, test_loader,model_name):
		self.train_len = len(train_loader.dataset)
		self.val_len = len(val_loader.dataset)
		self.batch_size=train_loader.batch_size
		self.current_log_interval=0
		self.val_accuracy_best = 0
		self.test_cm_best = np.zeros((10,10))
		self.len = train_loader

		print("training dataset size = {}\ntesting dataset size = {}\nbatch size = {}".format(self.train_len,self.val_len,self.batch_size))
		# for iteration for training, testing and validation batch
		test_result = []
		for epoch in range(1, self.epochs + 1):
			train_flag = self.train(train_loader = train_loader)
			if train_flag != True:
				return(False,epoch)
			val_flag,accuracy,val_loss = self.val(val_loader = val_loader)
			if val_flag != True:
				return(False,epoch)
			
			if self.val_accuracy_best < accuracy:
				self.save_checkpoint({'epoch': epoch,'state_dict': model.state_dict(),'best_accuracy': accuracy},
					filename='../output/model-{}/checkpoint/{}-best_test-epoch:{}-accuracy:{}.pth.tar'.format(model_name,model_name,epoch,accuracy))
				torch.save(model.state_dict(), "../data/model/bestx_model.pt")
				val_flag,cm,accuracy,test_loss = self.test(test_loader = test_loader)
				self.test_cm_best = cm
				print('Confusion Matrix:\n',cm.astype(np.uint8))
				self.val_accuracy_best = accuracy
			# print(self.test_accuracy_best)
			
			test_result.append([accuracy,val_loss,test_loss])
			self.current_epoch+=1

		return(True,test_result)

	def save_checkpoint(self,state, filename='../output/checkpoint.pth.tar'):
		"""Save checkpoint if a new best is achieved"""
		print ("=> Saving a new best")
		torch.save(state, filename)  # save checkpoint

#dataset class for cifar10
class Dataset(Dataset):
	def __init__(self, data,label,transform = None):
		self.labels = label
		self.X = data[0]
		self.Y = data[1]
		self.size = len(data[1])

		u,c = np.unique(data[1],return_counts=True)
		self.dist = {i:j for i,j in zip(u,c)}
		self.transform = transform
	
	# create a random sample of dataset for testing.
	def samples(self,sample_size=10):
		sample = [self.__getitem__(np.random.randint(0,self.size)) for i in range(sample_size)]
		return(sample)

	# for visualization of the training and testing dataset after augmentation
	def sample_image(self,sample_size=(2,2)):
		import matplotlib.pyplot as plt
		c=0
		for i in range(sample_size[0]):
			for j in range(sample_size[1]):
				c+=1
				axis = plt.subplot(sample_size[0],sample_size[1],c)
				item = self.__getitem__(np.random.randint(0,self.size))
				if type(item[0]) == torch.Tensor:
					axis.imshow(transforms.functional.to_pil_image(item[0]))
				else:
					axis.imshow(item[0]) 
				axis.set_title("{}: {}".format(str(item[1]),self.labels[item[1]])) 
				plt.yticks([])
				plt.xticks([])
		plt.show()

	def __len__(self):
		self.size = len(self.Y)
		return (self.size)
	
	# method for getting the dataset
	def __getitem__(self, idx):
		item = self.X[idx]
		label = self.Y[idx]
		if self.transform != None:
			item = self.transform(item)
		return (item, label);



# function to load the dataset
def load(folder = './data/tiny-imagenet-200/Train', split= .8, length = None, train_flag = False, avg_shape = None):
	data = [[] for i in range(200)]
	lst = glob(folder+'/*/*.JPEG')
	dict_cls = {}
	shape = []
	c = 0 
	for i in lst:
		img = cv2.imread(i)
		try:
			shape.append(img.shape)
		except:
			c+=1
			continue
		cls_id = i.split('/')[-2:][0]

		if cls_id not in dict_cls.keys():
			dict_cls[cls_id] = len(dict_cls)
			data[dict_cls[cls_id]].append(img)
			continue

		elif length != None and len(data[dict_cls[cls_id]]) < length:
			data[dict_cls[cls_id]].append(img)

		elif length == None : 
			data[dict_cls[cls_id]].append(img)
			
		else:
			s = 0
			for j in data:
				s += len(j)
			if length!= None and s >= len(data)*length:
				break 



	if train_flag == True: 
		avg_shape = np.average(np.array(shape),axis=0).astype(np.uint8)
		print('Shape = {}'.format(avg_shape))
		train, val = ([],[]), ([],[]) 
		for ix,i in enumerate(data):
			for _id,j in enumerate(i):
				if _id < int(split*length):
					train[0].append(cv2.resize(j,tuple(avg_shape[:2][::-1])))
					train[1].append(ix)
				else:
					val[0].append(cv2.resize(j,tuple(avg_shape[:2][::-1])))
					val[1].append(ix)
		datax = [train,val]
	else:
		test = ([],[]) 
		for ix,i in enumerate(data):
			for _id,j in enumerate(i):
				test[0].append(cv2.resize(j,tuple(avg_shape[:2][::-1])))
				test[1].append(ix)
		
		datax = test		
	return(datax, dict_cls,avg_shape)

def load_imagenet_dataset(transform=None,folder_path='./data/tiny-imagenet-200/', split=.8, len_train=500):

	# loading the dataset from the train_batch1 and test _batch from cifar10 dataset
	# [train,val],label_dict,avg_shape = load(folder = folder_path+'Train', split = split, train_flag = True, length=len_train)
	[train,val],label_dict,avg_shape = load(folder = folder_path+'Train', split = split, train_flag = True, length=len_train)
	# [train,val] = load(folder = './data/tiny-imagenet-200/Train', split = split, length=350)
	# [train,val] = load(folder = './data/tiny-imagenet-200/Train', split = split, length=150)

	test,label_dict,avg_shape = load(folder = folder_path+'Test', split = split, length=None, avg_shape = avg_shape)
	print(label_dict)
	label = []
	for i in label_dict.keys():
		label.append(i)

	#if 2 transforms are available, use train and test augmentation seperately
	if len(transform) == 2:
		return (Dataset(train, label, transform = transform[0]), Dataset(val, label, transform = transform[0]), Dataset(test, label, transform = transform[1]))
	else:
		return (Dataset(train, label, transform = transform[0]), Dataset(val, label, transform = transform[0]),  Dataset(test, label, transform = transform[0]))


if __name__ == '__main__':
	# training arguments
	parser = argparse.ArgumentParser(description='PyTorch CIFAR Example')
	parser.add_argument('--batch-size', type=int, default=64, metavar='N',
						help='input batch size for training (default: 64)')
	parser.add_argument('--epochs', type=int, default=10, metavar='N',
						help='number of epochs to train (default: 10)')
	parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
						help='learning rate (default: 0.01)')
	parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
						help='SGD momentum (default: 0.5)')
	parser.add_argument('--log-interval', type=int, default=20, metavar='N',
						help='how many batches to wait before logging training status')
	parser.add_argument('--save-model', action='store_true', default=True,
						help='For Saving the current Model')
	parser.add_argument('--test_model', default=False,
						help='For testing the current Model')
	parser.add_argument('--length', type=int, default=500,
						help='For testing the current Model')
	parser.add_argument('--split', type=float, default=.8,
						help='Validation data split for training the current Model')
	parser.add_argument('--model_name', default='resnet_34',
						help='For selecting Model')
	args = parser.parse_args()

	#random seeding the training
	torch.manual_seed(1)

	# creating the augmentation for the dataset
	transform_train = transforms.Compose([transforms.ToPILImage(mode='RGB'),
									transforms.RandomApply([
									transforms.ColorJitter(brightness=.1, contrast=.25, saturation=.25),
									transforms.RandomAffine(degrees=10, scale=None, resample=False, fillcolor=0),
									transforms.RandomHorizontalFlip(),
									transforms.RandomVerticalFlip(),
									], p =0.25),
									transforms.ToTensor(),	
									])
	transform_test = transforms.Compose([transforms.ToPILImage(mode='RGB'),
									transforms.ToTensor(),
									])

	#loading the cifar10 datset
	
	train,val,test = load_imagenet_dataset(transform=[transform_train,transform_test],folder_path='../data/tiny-imagenet-200/', split=args.split, len_train=args.length)
	
	# creating training,testing and validation datasets
	train_loader = torch.utils.data.DataLoader(dataset=train,batch_size=args.batch_size,
											   shuffle=True)
	val_loader = torch.utils.data.DataLoader(dataset=val,batch_size=args.batch_size,
											   shuffle=True)
	test_loader = torch.utils.data.DataLoader(dataset=test,batch_size=args.batch_size,
											   shuffle=False)

	# creating the model to CPU
	# opting GPU if GPU is available
	#loading the model
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(device)
	if args.model_name == 'resnet_18':
		model = models.resnet18(pretrained=False).to(device)
		model_name = args.model_name
	elif args.model_name == 'resnet_34':
		model = models.resnet34(pretrained=False).to(device)
		model_name = args.model_name
	elif args.model_name == 'resnet_50':
		model = models.resnet50(pretrained=False).to(device)
		model_name = args.model_name
	elif args.model_name == 'resnet_101':
		model = models.resnet101(pretrained=False).to(device)
		model_name = args.model_name+'_L%s'%args.length
	elif args.model_name == 'svhn_resnet_101':
		model = models.resnet101(pretrained=True).to(device)
		model_name = args.model_name

	print(model_name)
	# add more layers as required and chaning the no of last layer
	out_features = 200
	from collections import OrderedDict
	n_inputs = model.fc.in_features
	model.fc = nn.Linear(n_inputs, out_features)

	#creating the folders 	
	try:
		os.mkdir('../output/model-{}/'.format(model_name))
		os.mkdir('../output/model-{}/runs'.format(model_name))
		os.mkdir('../output/model-{}/checkpoint'.format(model_name))

	except :
		shutil.rmtree('../output/model-{}/'.format(model_name))
		os.mkdir('../output/model-{}/'.format(model_name))
		os.mkdir('../output/model-{}/runs'.format(model_name))
		os.mkdir('../output/model-{}/checkpoint'.format(model_name))


#	print('\nModel Summary:\n',model_summary(model,input_size=(3,32,32),device =device))

	if args.test_model == False:
		# training the model

		#logging of training
		log = SummaryWriter('../output/model-{}/runs/'.format(model_name))
		images,label = next(iter(train_loader))
		grid = vutils.make_grid(images.to(device))
		log.add_image('images',grid)
		log.add_graph(model.to(device),images.to(device))

		# initilizing the optimizer
		optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

		# creating the training object and iterating each epoch
		train = train_model(args.log_interval, model, device, optimizer, args.epochs, log =log)
		print(train.device)
		flag, result = train.iterate(train_loader = train_loader,val_loader = val_loader, test_loader =test_loader, model_name = model_name)
		# import ipdb;ipdb.set_trace()
		# print(result)
		with open('../output/model-{}/epoch.txt'.format(model_name),'w') as f:
			for _id,(a,b,c) in enumerate(result):
				f.writelines('{},{},{},{}\n'.format(_id,a,b,c))


		print('\n\n\n\n\nFinal confusion matrix :\n\n',train.test_cm_best)
		# saving the model
		if args.save_model:
			torch.save(model.state_dict(), "../output/model-{}/{}-final_model.pt".format(model_name,model_name))
		log.close()
		print('Completed training')
	else:
		# testing the model 
		model.load_state_dict(torch.load("../output/model-{}/{}-final_model.pt".format(model_name,model_name)))
		optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
		# careating the mdoel testing instance
		test = train_model(args.log_interval, model, device, optimizer, args.epochs, train= False)
		# getting the confusion matrix
		val_flag,cm,accuracy = test.test(test_loader = test_loader)
		print("\nConfusion Matrix:\n",cm)
		print('Accuracy:%d'%(np.sum(cm*np.eye(10))/np.sum(cm)*100))

		# creating the error rates
		error_rate = np.zeros(10) 
		for idx,i in enumerate(cm):
			error_rate[idx] = (np.sum(i)-i[idx])
		# printing the error rates
		for i,j in zip(error_rate,['n07734744', 'n02843684', 'n07875152', 'n02988304', 'n03670208', 'n07749582', 'n01770393', 'n02206856', 'n04398044', 'n01983481', 'n07720875', 'n02403003', 'n03804744', 'n01644900', 'n04532106', 'n02837789', 'n07615774', 'n03837869', 'n03201208', 'n07768694', 'n02814533', 'n07873807', 'n02165456', 'n04265275', 'n03089624', 'n02268443', 'n02481823', 'n04376876', 'n07715103', 'n09193705', 'n02950826', 'n02977058', 'n01910747', 'n03250847', 'n03992509', 'n02948072', 'n03770439', 'n01855672', 'n02504458', 'n03902125', 'n02190166', 'n02788148', 'n02793495', 'n02226429', 'n06596364', 'n04328186', 'n02236044', 'n02927161', 'n07695742', 'n03706229', 'n02113799', 'n01774384', 'n04417672', 'n07747607', 'n02486410', 'n02815834', 'n02364673', 'n03930313', 'n04507155', 'n03393912', 'n03179701', 'n03854065', 'n03891332', 'n01950731', 'n04146614', 'n03126707', 'n04067472', 'n04008634', 'n04311004', 'n03584254', 'n02892201', 'n09428293', 'n03814639', 'n02094433', 'n03447447', 'n02058221', 'n04149813', 'n04070727', 'n01945685', 'n03980874', 'n02795169', 'n02699494', 'n03160309', 'n04371430', 'n02808440', 'n02814860', 'n02233338', 'n04562935', 'n03937543', 'n02410509', 'n01917289', 'n02099712', 'n01774750', 'n02909870', 'n07753592', 'n01742172', 'n04465501', 'n03796401', 'n04597913', 'n02132136', 'n02281406', 'n03983396', 'n01984695', 'n02099601', 'n03085013', 'n03544143', 'n02129165', 'n01768244', 'n02125311', 'n01698640', 'n04456115', 'n07711569', 'n04540053', 'n04099969', 'n09332890', 'n04074963', 'n12267677', 'n04251144', 'n03400231', 'n03255030', 'n02074367', 'n02085620', 'n01629819', 'n03649909', 'n03733131', 'n02279972', 'n07920052', 'n04259630', 'n03355925', 'n02769748', 'n02123045', 'n02395406', 'n02509815', 'n01784675', 'n09256479', 'n04023962', 'n07583066', 'n03404251', 'n03100240', 'n03617480', 'n04532670', 'n04486054', 'n07871810', 'n09246464', 'n02123394', 'n01443537', 'n04118538', 'n02415577', 'n04560804', 'n02730930', 'n02002724', 'n04356056', 'n02666196', 'n02906734', 'n03838899', 'n04366367', 'n03424325', 'n04133789', 'n02321529', 'n04285008', 'n02823428', 'n02480495', 'n03637318', 'n02231487', 'n02999410', 'n03026506', 'n07614500', 'n02423022', 'n03970156', 'n03042490', 'n04501370', 'n03388043', 'n02437312', 'n02106662', 'n04275548', 'n02841315', 'n03662601', 'n01641577', 'n02124075', 'n03599486', 'n01882714', 'n04399382', 'n03014705', 'n02917067', 'n03976657', 'n04487081', 'n03977966', 'n07579787', 'n03444034', 'n02883205', 'n02802426', 'n01944390', 'n04254777', 'n02669723', 'n02791270', 'n02056570', 'n03763968', 'n04179913', 'n02963159', 'n04596742'] ):
			print('Class-{}:{}%'.format(j,i))

		print('Completed testing the best model')

