import torch
import torch.nn as nn
import torch.nn.init as init
import os
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F

from utils.metric import IoU
# data loader
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from utils.data_loader_error import UCF_101

import videotransforms
from videotransforms import HorizontalFlip
import numpy as np
import argparse
import time
import math 

# Loading model
from utils.c3d import C3D
from models import res_34

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def rampweight(iteration):
    ramp_up_end = 32000
    ramp_down_start = 100000

    if(iteration<ramp_up_end):
        ramp_weight = math.exp(-5 * math.pow((1 - iteration / ramp_up_end),2))
    elif(iteration>ramp_down_start):
        ramp_weight = math.exp(-12.5 * math.pow((1 - (120000 - iteration) / 20000),2)) 
    else:
        ramp_weight = 1 


    if(iteration==0):
        ramp_weight = 0

    return ramp_weight


parser = argparse.ArgumentParser(
    description='Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()

parser.add_argument('--num_workers', default=8, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
# parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
#                     help='initial learning rate')
# parser.add_argument('--weight_decay', default=5e-4, type=float,
#                     help='Weight decay for SGD')
parser.add_argument('--num_epochs', default=300, type=int,
                    help='Momentum value for optim')
parser.add_argument('--learning_rate', default=0.003, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--batchSize', default=64, type=int,
                    help='Batch Size')
parser.add_argument('--save_folder', default='./weights',
                    help='Directory for saving checkpoint models')
parser.add_argument('--test_flag', default=True, help='For training or Testing')
args = parser.parse_args()


# loading the dataset Train and Test
transform_rgb = transforms.Compose([videotransforms.ClipToTensor(div_255=False)])
transform_seg = transforms.Compose([videotransforms.ClipToTensor_seg()])

location = '/home/cap6412.student5/UCF-101/'

dsetTrain = UCF_101(videoPth = location, mode = 'Train', transform_rgb = transform_rgb)#, test_loc= 'test')
trainloader = DataLoader(dsetTrain, batch_size = args.batchSize, shuffle = True, num_workers = args.num_workers)


dsetTest = UCF_101(videoPth = location, mode = 'Test', transform_rgb = transform_rgb )#, test_loc= 'test')
testloader = DataLoader(dsetTrain, batch_size = args.batchSize, shuffle = True, num_workers = args.num_workers)


#c3d = C3D().cuda(0)
c3d = res_34.model(pretrained_path='models/res_34_features.pth')



if args.test_flag != True:
    # criterion for calculating loss
    c3d.train()
    criterion_classify = nn.CrossEntropyLoss()
    criterion_regression = nn.MSELoss()
    criterion_consistency_classify = torch.nn.KLDivLoss(size_average=False, reduce=False).cuda()

    optimizer = torch.optim.SGD(c3d.parameters(), lr=args.learning_rate, momentum=0.9)

    best_loss = 10

    # Train the Model
    for epoch in range(args.num_epochs):
        if (epoch+1) % 5 == 0:
            args.learning_rate = args.learning_rate/2
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.learning_rate
        
        # metrics
        total = 0
        correct = 0
        total_iou = 0
        count = 0
        total_loss = total_loss_consis = total_loss_l = total_loss_c = 0
        c=0

        for i, (clips,boxes,labels) in enumerate(trainloader):
            count+=1            
            clips_flipped,boxes_flipped = HorizontalFlip(clips,boxes)

            # converting clips and boxes 
            clips = Variable(clips).cuda(0)
            clips_flipped = Variable(clips_flipped).cuda(0)
            
            # getting the bounding boxes average in centre frames
            loc_org = Variable(torch.tensor((np.average(boxes[:,7:9,:],axis=1)/122.0))).cuda(0)
            loc_org_flipped = Variable(torch.tensor((np.average(boxes[:,7:9,:],axis=1)/122))).cuda(0)

            # converting the label to cuda tensor
            labels_ori = labels
            labels = Variable(labels).cuda(0)


            t0 = time.time()

            # Forward + Backward + Optimize
            optimizer.zero_grad()

            conf, loc = c3d(clips)
            conf_flip, loc_flip = c3d(clips_flipped)

            conf = F.softmax(conf)
            conf_flip = F.softmax(conf_flip)

            loc_l = torch.where(loc_org>0, loc, torch.zeros(loc.shape).cuda())
            loc_flip_l = torch.where(loc_org>0, loc_flip, torch.zeros(loc_flip.shape).cuda())

            # loss for localization(regression) and classification(cross-entropy)
            loss_l = Variable(torch.cuda.FloatTensor([0]))
            loss_c = Variable(torch.cuda.FloatTensor([0]))


            # calculating loss
            outputs = conf
            loss_c = criterion_classify(outputs, labels) 

#            import ipdb;ipdb.set_trace()

            for x,x_pred in zip(loc_org,loc):
                if x.sum() > 0:
                    loss_l += criterion_regression(x.float(),x_pred) 
            loss_l *= 10
            # print(loss_l)
            #loss_l = criterion_regression(loc_org,loc_l).mul(25)
            # outputs = torch.argmax(conf,dim=1)
            # labels_onehot = torch.nn.functional.one_hot(labels,num_classes=101)

            loss_c = loss_c.div(100.0)
            # Consistency Loss Classification
            # cls_conf = torch.max(conf).clone()
            # cls_conf_flip = torch.max(conf_flip).clone()
            # cls_id_flip = torch.argmax(conf_flip).clone()

            conf_sampled_flip = conf_flip + 1e-7
            conf_sampled = conf + 1e-7
            consistency_conf_loss_a = criterion_consistency_classify(conf_sampled.log(), conf_sampled_flip.detach()).sum(-1).mean()
            consistency_conf_loss_b = criterion_consistency_classify(conf_sampled_flip.log(), conf_sampled.detach()).sum(-1).mean()
            consistency_conf_loss = consistency_conf_loss_a + consistency_conf_loss_b

            # Consistency Loss Location
            consistency_loc_loss_x = torch.mean(torch.pow(loc[:, 0] + loc_flip[:, 0], exponent=2))
            consistency_loc_loss_y = torch.mean(torch.pow(loc[:, 1] - loc_flip[:, 1], exponent=2))
            consistency_loc_loss_w = torch.mean(torch.pow(loc[:, 2] - loc_flip[:, 2], exponent=2))
            consistency_loc_loss_h = torch.mean(torch.pow(loc[:, 3] - loc_flip[:, 3], exponent=2))

            consistency_loc_loss = torch.div(
                consistency_loc_loss_x + consistency_loc_loss_y + consistency_loc_loss_w + consistency_loc_loss_h,
                4)
            # Total loss
            if loc_org.sum() > 0:
                consistency_loss = torch.div(consistency_conf_loss,2) + consistency_loc_loss
            else:
                consistency_loss = torch.div(consistency_conf_loss,2) + consistency_loc_loss

            # #print("before backward")
            # loss.backward()
            # #print("after backward")
            # optimizer.step()

            # Backward loss propogation
            ramp_weight = rampweight(epoch+1)
            consistency_loss = torch.mul(consistency_loss, ramp_weight)

#            import ipdb;ipdb.set_trace()

            # loss = loss_l + loss_c + consistency_loss
            loss = loss_l + loss_c
            if(loss.data>0):
                optimizer.zero_grad()
                loss.backward()
                # loss_c.backward()
                optimizer.step()

            t1 = time.time()

            predicted_labels = torch.argmax(conf,dim=1)
            total += labels.size(0)
            correct += (predicted_labels == labels.cuda(0)).sum()
            
            iou = IoU(loc_org,loc_l)
            if iou!= 0 :
                total_iou += iou
                #print(total_iou,c,loc_org,loc)
                c+=1    

            total_loss_c += float(loss_c.data)
            total_loss_l += float(loss_l.data)
            total_loss_consis += float(consistency_loss.data)
            total_loss += float(loss.data)
            # print(float(loss.data),float(loss_c.data),float(loss_l.data),float(consistency_loss.data), '\t\t Acc: %d/%d'%(correct,total))
            # print(total_loss,total_loss_c,total_loss_l,total_loss_consis)
            # total_loss += 1
            # print(total,correct)
            
        if float(loss.data) < best_loss:
            best_loss = float(loss.data)
            torch.save(c3d.state_dict(), '%s/set_3_res_34_sgd_%d_%f.pth'%(args.save_folder,epoch+1,float(loss.data)))

        if (epoch + 1) % 1 == 0:

            print('Epoch [%d/%d], timer: %.4f sec/batch.'
                  % (epoch + 1, args.num_epochs, (t1 - t0)))
            print('loss: %.4f , loss_c: %.4f , loss_l: %.4f , loss_con: %.4f, lr : %.4f' % (total_loss/count, total_loss_c/count, total_loss_l/count, total_loss_consis/count,float(optimizer.param_groups[0]['lr'])))

            print('Training Accuracy : %d ' % (100.0 * float(correct) / total))
            print('Training IoU : %d ' % (100.0 * float(total_iou) / c))



else:
    save_no = 6
    pertained_weights = '%s/2_res_34_sgd_%d.pth'%(args.save_folder,save_no)
    c3d.load_state_dict(torch.load(pertained_weights))

    print('Loading the saved model %s'%pertained_weights)

    c3d.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0
    total = 0
    total_iou = 0
    c=0
 
    for i, (clips,boxes,labels) in enumerate(testloader):
        #print("validation mode")
        
        clips = Variable(clips).cuda(0)
        labels = Variable(labels).cuda(0)
        loc_org = Variable(torch.tensor((np.average(boxes[:,7:9,:],axis=1)/122.0))).cuda(0)

        
        conf, loc = c3d(clips)

        predicted_labels = torch.argmax(conf,dim=1)

        total += labels.size(0)
        correct += (predicted_labels == labels.cuda(0)).sum()
        iou = IoU(loc_org,loc)
        if iou != 0: 
            c+=1
            total_iou += iou 
            #print(total_iou, c) 

    print('Testing Accuracy : %d ' % (100.0 * float(correct) / total))
    print('Testing IoU : %d ' % (100.0 * float(total_iou) / c))
