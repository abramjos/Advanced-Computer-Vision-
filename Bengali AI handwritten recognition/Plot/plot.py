import numpy as np
import matplotlib.pyplot as plt
import re

split_key = ['train_loss: ', 'train_score: ', 'T Acc_G: ', 'T Acc_V: ', 'T Acc_C: ', 'valid_loss: ', 'valid_score: ', 'V Acc_G: ', 'V Acc_V: ', 'V Acc_C: ']

def read_file(file_path):
	with open(file_path,'r') as f:
		data = f.readlines()
	data_x = {}
	for i in data:
		if i[:4] == 'FOLD':
			key = int(re.search(r'[0-9]',i).group(0))
			data_x[key] = []

		if re.search(r'\[../50]',i):
			data_x[key].append(i)
	# return(data_x)

	data_np = {}
	for key,data_fold in data_x.items():
		data_np[key] = []
		for i in data_fold:
			line = i.replace('\n','')
			data_np[key].append(parse_line(line)[1])
		data_np[key] = np.array(data_np[key])
	return(data_np)

def parse_line(line):
	d_line = []
	try:
		for key in split_key:
			d_line.append(float(line.split(key)[1].split(' ')[0].replace('V','')))
		return(True, np.array(d_line))
	except:
		return(False, None)




file_path_34 = 'train_res34.out'
data_4_fold_34 = read_file(file_path_34)

file_path_50 = 'train_res50.out'
data_4_fold_50 = read_file(file_path_50)

file_path_101 = 'train_res101.out'
data_4_fold_101 = read_file(file_path_101)

##################################################################################
model_name = ['Resnet 18','Resnet 30','Resnet 50','Resnet 101']
model_params = [11689512,21797672,25557032,44549160]
model_params = [i/1000000 for i in model_params]
model_tt_model = [1241,2621,2621,0]
model_tt_image = [430,967,1348]
model_image = [120,280,400]
color=['r','g','b','y']




fold=3
x_34,x_50,x_101 = data_4_fold_34[fold],data_4_fold_50[fold],data_4_fold_101[fold]
# Plot 1
# plt.subplot(121)
fig, ax = plt.subplots(2,1)
for i,ver,d in zip(range(3),[34,50,101],[x_50,x_34,x_101]):
	for j,ij,jj in zip(range(2),[[0,5],[1,6]],['Loss','Score']):
		ax[j].plot(d[:,ij[0]],c=color[i], label = '%s- ResNet%d'%(jj,ver))
		ax[j].plot(d[:,ij[1]],c=color[i],linestyle='--')

ax[0].legend()
ax[1].legend()
ax[0].set_ylabel('Loss')
# ax[0].set_xlabel('No of Epochs')
ax[1].set_ylabel('Score')
ax[1].set_xlabel('No of Epochs')
ax[0].set_title('Loss of ResNet 34/50/101 | Training and Validation(--)')
ax[1].set_title('Score of ResNet 34/50/101 | Training and Validation(--)')
plt.show()


##################################################################################

fold=3
x_34,x_50,x_101 = data_4_fold_34[fold],data_4_fold_50[fold],data_4_fold_101[fold]
# Plot 1
# plt.subplot(121)
fig, ax = plt.subplots(3,1)
for i,ver,d in zip(range(3),[34,50,101],[x_34,x_50,x_101]):
	for j,ij,jj in zip(range(3),[[2,7],[3,8],[4,9]],['Acc_G','Acc_V','Acc_C']):
		ax[j].plot(np.array(d[:,ij[0]])*100,c=color[i], label = '%s- ResNet%d'%(jj,ver))
		# ax[j].plot(d[:,ij[1]],c=color[i],linestyle='--')

ax[0].legend()
ax[1].legend()
ax[2].legend()
ax[0].set_ylabel('Accuracy of Grapheme')
# ax[0].set_xlabel('No of Epochs')
ax[1].set_ylabel('Accuracy of Vowel')
# ax[1].set_xlabel('No of Epochs')
ax[2].set_ylabel('Accuracy of Consonant')
ax[2].set_xlabel('No of Epochs')
ax[0].set_title('Accuracy Grapheme Root | Training ')
ax[1].set_title('Accuracy Vowel | Training ')
ax[2].set_title('Accuracy Consonant | Training ')
plt.show()


##################################################################################
# plt.xlabel('No of Training samples')
plt.ylabel('Training Accuracy')
plt.legend([str(i)+' training samples' for i in model_image],scatterpoints=1, loc='lower left', ncol=3, fontsize=8)


fig, ax = plt.subplots(2,1)
for fold in range(4):
	x_34,x_50 = data_4_fold_34[fold],data_4_fold_50[fold]
	ax[0].plot(x_50[:,1],c=color[fold], label = 'train_score fold-%d'%fold)
	ax[0].plot(x_50[:,6],c=color[fold],linestyle='--', label = 'valid_score fold-%d'%fold)
	ax[1].plot(x_50[:,0],c=color[fold], label = 'train_loss fold-%d'%fold)
	ax[1].plot(x_50[:,5],c=color[fold],linestyle='--', label = 'valid_loss fold-%d'%fold)
ax[0].set_title('Training and Validation Score/Recall in 4 Folds')
ax[1].set_title('Training and Validation Loss  in 4 Folds')
ax[0].set_ylabel('Score')
ax[0].set_xlabel('No of Epochs')
ax[1].set_ylabel('Loss')
ax[1].set_xlabel('No of Epochs')
plt.legend()
plt.show()

plt.legend([str(i)+' training samples' for i in model_image],scatterpoints=1, loc='lower left', ncol=3, fontsize=8)
plt.show()


##################################################################################
fig, ax = plt.subplots(2,1)
for fold in range(4):
	x_34,x_50 = data_4_fold_34[fold],data_4_fold_50[fold]
	ax[0].plot(x_50[:,1],c=color[fold], label = 'train_score fold-%d'%fold)
	ax[0].plot(x_50[:,6],c=color[fold],linestyle='--', label = 'valid_score fold-%d'%fold)
	ax[1].plot(x_50[:,0],c=color[fold], label = 'train_loss fold-%d'%fold)
	ax[1].plot(x_50[:,5],c=color[fold],linestyle='--', label = 'valid_loss fold-%d'%fold)
ax[0].set_title('Training and Validation Score/Recall in 4 Folds')
ax[1].set_title('Training and Validation Loss  in 4 Folds')
ax[0].set_ylabel('Score')
ax[0].set_xlabel('No of Epochs')
ax[1].set_ylabel('Loss')
ax[1].set_xlabel('No of Epochs')
plt.legend()
plt.show()

plt.legend([str(i)+' training samples' for i in model_image],scatterpoints=1, loc='lower left', ncol=3, fontsize=8)
plt.show()


##################################################################################

import json
with open('sr.json','r') as f:
    data = json.load(f)
k = {i:[] for i in data[0].keys()}
for i in data:
    for j in i.keys():
        k[j].append(i[j])
data = k
len(data[0].keys())

'valid/loss_vowel','valid/loss', 'valid/loss_grapheme'
 'train/acc_grapheme', 'valid/recall', 'train/loss_consonant', 'epoch', 'valid/acc_grapheme', 'train/loss', 'train/loss_vowel', 'valid/acc_consonant', 'train/recall', 'train/acc_consonant', 'train/loss_grapheme', 'lr', 'valid/acc_vowel', 'train/acc_vowel', 'elapsed_time', 'valid/loss_consonant'

##################################################################################
fig, ax = plt.subplots(3,1)

ax[0].plot(np.array(data['train/acc_grapheme'])*100.0,c=color[1], label = 'acc_grapheme')
ax[0].plot(np.array(data['valid/acc_grapheme'])*100.0,c=color[1], linestyle='--')
ax[0].plot(np.array(data['train/acc_consonant'])*100.0,c=color[2], label = 'acc_consonant')
ax[0].plot(np.array(data['valid/acc_consonant'])*100.0,c=color[2], linestyle='--')
ax[0].plot(np.array(data['train/acc_vowel'])*100.0,c=color[0], label = 'acc_vowel')
ax[0].plot(np.array(data['valid/acc_vowel'])*100.0,c=color[0], linestyle='--')
ax[0].legend()
ax[1].plot(data['train/recall'],c=color[1], label = 'train_recall')
ax[1].plot(data['valid/recall'],c=color[1], linestyle='--',  label = 'valid_recall')
ax[1].legend()
ax[2].plot(data['train/loss'],c=color[3], label = 'train_loss')
ax[2].plot(data['valid/loss'],c=color[3], linestyle='--',  label = 'valid_loss')
ax[2].legend()

# ax[0].set_title('Accuracy')
# ax[1].set_title('Recall')
# ax[2].set_title('Loss')

for i,x in zip(range(3),['Accuracy','Recall/Score','Loss']):
	ax[i].set_ylabel('%s'%x)
	ax[i].set_xlabel('No of Epochs')
	ax[i].set_xlim(0,80)
	ax[i].set_xticks([i*2 for i in range(0,40)])


ax[0].set_title('Training and Validation profiles')
plt.show()
