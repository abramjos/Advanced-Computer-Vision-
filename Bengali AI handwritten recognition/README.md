#Abalation study

check folder ResNet_Ablation and README @ ResNet_Ablation/README.md for ablation study


#SE-ResNeXT
Consider changing the configuration file based on the user preference
batch_size=2 #bathc size
device='cuda:0' # CPUs available
out='.' # out dir
image_size=128 # image size.(128,128)
arch='pretrained' # to load pretrained model
model_name='se_resnext50_32x4d' # to load model name


datadir = Path('../input/bengaliai-cv19') # data dir 
featherdir = Path('../input/bengaliaicv19feather') # for feather low weight files 
outdir = Path('./out_SR128x64_8/')


Use script 
python train.py for training the model

for training the the server use
sbatch train.slurm 
