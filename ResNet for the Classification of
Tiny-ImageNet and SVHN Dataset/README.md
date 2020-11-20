
'''

Assignment 1 for Advanced Computer Vision- CAP 6412.
Abraham Jose
01-30-2020
abraham@knights.ucf.edu 

'''

#DIRECTORY STRUCTURE

---- model/
  |  |- imagenet.py 
  |	 |- svhn.py
  |   - slurm_out/
  |- output/
  |  |- model-{model_name}/
  |	  		|- runs/
  |	  		|- checkpoints/
  |	  		- final_weigths
   - data/
   	 |- svhn/
   	 |	 |- test_32x32.mat
   	 |	  - train_32x32.mat
   	  - tiny-imagenet-200/
   	  	 |- Train/ 
   	  	 |   - 200 folders for 200 classes and images 
   	  	  - Test/
   	  	  	 - 200 folders for 200 classes and images

#CREATING ENVIRONMENT
##Creating Environment for training and testing the model

locate the environment.yml file and execute the following command

''
conda env create -n {environment_name} -f environment.yml
''
example: 

*conda env create -n assignment_1 -f environment.yml*


#IMAGENET DATASET
imagenet.py
To Train imagenet dataset, convert the images to image folders(Test and Train dataset). The model will create 20% split on the Training dataset for validation.

##Usage
supported resnet models = resnet18, resnet34, resnet50, resnet101

''
python imagenet.py --batch-size --epochs --lr --momentum --log-interval --save-model --test_model --length  --model_name
''
--batch-size   : Training batch size (default 16)
--epochs       : No of epochs for training the model (default 10)
--lr           : Learning rate (default 0.01)
--momentum     : Momentum for the training(default 0.5)
--log-interval : Logging interval(default 20)
--save-model   : Saving the final model(default True)
--test_model   : Flag for Testing the model(default False)
--length       : Length of the dataset(default 500/class)
--model_name   : model_name('resnet_18','resnet_34','resnet_50','resnet_101')

example:
*python imagenet.py --length 500 --epoch 5000 --model_name resnet_101 --log-interval 100
python imagenet.py --length 350 --epoch 5000 --model_name resnet_18 --log-interval 100*


#SVHN DATASET
svhn.py
To Train svhn dataset, make sure that the train_32x32.mat and test_32x32.mat(format 2) is available in data/svhn folder. The model will create 20% split on the Training dataset for validation.

##Usage
supported resnet models = resnet18, resnet34, resnet50, resnet101

''
python svhn.py --batch-size --epochs --lr --momentum --log-interval --save-model --test_model 
''
--model_name   : model_name('resnet_18','resnet_34','resnet_50','resnet_101','svhn_resnet_101')
--batch-size   : Training batch size (default 16)
--epochs       : No of epochs for training the model (default 10)
--lr           : Learning rate (default 0.01)
--momentum     : Momentum for the training(default 0.5)
--log-interval : Logging interval(default 20)
--save-model   : Saving the final model(default True)
--test_model   : Flag for Testing the model(default False)

example:
*python svhn.py --length 500 --epoch 5000 --model_name svhn_resnet_101 --log-interval 100*

#TRAINING IN ARCC-CLUSTER
For training the model in ARCC Cluster, the following command can be used using the slurm files

*sbatch imagenet_resnet_18.slurm
sbatch imagenet_resnet_101.slurm*
The slurm operation requires the slurm_out/ folder in the code folder.


#TESTING 
The model after training will create the final checkpoints as *'.pth.tar'* in output/model-{model_name}/. Once the training is completed, use the following command to test the model in the testing dataset. The output is error rate in each class/category and the corresponding confusion matrix and accuracy for the model.

*python imagenet.py --length 500 --model_name resnet_101 --test_model False*
*python svhn.py --model_name svhn_resnet_101 --test_model False*
