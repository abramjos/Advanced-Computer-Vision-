#!/bin/bash
#SBATCH --time=4-11:59:00
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --output=slurm_out/IN_res_101.out
#SBATCH --job-name=IN_RES101

# Give this process 1 task (per GPU, but only one GPU), then assign eight 8per task
# (so 8 cores overall).  Then enforce that slurm assigns only CPUs that are on the
# socket closest to the GPU you get.

# If you want two GPUs:
# #SBATCH --ntasks=2
# #SBATCH --gres=gpu:2


# Output some preliminaries before we begin
date
echo "Slurm nodes: $SLURM_JOB_NODELIST"
NUM_GPUS=`echo $GPU_DEVICE_ORDINAL | tr ',' '\n' | wc -l`
echo "You were assigned $NUM_GPUS gpu(s)"

# Load the TensorFlow module
module load anaconda

# Activate the GPU version of TensorFlow
source activate pyt

echo
echo " Started Training"
echo "=================================================================================================="
echo "Training 500.."
python imagenet.py --length 500 --epoch 5000 --model_name resnet_101 --log-interval 100

echo "Training 350.."
python imagenet.py --length 350 --epoch 5000 --model_name resnet_101 --log-interval 100

echo "Training 150.."
python imagenet.py --length 150 --epoch 5000 --model_name resnet_101 --log-interval 100

echo "=================================================================================================="

echo "Training Complete"


# You're done!
echo "Ending script..."
date


