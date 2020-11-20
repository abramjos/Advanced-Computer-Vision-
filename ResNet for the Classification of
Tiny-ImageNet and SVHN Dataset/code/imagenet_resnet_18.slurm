#!/bin/bash
#SBATCH --time=2-12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --output=slurm_out/IN_res_18.out
#SBATCH --job-name=IN_RES18

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
#module load cuda/cuda-10.0
#module load tensorflow/tensorflow-1.6.0

#source activate aj_tf
#python -c "import tensorflow as tf; tf.Session()"
# Activate the GPU version of TensorFlow
source activate pyt

echo
echo " Started Training"
echo "=================================================================================================="
python imagenet.py --length 500 --epoch 5000 --model_name resnet_18 --log-interval 100
echo "=================================================================================================="

echo "Training Complete"


# You're done!
echo "Ending script..."
date


