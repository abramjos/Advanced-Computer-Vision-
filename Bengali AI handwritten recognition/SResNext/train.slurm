#!/bin/bash
#SBATCH --time=3-11:59:00
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_out/train_SResNeXt_8.out
#SBATCH --job-name=AJ_SR_8

# Give this process 1 task (per GPU, but only one GPU), then assign eight 8per task
# (so 8 cores overall).  Then enforce that slurm assigns only CPUs that are on the
# socket closest to the GPU you get.

echo "Slurm nodes: $SLURM_JOB_NODELIST"
NUM_GPUS=`echo $GPU_DEVICE_ORDINAL | tr ',' '\n' | wc -l`
echo "You were assigned $NUM_GPUS gpu(s)"

# Load the TensorFlow module
module load anaconda

# Activate the GPU version of TensorFlow
source activate pyt

echo " Started Training"
echo "=================================================================================================="
python train.py
echo "=================================================================================================="

echo "Training Complete"


# You're done!
echo "Ending script..."
date


