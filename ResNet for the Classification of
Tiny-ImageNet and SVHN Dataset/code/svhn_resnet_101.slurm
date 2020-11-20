#!/bin/bash
#SBATCH --time=2-12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --output=slurm_out/SVHN_res.out
#SBATCH --job-name=SVHN_RES

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
python svhn.py --epoch 5000 --model_name svhn_resnet_101 --log-interval 10
echo "=================================================================================================="

echo "Training Complete"


# You're done!
echo "Ending script..."
date


