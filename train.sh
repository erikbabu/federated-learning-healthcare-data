#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --exclude=sicklebill
#SBATCH --mail-type=ALL
#SBATCH --mail-user=eb1816

export PATH=/vol/bitbucket/eb1816/individual_project/venv/bin/:$PATH
source activate
source source /vol/cuda/10.1.105-cudnn7.6.5.32/setup.sh
TERM=vt100
/usr/bin/nvidia-smi
uptime

if [[ $# -ne 3 ]] ; then
    echo "Usage: $0 /path/to/config.json progress_folder_name <use_fl>"
    exit 1
fi

python Chexpert_SOTA/bin/train.py $1 $2 --num_workers 4 --device_ids "0" --fl $3 --logtofile True

# Place below line in invocation to use pre-trained weights
# --pre_train "Chexpert_SOTA/config/pre_train.pth"