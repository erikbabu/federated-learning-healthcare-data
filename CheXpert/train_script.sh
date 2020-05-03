#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=eb1816 # required to send email notifcations - please replace <your_username> with your college login name or email address
export PATH=/vol/bitbucket/eb1816/individual_project/venv/bin/:$PATH
source activate

if [[ $# -eq 0 ]] ; then
    echo 'Please specify name of folder to save progress as 1st argument'
    exit 1
fi

python Chexpert_SOTA/bin/train.py Chexpert_SOTA/config/config.json $1 --num_workers 4 --device_ids "0" --logtofile True

# Place below line in invocation to use pre-trained weights
# --pre_train "Chexpert_SOTA/config/pre_train.pth"

# Place below line in invocation to resume training
# --resume 1
