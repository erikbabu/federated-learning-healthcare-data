#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=eb1816 # required to send email notifcations - please replace <your_username> with your college login name or email address
export PATH=/vol/bitbucket/eb1816/individual_project/venv/bin/:$PATH
source activate

python Chexpert_SOTA/bin/train.py Chexpert_SOTA/config/central.json logdir --num_workers 8 --device_ids "0" --pre_train "Chexpert_SOTA/config/pre_train.pth"
