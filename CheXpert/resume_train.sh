#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=eb1816 # required to send email notifcations - please replace <your_username> with your college login name or email address
export PATH=/vol/bitbucket/eb1816/individual_project/venv/bin/:$PATH
source activate

if [[ $# -ne 2 ]] ; then
    echo "Usage: $0 /path/to/config.json progress_folder_name"
    exit 1
fi

python Chexpert_SOTA/bin/train.py $1 $2 --num_workers 4 --device_ids "0" --logtofile True --resume 1

# Place below line in invocation to use pre-trained weights
# --pre_train "Chexpert_SOTA/config/pre_train.pth"
