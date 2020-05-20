#!/bin/bash
export PATH=/vol/bitbucket/eb1816/individual_project/venv/bin/:$PATH
source activate

if [[ $# -ne 4 ]] ; then
    echo "Usage: $0 path/to/progress/folder path/to/results/folder result_prefixes <use_fl>"
    exit 1
fi

if [ -d "$1" ] && [ -d "$2" ]; then

    dir=$(pwd)

    # Create best model checkpoint file
    cd $1
    cp "best1.ckpt" "best.ckpt"
    echo "Created model checkpoint file"

    # Perform evaluation
    echo "Running model evaluation..."
    python classification/bin/test.py --fl $4
    python classification/bin/roc.py $3
    echo "Completed model evaluation"

    cd $dir
    
    # Move results to corresponding folder
    cp -r "$1/test/." $2

    # Clean up
    rm -rf "$1/test/"
    rm "$1/best.ckpt"

    # Run performance profile
    python Chexpert_SOTA/bin/performance_profile.py --cfg_path "$2/config.json" --file_name "$2/profile_results"
else
    echo "One or more arguments passed are not valid directories";
    exit 1
fi
