# Launch jupyter notebook from lab machine

1. ssh into lab machine (e.g. ray04) 

2. activate project virtual environment

3. set **PROJECT_DATA_BASE_DIR** environment variable showing full path of location of CheXpert data. This requires the user to have access to their own copy of the dataset, as specified in the root README file. From a lab machine terminal, to use my configuration, run  ```export PROJECT_DATA_BASE_DIR=/vol/bitbucket/eb1816/individual_project/data```

4. run ```jupyter-lab --no-browser --port=8888```

5. copy the link generated.


6. in new **local** terminal window, run ```ssh -N -f -L localhost:8888:localhost:8888 <username>@<machine_id>.doc.ic.ac.uk```

7. paste generated link in local browser

