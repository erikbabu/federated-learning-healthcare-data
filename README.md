# Download CheXpert dataset 
Use link sent in email from Stanford

# Clone repository
The majority of the model code is sourced from this repository: https://github.com/jfhealthcare/Chexpert. We adapt it to perform Federated Learning and track computational and communication overhead during the model training phase. 

# Activate Virtual Environment
To use my virtual environment from a DoC machine, simply run the following command: 

```source /vol/bitbucket/eb1816/individual_project/venv/bin/activate```

To create the virtual environment yourself, in the project root directory, run the following: 

```python3 -m venv <your_venv_name>```

```source <your_venv_name>/bin/activate```

```pip install -r requirements.txt```

This can take several minutes. 


# Customise for own file paths and requirements
Create a json file (similar to that in ```config/example.json```) and point it to the location of train and dev csv files. 

**NOTE**: You must have your own copy of the CheXpert dataset. Mine cannot be used by anybody else. According to the Stanford University School of Medicine CheXpert Dataset Research Use Agreement:

>YOU MAY NOT DISTRIBUTE, PUBLISH, OR REPRODUCE A COPY of any portion or all of the CheXpert Dataset to others without specific prior written permission from the School of Medicine.

For this reason, I am the only person with rwx permissions on the dataset folder in my bitbucket.

# Run centralised training
```./train.sh <path/to/config.json> <folder_name> False```

```folder_name``` is where the progress will be saved, the final argument specifies whether or not to train the model using FL. 

# Resume centralised training
```./resume_train.sh <path/to/config.json> <folder_name>```

```folder_name``` is where the progress from the previous run was stored. 

**Note**: The same config file and folder name must be used between different runs!

# Partition data into different institution CSV files

Run the ```quantitative_sampling.ipynb``` notebook in ```notebooks/```. Further instructions found in folder README. 

# Customise for own file paths and requirements for FL
Create a json file (similar to that in ```config/example_FL.json```) and point it to the location of train and dev csv files.

The difference between this and the centralised json config files are: 

* The inclusion of the fl_technique field. Choose from "FedAvg", "WFedAvg" and "FedProx"
* The inclusion of the local_epoch field. The original epoch field is kept for backwards compatibility and will now represent the number of communication rounds.
* The train_csv folder is now a list of file paths representing the train files for the different institutions.
* The inclusion of a train_proportions field, representing the splits of data.
* The inclusion of a mu field, representing the mu value to use in FedProx (not necessary to add if not using FedProx).

# Run FL training
```./train.sh <path/to/config.json> <folder_name> True```

See *Run Centralised training* section for argument descriptions. 

# Resume FL training
**Not yet implemented**

# Profile and evaluate models

A script ```evaluate.sh``` exists to first evaluate the model and then run system profiling. 

Example invocation:
```./evaluate.sh central_full/ results/100_percent/full/ central_100 False``` 

Where ```central_full/``` is the location of the folder containing the saved progress of the model being trained, ```results/100_percent/full/``` is the location of the folder to save the generated charts (and also the location of the corresponding config.json file), ```central_100``` is the prefix to give the auc charts generated, and the final argument specifies whether or not to evaluate a model trained using FL. 

To run the profiler separately: 


```python Chexpert_SOTA/bin/performance_profile.py --cfg_path <path/to/cfg.json> --filename <filename>```

**Note:** 

- Default filename is ```profile_results.txt```.
- It will create a ```profile_results.txt``` file for every client if evaluating a model trained using FL.
- Disregard the following warnings from the profiler for the following classes:

    - <class 'model.backbone.densenet._DenseLayer'>
    - <class 'model.backbone.densenet._DenseBlock'>
    - <class 'model.backbone.densenet._Transition'>
    - <class 'torch.nn.modules.container.Sequential'>
    - <class 'model.backbone.densenet.DenseNet'>
    - <class 'model.classifier.Classifier'>
    - <class 'model.attention_map.CAModule'>
    - <class 'model.attention_map.SAModule'>
    - <class 'model.attention_map.Conv2dNormRelu'>
    - <class 'model.attention_map.FPAModule'>
    - <class 'model.attention_map.AttentionMap'>

    This is because these classes are simply wrappers of subclasses, which themselves are profiled correctly.

# Experimental Reproducibility and Transparency

For transparency, the log files of each experiment run on the full dataset are available in the ```experiments/``` folder. The evaluation for all experiments is placed in the ```results/``` folder, along with a file for each experiment detailing what we are investigating. Additionally, the ```config.json``` files are also present for each experiment in the results folders. To reproduce the results obtained, run the train script, using these ```config.json``` files as the configuration parameter. 

The checkpoints of the best performing models are also available, though not on the git repository as the files were too large. To run the evaluation script on the models generated by the experiments, run the evaluation script, but use the path ```/vol/bitbucket/eb1816/individual_project/federated-learning-healthcare-data/CheXpert/experiments/<experiment_folder_name>/``` as the saved progress folder location parameter. 

**Note**: To run the evaluation as detailed above, you need to be connected to the college network or running the commands on a DoC machine, otherwise you will not be able to access the files on my bitbucket. 

### **Results/Experiment folder naming convention**
To assist with quickly finding a corresponding experiment/result folder, we follow this naming convention:

- ```central_sampled_<n>```: The benchmark model trained on the n% sample of the aggregated data

- ```central_sampled_<n>_<split_1>_..._<split_k>```: The baseline models trained on the n% sample of the aggregated data, where institution ```i``` contributes ```<split_i>``` of the total data

- ```central_full```: The benchmark model trained on 100% of the aggregated data

- ```central_full_<split>_<institution_id>```: The baseline model for institution ```<institution_id>``` trained on ```<split>```% of the entire dataset

- ```<n>_<splits>_<fl_technique>```: FL model trained on ```<n>```% sample of the full dataset, using ```<fl_technique>```, where the data is partitioned according to ```<splits>```. 


**NOTE:** The results shown in the final report are from the ```central_full```, ```central_full_<split>_<institution_id>``` and ```100_<splits>_<fl_technique>``` folders i.e. the benchmark, baseline and FL experiments run on the full dataset.
