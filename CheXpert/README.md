# Download CheXpert dataset 
Use link sent in email from Stanford

# Clone repository of current SOTA solution
https://github.com/jfhealthcare/Chexpert

Rename folder to Chexpert_SOTA.

Remove all git artifacts (to bypass lack of root access on lab machines).

# Customise for own file paths and requirements
Create a json file (similar to that in ```config/example.json```) and point it to the location of train and dev csv files. 

# Run centralised training
```./train.sh <path/to/config.json> <folder_name>```

```folder_name``` is where the progress will be saved.

# Resume centralised training
```./resume_train.sh <path/to/config.json> <folder_name>```

```folder_name``` is where the progress from the previous run was stored. 

**Note**: The same config file and folder name must be used between different runs!

# Profile and evaluate centralised model

A script ```evaluate.sh``` exists to first evaluate the model and then run system profiling. 

Example invocation:
```./evaluate.sh central_full/ results/100_percent/full/ central_100 False``` 

Where ```central_full/``` is the location of the folder containing the saved progress of the model being trained, ```results/100_percent/full/``` is the location of the folder to save the generated charts (and also the location of the corresponding config.json file), ```central_100``` is the prefix to give the auc charts generated, and the final argument specifies whether or not to train using FL. 

To run the profiler separately: 


```python Chexpert_SOTA/bin/performance_profile.py --cfg_path <path/to/cfg.json> --filename <filename>```

**Note:** 

- Default filename is ```profile_results.txt```.
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


# Partition data into different institution CSV files

Run the ```quantitative_sampling.ipynb``` notebook in ```notebooks/```. Further instructions found in folder README. 

# Customise for own file paths and requirements for FL
Create a json file (similar to that in ```config/example_FL.json```) and point it to the location of train and dev csv files.

# Run FL training
```./train.sh <path/to/config.json> <folder_name> True```

See *Run Centralised training* section for argument descriptions. 

# Resume FL training
**Not yet implemented**