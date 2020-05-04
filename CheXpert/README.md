# 1. Download CheXpert dataset 
Use link sent in email from Stanford

# 2. Clone repository of current SOTA solution
https://github.com/jfhealthcare/Chexpert

Rename folder to Chexpert_SOTA.

Remove all git artifacts (to bypass lack of root access on lab machines).

# 3. Customise for own file paths and requirements
In ```config/``` create a json file and point it to the location of train and dev csv files. 

# 4a. Run centralised training
```./train.sh <folder_name>```

```folder_name``` is where the progress will be saved.

# 4b. Resume centralised training
```./resume_train.sh <folder_name>```

```folder_name``` is where the progress from the previous run was stored. 

**Note**: The same folder name must be used between different runs!

# 5. Run profiler
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


# 6. Partition data into different institution CSV files

Run the ```quantitative_sampling.ipynb``` notebook in ```notebooks/```. Further instructions found in folder README. 