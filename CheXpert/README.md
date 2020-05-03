# 1. Download CheXpert dataset 
Use link sent in email from Stanford

# 2. Clone repository of current SOTA solution
https://github.com/jfhealthcare/Chexpert

Rename folder to Chexpert_SOTA.

Remove all git artifacts (to bypass lack of root access on lab machines).

# 3. Customise for own file paths and requirements
In ```config/``` create a json file and point it to the location of train and dev csv files. 

# 4. Run centralised training
```./train_script.sh```