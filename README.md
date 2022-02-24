# cfact-label-inference

## Install and download prerequisites

-installing required packages: pip install -r requirements.txt

## Running the Experiment on Synthetic Data

- run "synthetic_experiment.py"
- all experimental results will be stored in directory "./results_synthetic"
- run "evaluation_synthetic.py" to generate the plots from the evaluation results
 
## Running the Experiment on Real Data 

### Prerequisite
-download CIFAR-10H dataset into the directory "./data" from https://github.com/jcpeterson/cifar-10h
 
### Generate the preprocessed data used

-run "get_features_vgg19.py" to generate the features of the data with VGG19
-run "join_feat_cifar10h_labels.py" to join the features and human label predictions of data set CIFAR-10H in one dataframe
-run "preprocessed_data.py" to resample the data and experts to obtain a higher disagreement ratio
- the training and test set are stored in directory "./data", features and labels are stored in separate matrices
 
### Run the experiment
- run "real_experiment.py"
- all experimental results will be stored in directory "./results_real"
- run "evaluation_real.py" to generate the plots from the evaluation results

