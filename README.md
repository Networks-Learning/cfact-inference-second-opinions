# cfact-label-inference

## Install and download prerequisites

Install required packages with pip install -r requirements.txt

## Running the Experiment on Synthetic Data

- Run "synthetic_experiment.py"
- All experimental results will be stored in directory "./results_synthetic"
- Run "evaluation_synthetic.py" to generate the plots from the evaluation results
 
## Running the Experiment on Real Data 

### Prerequisite

Download CIFAR-10H dataset into the directory "./data" from https://github.com/jcpeterson/cifar-10h
 
### Generating the preprocessed data used

- Run "get_features_vgg19.py" to generate the features of the data with VGG19
- Run "join_feat_cifar10h_labels.py" to join the features and human label predictions of data set CIFAR-10H in one dataframe
- Run "preprocessed_data.py" to resample the data and experts to obtain a higher disagreement ratio
- The training and test set are stored in directory "./data", features and labels are stored in separate matrices
 
### Running the experiment
- Run "real_experiment.py"
- All experimental results will be stored in directory "./results_real"
- Run "evaluation_real.py" to generate the plots from the evaluation results

