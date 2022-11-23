# AutoML lecture 2021 (Freiburg & Hannover)
## Constrained MO-DEHB

DEHB is a recent method which combines Differential Evolution and Hyperband to tackle the
Hyperparameter Optimization problem. In this work I focused on extending DEHB for the constrained multi-objective scenario
in order to optimize the Top-3 Accuracy given Precision and model size constraints in the 17 Category Flower Dataset.

## Running the optimizer

To run the optimizer simply run: <BR>
```
python AutoML.py --optimizer DEHB --eta 3 --constraint_max_model_size 2000000 --constraint_min_precision 0.39 --seed 123
```
Best configuration found when running MO-DEHB will be saved in a file named: 
```
dehb_incumbent_eta_3_precision_{precision_constraint}_model_size_{max_model_size_constraint}.json
```
## Best configuration found
The best hyperparameter configuration that I found while experimenting with different constraint values is located in:
```
final_results/dehb_incumbent_eta_3_precision_0.6_model_size_10000000.json
```
The same configuration can also be found in ``opt_cfg.json``.<BR>
I loaded this configuration in ``main.py`` and ran it by passing the argument ``use_test_data True`` and got these results:<BR><BR>
Top3-Accuracy: 0.9117647409439087<BR>
Precision Score: 0.7719432803684307<BR><BR>
When running the configuration on the validation data using CV I get these results:<BR><BR>
Top3-Accuracy: 0.8617647091547648<BR>
Precision Score: 0.6392299077796949

To reproduce these results, simply run:<BR> 
```
python AutoML.py --optimizer DEHB --constraint_max_model_size 10000000 --constraint_min_precision 0.6 --seed 123
```
## Final Project

This repository contains all things needed for the final project.
Your task is to optimize a networks accuracy (maximize) with the following constraints: network size (upper bound) and precision (lower bound),
such that your network reaches a maximal accuracy under the given constraints.
The constraint values that we provide in the scipt are only an example. We will reevaluate your algorithms with different constraints after your submission. That is, you cannot 
simply satisfy the model size constraint with a specially designed configuration space.

## Repo structure
* [micro17flower](micro17flower) <BR>
  contains a downsampled version of a [flower classification dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/17/index.html).
  We cropped ans resized it such that the resulting images are of size 16x16. You are only allowed to run your optimizer with the data in [train](micro17flower/train)

* [src](src) Source folder      
    * [bohb.py](src/bohb.py) <BR>
      contains a simple example of how to use BOHB in SMAC on the dataset. This implementation does not consider any constraint!
    
    * [cnn.py](src/cnn.py)<BR>
      contains the source code of the network you need to optimize. It optimizes the top-3 accuracy.
    
    * [main.py](src/main.py)<BR>
      contains an example script that shows you how to instantiate the network and how to evaluate metrics required 
      in this project. This file also gives you the **default configuration** that always has to be in yourserch space.

    * [AutoML.py](src/AutoML.py) <BR>
      You will implement your own optimizers under this script. It should at least be able to receive the constraint values (network size and precision) and 
      store the optimal configuration to "opt_cfg.json"
    
    * [utils.py](src/utils.py)<BR>
      contains simple helper functions for cnn.py
      
    * [optimizer.py](src/optimizer.py)<BR>
      contains the implementation of the DEHB optimizer
