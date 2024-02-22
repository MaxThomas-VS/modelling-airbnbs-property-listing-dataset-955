# Modelling property listings from AirBnB

## Summary
Various machine learning models are used to model AirBnB property listing data. Each is tuned over a wider range of hyperparameters. The nightly price is predicted with seven regression models. Gradient Boosting Regression performs best (test set RMSE = 89). A fully connected neural network is also used to predict the nightly price. The neural network is tested over 128 configurations. The best performing neural net had 2 hidden layers with 16 and 32 nodes, a learning rate of 0.1, and a momentum of 0.99 - producing a test set RMSE 98. Four classification models are used to predict the category of each listing. Logistic regression performs best with an accuracy accuracy of 0.79. Finally, the code is resused to predict the number of bedrooms of each listing using the classification algorithms. Random Forest Classifier performs best with an accuracy of 0.8.

## Results
Results are presented in full [here](/results/results.ipynb).  

## Directory structure
The key files are ```/code/modelling.py``` and ```/code/nn.py```. The regression and classification models are tuned and evaluated by ```modelling.py```, which also calls ```nn.py``` to build and train the neural network. ```modelling.py``` takes command line options to cusomise it's behavior, and all relevant commands are detailed in the ```/code/run``` scripts. For example, 
```
python -W ignore modelling.py --do_train=True --do_evaluation=True --ml_model=sgdregressor,decisiontreeregressor,randomforestregressor,gradientboostingregressor,svr,kernelridge,baeysianridge
```
will train and evaluate all the regression models (the ```-W ignore``` flag supresses errors from bad hyperparameter choices).

```/data``` stores AirBnB data, ```/models``` stores trained models, and ```/setup``` contains the required conda environment.

## Replication
To replicate the results, first create the environment with 
```
conda env create -f setup/environment.yaml
```
Next, execute *run* to load and clean the data, tune a suite of regression, classification, and neural network models to predict the price of listings.
```
cd code
./run
```
