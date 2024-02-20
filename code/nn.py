#%%
import argument_parser as ap
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error, r2_score 
from torchvision.transforms import PILToTensor
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import yaml
import random
import os
import datetime
from functools import wraps
import time
import json
import pprint

from tabular_data import ModelData

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

#%%
def data_for_nn(filename='listing.csv', filepath='../data/intermediate/airbnb-property-listings/tabular_data/', column='Price_Night', random_state=1):
    '''
    Load data from file and split into training, validation and test sets
    '''
    loader = ModelData()
    df = loader.load_tabular_data(filename, filepath)
    X, y = loader.extract_label(df, column)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test

class AirBNBDataset(Dataset):
    '''
    Custom dataset for AirBNB data, inheriting from torch.utils.data.Dataset.

    Arguments:
    Xs: pandas DataFrame, features
    ys: pandas Series, labels
    
    
    '''
    def __init__(self, Xs, ys):
        super().__init__() # Inherit from torch.utils.data.Dataset
        self.X, self.y = Xs, ys # Assign features and labels to self.X and self.y

  #  def load_tabular_data(self, filename, filepath):
   #     # 
   #     loader = ModelData()
  #      df = loader.load_tabular_data(filename, filepath)
   #     return df
    
   # def extract_label(self, df, column, numeric_only=True):
   #     loader = ModelData()
   #     features, label = loader.extract_label(df, column, numeric_only)
   #     return features, label

    def __getitem__(self, index):
        # Define behaviour of indexing
        return (torch.tensor(self.X.iloc[index], dtype=torch.float32), 
                torch.tensor(self.y.iloc[index], dtype=torch.float32).reshape(-1))
    
    def __len__(self):
        # Define behaviour of len()
        return len(self.y)


def train(model, config, train_loader, val_loader, X_train, X_val, y_train, y_val, epochs=10, random_seed=1):
    '''
    Train neural network model using the given configuration and data.
    
    Arguments:
    model: torch.nn.Module as NN class, the model to be trained
    config: dict, configuration for the model
    train_loader: torch.utils.data.DataLoader, training data
    val_loader: torch.utils.data.DataLoader, validation data
    X_train: pandas DataFrame, training features
    X_val: pandas DataFrame, validation features
    y_train: pandas Series, training labels
    y_val: pandas Series, validation labels
    epochs: int, number of epochs to train for
    random_seed: int, random seed for reproducibility

    Returns:
    tl: list, training loss for each epoch (RMSE)
    vl: list, validation loss for each epoch (RMSE)
    '''
    if config['optimiser'] == 'SGD':
        optimiser = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=config['momentum'])
    elif config['optimiser'] == 'Adam':
        optimiser = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    writer = SummaryWriter()

    tl = []
    vl = []
    for epoch in range(epochs):

        train_loss = []
        val_loss = []

        torch.manual_seed(random_seed)
        for batch in val_loader:
            features, labels = batch
            predictions = model(features)
            loss = F.mse_loss(predictions, labels)
            #writer.add_scalar('Val loss', val_loss.item(), batch_idx)
            val_loss.append(np.sqrt(loss.item()))
            #val_loss.append(rmse)
        val_loss = np.mean(val_loss)
        try:
            val_loss = np.sqrt(mean_squared_error(y_val, model(torch.tensor(X_val.values, dtype=torch.float32)).detach().numpy()))
        except:
            val_loss = np.inf
        #first loop over training batches
        for batch in train_loader:
            features, labels = batch
            predictions = model(features)
            #print(predictions[0:3])
            loss = F.mse_loss(predictions, labels)
            loss.backward() # populates gradients
            #print(loss.item())
            optimiser.step() # optimises based on gradients of model.parameters
            optimiser.zero_grad() # reset gradients as otherwise they accumulate
            #writer.add_scalar('Train loss', loss.item(), batch_idx) \
            
            train_loss.append(np.sqrt(loss.item()))
            #train_loss.append(rmse)
        train_loss = np.mean(train_loss)

        try:
            train_loss = np.sqrt(mean_squared_error(y_train, model(torch.tensor(X_train.values, dtype=torch.float32)).detach().numpy()))
        except:
            train_loss = np.inf

        tl.append(train_loss )
        vl.append(val_loss)

        writer.add_scalars(f'Loss', {'Train': train_loss}, epoch)
    return tl, vl


class NN(torch.nn.Module):
    '''
    Neural network class, inheriting from torch.nn.Module.
    
    Arguments:
    config: int, configuration index to pick the yaml config file'''
    def __init__(self, config=0):
        super().__init__()

        config = get_nn_config('nn_config_' + str(config) + '.yaml')

        self.n_features = config['n_features']
        self.width = config['hidden_layer_width']
        self.depth = config['depth']

        self.define_architecture()


    def define_architecture(self):
        '''
        Set up two or three layer deep neural network with ReLU activation functions.'''
        if self.depth == 2:
            self.layers = torch.nn.Sequential(
                        torch.nn.Linear(self.n_features, self.width),
                        torch.nn.ReLU(),
                        torch.nn.Linear(self.width, 1)
                                        )
        elif self.depth == 3:
            self.layers = torch.nn.Sequential(
                        torch.nn.Linear(self.n_features, int(self.width/2)),
                        torch.nn.ReLU(),
                        torch.nn.Linear(int(self.width/2), self.width),
                        torch.nn.ReLU(),
                        torch.nn.Linear(self.width, 1)
                                        )
            

    def forward(self, X):
        # Define forward pass
        return self.layers(X)


def get_nn_config(filename='nn_config_0.yaml'):
    # Load configuration from yaml file to python dict
    with open('nn_configs/' + filename, 'r') as file:
        config = yaml.safe_load(file)
    return config

def generate_nn_config_range(filename='nn_configs/config_ranges.yaml'):
    # Generate various configs from yaml containg ranges for hyperparameters
    with open(filename, 'r') as file:
        config_ranges = yaml.safe_load(file)
    return config_ranges
    
     

def generate_nn_configs(n_configs, ranges='./nn_configs/config_ranges.yaml', export_path='./nn_configs/'):
    # Generates a number of config files from a range of hyperparameters.
    # These are saved as yaml files in the nn_configs directory.
    # nn_configs/config_ranges.yaml contains the ranges for the hyperparameters
    config_ranges = generate_nn_config_range(ranges)
    for ix in range(n_configs):
        config = {}
        for key in config_ranges.keys():
            try:
                config[key] = random.choice(config_ranges[key])
            except:
                config[key] = config_ranges[key]
        with open(export_path + 'nn_config_' + str(ix) + '.yaml', 'w') as file:
            yaml.dump(config, file)
    

def plot_training_curve(train_loss, val_loss, title, show=False):
    # Plot training and validation curves over the epochs
    fig = plt.figure(dpi=300)
    plt.grid(True)
    plt.plot(train_loss, 'k', label='train')
    plt.plot(val_loss, 'k:', label='val')
    plt.legend()
    plot_title = title + ' training curve'
    plt.title(plot_title)
    plt.ylabel('RMSE')
    plt.xlabel('Epoch')
    if show:
        plt.show()
    return fig

def plot_validation_scatter(model, X_test, y_test, config_ext, test_loss, show=False):
    # Plot scatter of true vs predicted values for the test set
    fig = plt.figure(dpi=300)
    plt.grid(True)
    tt = torch.tensor(X_test.values, dtype=torch.float32)
    pred = model(tt)
    pred = pred.detach().numpy()
    plt.scatter(y_test,pred)
    plot_title =  config_ext + ', test loss: ' + str(test_loss)
    plt.title('Performance for ' + plot_title)
    plt.xlabel('True')
    plt.ylabel('Predicted')
    if show:
        plt.show()
    return fig

def get_test_loss(model, test_loader, X_test, y_test):
    # Calculate test loss for a given model
    test_loss = np.sqrt(mean_squared_error(y_test, model(torch.tensor(X_test.values, dtype=torch.float32)).detach().numpy()))
    print(test_loss)
    return test_loss

def datetime_dir_name(arguments=None):
    # Generate a directory name based on the current date and time
    if not arguments:
        now = datetime.datetime.now()
    else:
        now = arguments['run_time']
    return now.strftime('%Y-%m-%d_%H-%M-%S') + '/'

def train_various_configurations(arguments):
    # Train a number of models with different configurations
    savedir = datetime_dir_name(arguments)
    
    n_configs = arguments['nn_n_configs']
    epochs = arguments['nn_epochs']
    batch_size = arguments['batch_size']
    

    X_train, X_val, X_test, y_train, y_val, y_test = data_for_nn(column=arguments['label'], random_state=arguments['random_state'])
        
    train_data = AirBNBDataset(X_train, y_train)
    val_data = AirBNBDataset(X_val, y_val)
    test_data = AirBNBDataset(X_test, y_test)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)


    generate_nn_configs(n_configs)
    best_model = None
    best_loss = np.inf
    for iconfig in range(n_configs):

        print('Training model ' + str(1 + iconfig) + ' of ' + str(n_configs) + ' (index ' + str(iconfig) + ')')
        
        config_ext = str(iconfig)
        
        config = get_nn_config('nn_config_' + config_ext + '.yaml')

        model = NN()

        test_loss = get_test_loss(model, test_loader, X_test, y_test)
        test_loss_original = int(np.round(test_loss, 0))

        try:
            print('Original test loss: ' + str(test_loss_original))
        except:
            print('Original test loss: ' + 'N/A')

        start_train = time.time()
        tl, vl = train(model=model, 
                       train_loader=train_loader, 
                       val_loader=val_loader, 
                       config=config, 
                       epochs=epochs, 
                       random_seed=arguments['random_state'],
                       X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val )
        train_time = time.time() - start_train

        try:
            start_test = time.time()
            test_loss = get_test_loss(model, test_loader, X_test, y_test)
            test_time = time.time() - start_test
            test_loss_trained = int(round(test_loss, 0)) 
            #print('Trained test loss: ' + str(test_loss_trained))
        except:
            #print('Trained test loss: ' + 'N/A')
            pass

        f_train = plot_training_curve(tl, vl, config_ext)
        f_val = plot_validation_scatter(model, X_test, y_test, config_ext, test_loss_trained)

        if test_loss_trained < best_loss:
            best_loss = test_loss_trained
            best_model = iconfig

        try:
            test_pred = model(torch.tensor(X_test.values, dtype=torch.float32))
            test_pred = test_pred.detach().numpy()
            test_r2 = r2_score(y_test, test_pred)
            test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))

            train_pred = model(torch.tensor(X_train.values, dtype=torch.float32))
            train_pred = train_pred.detach().numpy()
            train_r2 = r2_score(y_train, train_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))   

            val_pred = model(torch.tensor(X_val.values, dtype=torch.float32))   
            val_pred = val_pred.detach().numpy()    
            val_r2 = r2_score(y_val, val_pred)  
            val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))

            metrics = {'train_time': train_time, 
                    'RMSE_loss': {'train': train_rmse,
                                    'val': val_rmse,
                                    'test': test_rmse}, 
                    'r2':        {'train': train_r2,
                                    'val': val_r2,
                                    'test': test_r2}, 
                        'inference_latency': test_time}
        except:
            metrics = {'Failed model': 'N/A'}
        
        pprint.pprint(metrics)

        # %%
        results_dir = '../models/' + arguments['label'] + '/nn/' + savedir + config_ext + '/'
        os.system('mkdir -p ' + results_dir)
        torch.save(model.state_dict(), results_dir + 'nn.pt')
        save_str = 'cp nn_configs/nn_config_'+config_ext+'.yaml ' + results_dir
        os.system(save_str)
        with open(results_dir + 'metrics.json', 'w') as fp:
            json.dump(metrics, fp) 
        f_train.savefig(results_dir + 'training_curve.png')
        f_val.savefig(results_dir + 'validation_scatter.png')

        print('================ Done ================')
    print('Best model index: ' + str(best_model) + ' with test loss: ' + str(best_loss))


#%%
if __name__ == '__main__':
    

    arguments = ap.make_args()

    train_various_configurations(arguments)

    
    #example = next(iter(train_loader))
    #print(example)
    #features, labels = example
    #features = features.type(torch.float32)
    ##features = features.reshape(batch_size, -1)
    #%%
    #config_ext = 'base'
    #config = get_nn_config('nn_config_' + config_ext + '.yaml')
    #model = NN()
    #get_test_loss(model, X_test, y_test)

    #model = LinearRegression()
    #model(features)
    #test_data = AirBNBDataset(X_test, y_test)
    #test_loader = DataLoader(test_data, batch_size=len(test_data))
    #for batch in test_loader:
    #        features, labels = batch
    #        predictions = model(features)
    #        loss = F.mse_loss(predictions, labels)
    #        print(np.sqrt(loss.item()))
    #%%

    #tl, vl = train(model, config, epochs=50)

    #%%


    #%%
    #test_loss = int(round(get_test_loss(model, X_test, y_test), 0))

    #f_train = plot_training_curve(tl, vl, config_ext)
    #f_val = plot_validation_scatter(model, X_test, y_test, config_ext, test_loss)

    # %%
    ##torch.save(model.state_dict(), results_dir + 'nn.pt')
    #os.system('mkdir -p ' + results_dir)
    #save_str = 'cp nn_configs/nn_config_'+config_ext+'.yaml ' + results_dir
    ####results_dir = '../models/nn/' + config_ext + '/'
    #os.system(save_str)
    #f_val.savefig(results_dir + 'validation_scatter.png')
    ##f_train.savefig(results_dir + 'training_curve.png')

    # %%
    #state_dict = torch.load('test_model.pt')
    #new_model = NN()
    #new_model.load_state_dict(state_dict)
    #train(new_model, epochs=10)

    # %%