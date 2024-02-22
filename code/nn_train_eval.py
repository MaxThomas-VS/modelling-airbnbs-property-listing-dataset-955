#%%
import datetime
import os
import joblib
import numpy as np
import torch
from types import SimpleNamespace    
import yaml
import json
from sklearn.metrics import r2_score
import time
import matplotlib.pyplot as plt
from argument_parser import make_args

from tabular_data import get_tabular_data

def datetime_dir_name(arguments=None):
    # Generate a directory name based on the current date and time
    if not arguments:
        now = datetime.datetime.now()
    else:
        now = arguments['run_time']
    return now.strftime('%Y-%m-%d_%H-%M-%S') + '/'

def make_results_dir(arguments):
    # Make a directory to store the results of the current run
    results_dir = '../models/' + arguments['label'] + '/nn/' + datetime_dir_name(arguments)
    os.system('mkdir -p ' + results_dir)
    return results_dir

def make_datasets(arguments):
    # Load the data and make datasets
    X_train, X_test, y_train, y_test = get_tabular_data(arguments)
    return X_train, X_test, y_train, y_test

def save_data_to_results(X_train, X_test, y_train, y_test, results_dir):
    # Save the data to the results directory
    savedir = results_dir + 'data/'
    os.system('mkdir -p ' + savedir)
    joblib.dump(X_train, savedir + 'X_train.pkl')
    joblib.dump(X_test, savedir + 'X_test.pkl')
    joblib.dump(y_train, savedir + 'y_train.pkl')
    joblib.dump(y_test, savedir + 'y_test.pkl')

def load_data_from_results(results_dir):
    savedir = results_dir + 'data/'
    data = {}
    data['X_train'] = joblib.load(savedir + 'X_train.pkl')
    data['X_test'] = joblib.load(savedir + 'X_test.pkl')
    data['y_train'] = joblib.load(savedir + 'y_train.pkl')
    data['y_test'] = joblib.load(savedir + 'y_test.pkl')
    return SimpleNamespace(**data)

def make_dataloaders(arguments, X_train, X_test, y_train, y_test):
    random_seed = arguments['random_state']
    batch_size = arguments['batch_size']
    val_split_size = arguments['nn_val_split_size']

    dataset = AirBNBDataset(X_train, y_train)

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(val_split_size * dataset_size))
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    valid_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                            sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler)

    # define loaders for evaluating (where all data is used)
    train_eval = torch.utils.data.DataLoader(dataset, batch_size=len(train_indices), 
                                            sampler=train_sampler)
    val_eval = torch.utils.data.DataLoader(dataset, batch_size=len(val_indices),
                                                sampler=valid_sampler)

    test_dataset = AirBNBDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(y_test))

    loaders =  {'train_loader': train_loader, 'validation_loader': validation_loader, 
            'train_eval': train_eval, 'val_eval': val_eval, 'test_eval': test_loader}
    return SimpleNamespace(**loaders)


def define_loss_criteria():
    return torch.nn.MSELoss()

def get_nn_config(filename='nn_config_0.yaml'):
    # Load configuration from yaml file to python dict
    with open('nn_configs/' + filename, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_configs(arguments):
    if arguments['nn_single_config'] is None:
        configs = []
        for iconfig in range(arguments['nn_n_configs']):
            configs.append(get_nn_config('nn_config_' + str(iconfig) + '.yaml'))
        return configs
    else:
        return [get_nn_config(arguments['nn_single_config'])]

def get_loss_metric(model, loader):
    for X, y in loader:
        y_pred = model(X)
        loss_metric = np.sqrt(np.mean((y_pred.detach().numpy() - y.detach().numpy())**2))
        #r2 = r2_score(y.detach().numpy(), y_pred.detach().numpy())
    return float(loss_metric)#, float(r2)

def get_r2(model, loader):
    for X, y in loader:
        y_pred = model(X)
        #loss_metric = np.sqrt(np.mean((y_pred.detach().numpy() - y.detach().numpy())**2))
        r2 = r2_score(y.detach().numpy(), y_pred.detach().numpy())
    return float(r2)

def train_model(arguments, loaders, configs, loss_criteria):  

    epochs = arguments['nn_epochs']
    criterion = define_loss_criteria()
    trained = []

    
    for iconfig, config in enumerate(configs):

        start = time.time() 

        model = NN(config)
        if config['optimiser'] == 'Adam':
            optimiser = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        elif config['optimiser'] == 'SGD':
            optimiser = torch.optim.SGD(model.parameters(), lr=config['learning_rate'])

        train_losses, val_losses, test_losses = [], [], []

        for epoch in range(epochs):
            for X, y in loaders.train_loader:
                y_pred = model(X)
                loss = criterion(y_pred, y)
                loss.backward()
                optimiser.step()
                optimiser.zero_grad()

            train_losses.append(get_loss_metric(model, loaders.train_eval))
            val_losses.append(get_loss_metric(model, loaders.val_eval))

        try:
            trained_rmse = get_loss_metric(model, loaders.train_eval)
            trained_r2 = get_r2(model, loaders.train_eval)
            val_rmse = get_loss_metric(model, loaders.val_eval)
            val_r2 = get_r2(model, loaders.val_eval)
            test_rmse = get_loss_metric(model, loaders.test_eval)
            test_r2 = get_r2(model, loaders.test_eval)
        except:
            trained_rmse, trained_r2, val_rmse, val_r2, test_rmse, test_r2 = None, None, None, None, None, None

        train_time = time.time() - start

        trained.append({'config': config, 'model': model, 
                        'train_losses': train_losses, 'val_losses': val_losses, 
                        'train_r2': trained_r2, 'val_r2': val_r2, 'test_r2': test_r2,
                        'train_rmse': trained_rmse, 'val_rmse': val_rmse, 'test_rmse': test_rmse,
                        'train_time': train_time})
        
    return trained

def save_trained_models(trained, results_dir):
    for iconfig, one_config in enumerate(trained):
        savedir = results_dir + str(iconfig) + '/'
        os.system('mkdir -p ' + savedir)
        torch.save(one_config['model'], savedir + 'model.pt')
        with open(savedir + 'metrics.json', 'w') as file:
            metrics = {'loss_curves': {'train': one_config['train_losses'], 'val': one_config['val_losses']},
                        'r2': {'train': one_config['train_r2'], 'val': one_config['val_r2'], 'test': one_config['test_r2']},
                        'rmse': {'train': one_config['train_rmse'], 'val': one_config['val_rmse'], 'test': one_config['test_rmse']},
                        'train_time': one_config['train_time']}
            json.dump(metrics, file)
        os.system('cp nn_configs/nn_config_' + str(iconfig) + '.yaml ' + savedir)
        

class AirBNBDataset(torch.utils.data.Dataset):
    '''
    Custom dataset for AirBNB data, inheriting from torch.utils.data.Dataset.

    Arguments:
    Xs: pandas DataFrame, features
    ys: pandas Series, labels
    '''
    def __init__(self, Xs, ys):
        super().__init__() # Inherit from torch.utils.data.Dataset
        self.X, self.y = Xs, ys # Assign features and labels to self.X and self.y

    def __getitem__(self, index):
        # Define behaviour of indexing
        return (torch.tensor(self.X.iloc[index], dtype=torch.float32), 
                torch.tensor(self.y.iloc[index], dtype=torch.float32).reshape(-1))
    
    def __len__(self):
        # Define behaviour of len()
        return len(self.y)

class NN(torch.nn.Module):
    '''
    Neural network class, inheriting from torch.nn.Module.
    
    Arguments:
    config: int, configuration index to pick the yaml config file'''
    def __init__(self, config):
        super().__init__()

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
    

def train_nn(arguments):

    results_dir = make_results_dir(arguments)

    X_train, X_test, y_train, y_test = make_datasets(arguments)

    save_data_to_results(X_train, X_test, y_train, y_test, results_dir)

    loaders = make_dataloaders(arguments, X_train, X_test, y_train, y_test)

    configs = get_configs(arguments)

    loss_criteria = define_loss_criteria()

    trained = train_model(arguments, loaders=loaders, configs=configs, loss_criteria=loss_criteria)

    save_trained_models(trained, results_dir)

    return results_dir, trained

def evaluate_nn(config, results_dir, plot_scatter=False, plot_loss=False):
    # Load in the trained models and evaluate them
    data = load_data_from_results(results_dir)
    savedir = results_dir + config + '/'
    model = torch.load(savedir + 'model.pt')
    with open(savedir + 'metrics.json', 'r') as file:
        metrics = json.load(file)
    rmse = metrics['rmse']['test']
    if plot_scatter:
        # Plot the scatter plot of the test performance
        fig = make_scatter_plot(data.X_test, data.y_test, model)
        fig.savefig(savedir + 'test_scatter.png')
    if plot_loss:
        # Plot the train and val loss
        fig = make_loss_curve_plot(metrics['loss_curves']['train'], metrics['loss_curves']['val'])
        fig.savefig(savedir + 'loss_curves.png')
    return rmse

def make_scatter_plot(X, y, model):
    # Plot the scatter plot of the test performance
    y_pred = model(torch.tensor(X.values, dtype=torch.float32))
    y_pred = y_pred.detach().numpy()
    fig = plt.figure(dpi=300, figsize=(4, 4))
    plt.scatter(y, y_pred, c='k')
    plt.plot([0, 1000], [0, 1000], 'k--')
    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.xlim([0, 1000])
    plt.ylim([0, 1000])
    plt.grid()
    return fig

def make_loss_curve_plot(train_loss, val_loss):
    # Plot the train and val loss
    fig = plt.figure(dpi=300, figsize=(4, 4))
    plt.plot(train_loss, 'k', label='train')
    plt.plot(val_loss, 'k:', label='val')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid()
    return fig

def find_best_model(results_dir, configs):
    # Find the best model in the results directory
    best_rmse = np.inf
    for config in configs:
        config = str(config)
        rmse = evaluate_nn(config, results_dir)
        try:
            if rmse < best_rmse:
                best_rmse = rmse
                best_config = config
        except:
            pass
    return best_rmse, best_config
        
def train_and_evaluate_nn(arguments):
    results_dir, trained = train_nn(arguments)
    best_rmse, best_config = find_best_model(results_dir, range(arguments['nn_n_configs']))
    evaluate_nn(best_config, results_dir, plot_scatter=True, plot_loss=True)
    print('Best RMSE:', best_rmse)
    print('Best config:', best_config)
    print('Best results at:', results_dir + best_config + '/')
    return best_rmse, best_config

if __name__ == '__main__':

    arguments = make_args()

    train_and_evaluate_nn(arguments.copy())

#%%
#arguments = parse_args()

##results_dir, trained = train_nn(arguments.copy())

#best_rmse, best_config = find_best_model(results_dir, range(arguments['nn_n_configs']))

#evaluate_nn(best_config, results_dir, plot_scatter=True, plot_loss=True)
# make results directory
# get data
# save data to results directory
# make datasets
# make evaluation datasets
# define loss criteria
# train models and get series of train and val loss
    # return config, model, train loss, val loss, test loss

# evaluate models
    # load in each model
    # plot the train and val loss
    # plot the scatter plot of the test performance


# %%
