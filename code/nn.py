#%%
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.datasets import load_diabetes
from torchvision.datasets import MNIST
from torchvision.transforms import PILToTensor
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import yaml
import random
import os

from tabular_data import ModelData

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

#%%
def data_for_nn(filename='listing.csv', filepath='../data/intermediate/airbnb-property-listings/tabular_data/', column='Price_Night', random_state=1):
    loader = ModelData()
    df = loader.load_tabular_data(filename, filepath)
    X, y = loader.extract_label(df, column)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.6, random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test

class AirBNBDataset(Dataset):
    def __init__(self, Xs, ys):
        super().__init__()
        self.X, self.y = Xs, ys

    def load_tabular_data(self, filename, filepath):
        loader = ModelData()
        df = loader.load_tabular_data(filename, filepath)
        return df
    
    def extract_label(self, df, column, numeric_only=True):
        loader = ModelData()
        features, label = loader.extract_label(df, column, numeric_only)
        return features, label

    def __getitem__(self, index):
        return (torch.tensor(self.X.iloc[index], dtype=torch.float32), 
                torch.tensor(self.y.iloc[index], dtype=torch.float32).reshape(-1))
    
    def __len__(self):
        return len(self.y)
    

def train(model, config, epochs=10):

    if config['optimiser'] == 'SGD':
        optimiser = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=config['momentum'])
    elif config['optimiser'] == 'Adam':
        optimiser = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    writer = SummaryWriter()

    val_features, val_labels = next(iter(val_loader)) # val data in one batch, grab all

    batch_idx = 0
    tl = []
    vl = []
    for epoch in range(epochs):

        train_loss = []
        val_loss = []

        for batch in val_loader:
            features, labels = batch
            predictions = model(features)
            loss = F.mse_loss(predictions, labels)
            #writer.add_scalar('Val loss', val_loss.item(), batch_idx)
            val_loss.append(np.sqrt(loss.item()))
        val_loss = np.mean(val_loss)

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
        train_loss = np.mean(train_loss)

        


        tl.append(train_loss )
        vl.append(val_loss)
    return tl, vl
        # then, with best model, loop over validation batches
        

       
        #writer.add_scalars(f'loss/check_info', {
        #    'train': train_loss / len(train_loader),
        #    'val': val_loss / len(val_loader),
        #}, epoch)
       # print(f'Epoch {epoch} train loss: {train_loss / len(train_loader)}, val loss: {val_loss / len(val_loader)}')
        


class NN(torch.nn.Module):
    def __init__(self, config=0):
        super().__init__()

        config = get_nn_config('nn_config_' + str(config) + '.yaml')

        self.n_features = config['n_features']
        self.width = config['hidden_layer_width']
        self.depth = config['depth']

        self.define_architecture()
        #self.layers = torch.nn.Sequential(
         #               torch.nn.Linear(self.n_features, self.width),
         #               torch.nn.ReLU(),
         #               torch.nn.Linear(self.n_features, 1)
        #                              )   

    def define_architecture(self):
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
        return self.layers(X)


def get_nn_config(filename='nn_config_0.yaml'):
    with open('nn_configs/' + filename, 'r') as file:
        config = yaml.safe_load(file)
    return config

def generate_nn_config_range(filename='nn_configs/config_ranges.yaml'):
    with open(filename, 'r') as file:
        config_ranges = yaml.safe_load(file)
    return config_ranges
    
     

def generate_nn_configs(ranges='./nn_configs/config_ranges.yaml', export_path='./nn_configs/'):
    config_ranges = generate_nn_config_range(ranges)
    for ix in range(16):
        config = {}
        for key in config_ranges.keys():
            try:
                config[key] = random.choice(config_ranges[key])
            except:
                config[key] = config_ranges[key]
        with open(export_path + 'nn_config_' + str(ix) + '.yaml', 'w') as file:
            yaml.dump(config, file)
    

def plot_training_curve(train_loss, val_loss, title, show=False):
    fig = plt.figure()
    plt.plot(train_loss, label='train')
    plt.plot(val_loss, label='val')
    plt.legend()
    plot_title = title + ' training curve'
    plt.title(plot_title)
    if show:
        plt.show()
    return fig

def plot_validation_scatter(model, X_test, y_test, config_ext, test_loss, show=False):
    fig = plt.figure()
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

def get_test_loss(model, X_test, y_test):
    test_data = AirBNBDataset(X_test, y_test)
    test_loader = DataLoader(test_data, batch_size=len(test_data))
    test_loss = []
    for batch in test_loader:
            features, labels = batch
            predictions = model(features)
            loss = F.mse_loss(predictions, labels)
            test_loss.append(np.sqrt(loss.item()))
    return np.mean(test_loss)

#%%
    
n_configs = 4
epochs = 50

X_train, X_val, X_test, y_train, y_val, y_test = data_for_nn()
    
train_data = AirBNBDataset(X_train, y_train)
val_data = AirBNBDataset(X_val, y_val)
test_data = AirBNBDataset(X_test, y_test)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64)
test_loader = DataLoader(test_data, batch_size=64)


configs = generate_nn_configs()
for iconfig in range(n_configs):

    print('Training model ' + str(1 + iconfig) + ' of ' + str(n_configs))
    
    config_ext = str(iconfig)
    
    config = get_nn_config('nn_config_' + config_ext + '.yaml')

    model = NN()

    test_loss_original = int(np.round(get_test_loss(model, X_test, y_test), 0))

    print('Original test loss: ' + str(test_loss_original))

    tl, vl = train(model, config, epochs=epochs)

    test_loss_trained = int(round(get_test_loss(model, X_test, y_test), 0))
    print('Trained test loss: ' + str(test_loss_trained))

    f_train = plot_training_curve(tl, vl, config_ext)
    f_val = plot_validation_scatter(model, X_test, y_test, config_ext, test_loss_trained)

    # %%
    results_dir = '../models/nn/' + config_ext + '/'
    os.system('mkdir -p ' + results_dir)
    torch.save(model.state_dict(), results_dir + 'nn.pt')
    save_str = 'cp nn_configs/nn_config_'+config_ext+'.yaml ' + results_dir
    os.system(save_str)
    f_train.savefig(results_dir + 'training_curve.png')
    f_val.savefig(results_dir + 'validation_scatter.png')

    print('================ Done ================')

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