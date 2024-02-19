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
            print(predictions[0:3])
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
    with open(filename, 'r') as file:
        config = yaml.safe_load(file)
    return config

#%%
    
X_train, X_val, X_test, y_train, y_val, y_test = data_for_nn()
    
train_data = AirBNBDataset(X_train, y_train)
val_data = AirBNBDataset(X_val, y_val)

train_loader = DataLoader(train_data, batch_size=216, shuffle=True)
val_loader = DataLoader(val_data, batch_size=216)


#example = next(iter(train_loader))
#print(example)
#features, labels = example
#features = features.type(torch.float32)
##features = features.reshape(batch_size, -1)
#%%
config = get_nn_config('nn_config_0.yaml')
model = NN()

#model = LinearRegression()
#model(features)
test_data = AirBNBDataset(X_test, y_test)
test_loader = DataLoader(test_data, batch_size=len(test_data))
for batch in test_loader:
        features, labels = batch
        predictions = model(features)
        loss = F.mse_loss(predictions, labels)
        print(np.sqrt(loss.item()))
#%%

tl, vl = train(model, config, epochs=50)

#%%
plt.plot(tl, label='train')
plt.plot(vl, label='val')
plt.legend()
plt.ylim([0, 300])

#%%
test_data = AirBNBDataset(X_test, y_test)
test_loader = DataLoader(test_data, batch_size=len(test_data))
for batch in test_loader:
        features, labels = batch
        predictions = model(features)
        loss = F.mse_loss(predictions, labels)
        print(np.sqrt(loss.item()))

tt = torch.tensor(X_test.values, dtype=torch.float32)
pred = model(tt)
pred = pred.detach().numpy()
plt.scatter(y_test,pred)

# %%
torch.save(model.state_dict(), 'test_model.pt')

# %%
state_dict = torch.load('test_model.pt')
new_model = NN()
new_model.load_state_dict(state_dict)
train(new_model, epochs=10)

# %%