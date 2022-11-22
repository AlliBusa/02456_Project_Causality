# Imports

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

# packages needed for creating x
import torch
from torch import nn, Tensor
import torch.nn.functional as F

# import torch.optim as optim
import torch.nn.init as init
from torch.nn.parameter import Parameter

# from torch.nn.functional import softplus
from torch.distributions import Distribution, Normal

from typing import *

# data imports
from torch.utils.data import Dataset, DataLoader

# from torchvision.transforms import ToTensor
# from functools import reduce


def p(x):
    """Converts input array to pandas dataframe"""
    return pd.DataFrame(x)


##################################
# Generate Zs and Ys


def generate_z_and_y(E):
    """
    Takes in a list of means representing different environments and
    generates latent variables (z's) and y for that environment

    Currently only works for 4 environments and 4 output variables

    Args:
        E (list): A list of four numbers representing four means of four different environments

    Returns:
        _envs (nested dictionary): A nested dictionary containing all environments
        and latent variables for each 
            ex: envs.keys() = [0,1] -> contains data for environment 0 and 1
                envs[0].keys() = ["Y","Zs"] -> Each environment contains a dictionary 
                with a numpy array of Y values and a numpy array for Z values 
                The Z values are organized column wise (ie the first column contains the first latent variable)
    """

    beta_z1 = 1  # np.random.normal(0,1)
    beta_z2 = 1  # np.random.normal(0,1)
    beta_z3 = 1  # np.random.normal(0,1)

    environments = {}

    for i in range(len(E)):
        # create nested dictionary
        single_env = {}

        # Draw beta
        beta1 = 1  # np.random.normal(0,1)
        beta2 = 1  # np.random.normal(0,1)

        # Create Z1
        Z1 = np.random.normal(beta1 * E[i], 1, 1000)
        # Z1=np.vstack((Z1, np.array([S[i]]*1000))).reshape(-1,1000).T

        # Create Z2
        Z2 = np.random.normal(2 * beta2 * E[i], 2, 1000)
        # Z2=np.vstack((Z2, np.array([S[i]]*1000))).reshape(-1,1000).T

        # Create Y
        Y = np.zeros(1000)

        # Create Z3
        Z3 = np.zeros(1000)

        for j in range(len(Z1)):
            Y[j] = np.random.normal(beta_z1 * Z1[j] + beta_z2 * Z2[j], 1)

            Z3[j] = np.random.normal(3 * beta_z3 * Y[j], 1)
        single_env["Y"] = Y
        single_env["Zs"] = np.column_stack([Z1, Z2, Z3])

        environments[
            i
        ] = single_env  # put nested dictionary into environment dictionary

    return environments


#####################################
# Generate X's

# define network
class Net(nn.Module):
    def __init__(self, num_features, num_hidden, num_output):
        super(Net, self).__init__()
        # hidden layer
        self.W_1 = Parameter(
            init.xavier_normal_(torch.Tensor(num_hidden, num_features))
        )
        self.b_1 = Parameter(init.constant_(torch.Tensor(num_hidden), 0))
        # output layer
        self.W_2 = Parameter(
            init.xavier_normal_(torch.Tensor(num_output, num_hidden))
        )
        self.b_2 = Parameter(init.constant_(torch.Tensor(num_output), 0))
        # define activation function in constructor
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        x = F.linear(x, self.W_1, self.b_1)
        x = self.activation(x)
        x = F.linear(x, self.W_2, self.b_2)
        return x


def generate_x_from_z(env, net):
    """ Runs neural network on latent variables to return X"""

    return net(torch.from_numpy(env["Zs"].astype("float32")))


def generate_all_x(envs, net):
    """ Iterates through an array of latent variables for different environments 
    Returns a dictionary which maps environments to X's
    """
    X = {}
    for i in envs.keys():  # iterate through all environments
        X[i] = generate_x_from_z(envs[i], net)
    return X


########################################
# Create Pytorch Dataset


class CustomSyntheticDataset(Dataset):
    def __init__(self, X, env):
        self.X = X
        self.Zs = env["Zs"]
        self.Y = env["Y"]

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        # get label from Y
        label = self.Y[idx, :]
        return self.X, self.Zs, label

    def return_training(self):
        return self.X

    def return_testing(self):
        return self.Y


#########################################
# Load Pytorch Dataset


def load_data(dataset_instance, batch_size=64, eval_batch_size=100):

    dset_train = dataset_instance.return_training()
    dset_test = dataset_instance.return_testing()
    # The loaders perform the actual work
    train_loader = DataLoader(dset_train[:910, :], batch_size=batch_size)
    test_loader = DataLoader(dset_train[910:, :], batch_size=eval_batch_size)

    return train_loader, test_loader


########################################
# Wrapper Function for Creating Synthetic Dataset

# Hyperparameters
num_classes = 10
num_l1 = 6
num_features = 3

# net = Net(num_features, num_l1, num_classes)


def generate_3Z_synthetic_data(E):
    # Hyperparameters
    num_classes = 10
    num_l1 = 6
    num_features = 3
    envs = generate_z_and_y(E)
    net = Net(num_features, num_l1, num_classes)
    # generate X for all environments
    Xs = generate_all_x(envs, net)
    # Create Pytorch Dataset , for environment 1
    pytorch_dataset = CustomSyntheticDataset(Xs[0], envs[0])
    # Put Pytorch Dataset in Data Loader
    train_loader, test_loader = load_data(pytorch_dataset)
    return envs, Xs, train_loader, test_loader


E = [0.2, 2, 3, 5]  # environmental factors
envs, Xs, train_loader, test_loader = generate_3Z_synthetic_data(E)

###############################################

###############################################

# Implement Gaussian Distribution


class ReparameterizedDiagonalGaussian(Distribution):
    """
    A distribution `N(y | mu, sigma I)` compatible with the reparameterization trick given `epsilon ~ N(0, 1)`.
    """

    def __init__(self, mu: Tensor, log_sigma: Tensor):
        assert (
            mu.shape == log_sigma.shape
        ), f"Tensors `mu` : {mu.shape} and ` log_sigma` : {log_sigma.shape} must be of the same shape"
        self.mu = mu
        self.sigma = log_sigma.exp()

    def sample_epsilon(self) -> Tensor:
        """`\eps ~ N(0, I)`"""
        return torch.empty_like(self.mu).normal_()

    def sample(self) -> Tensor:
        """sample `z ~ N(z | mu, sigma)` (without gradients)"""
        with torch.no_grad():
            return self.rsample()

    def rsample(self) -> Tensor:
        """sample `z ~ N(z | mu, sigma)` (with the reparameterization trick) """
        return self.sample_epsilon() * self.sigma + self.mu

    def log_prob(self, z: Tensor) -> Tensor:
        """return the log probability: log `p(z)`"""
        # create a normal distribution to sample from
        m = Normal(self.mu, self.sigma)
        # get probability of choosing z from that distribution
        return m.log_prob(z)


###############################################

###############################################

# Plotting
def plot_latent_2d(Z1, Z2, Env):
    plot_df = pd.concat([Z1, Z2, Env], axis=1)
    plot_df.columns = ["x", "y", "Env"]
    sns.scatterplot(data=plot_df, x="x", y="y", hue="Env")


def plot_y_dist(Y):
    hue = pd.DataFrame(Y.stack()).index.get_level_values(1)
    xindex = pd.DataFrame(Y.stack()).index.get_level_values(0)
    stacked = pd.DataFrame(Y.stack())
    stacked["x"] = xindex
    stacked["Env"] = hue
    sns.kdeplot(data=stacked, x=0, hue="Env")
    plt.show()
