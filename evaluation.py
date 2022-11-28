import numpy as np
from scipy.optimize import linear_sum_assignment
import random
import matplotlib.pyplot as plt


def MCC(true_z, predicted_z):
    """Caluclates the Correlation Coefficient between all pairs of true 
    and recovered latent variables for one environment 

    Uses Pearsons Corr Coef

    from paper: 
    We also compute the mean correlation coefficient (MCC) used in Khemakhem et al. (2020a), which
    can be obtained by calculating the correlation coefficient between all pairs of true and recovered
    latent factors and then solving a linear sum assignment problem by assigning each recovered latent
    factor to the true latent factor with which it best correlates

    Args:
        true_z (numpy array): 2D dimensional numpy array, where columns represent variables
        predicted_z (numpy array): _description_
    """
    num_true = len(true_z[0])
    num_predicted = len(predicted_z[0])
    corr_matrix = np.corrcoef(true_z, predicted_z, rowvar=False)
    reduced_matrix = corr_matrix[
        0:num_true, num_true : len(corr_matrix[0]) + 1
    ]  # where rows are true and columns are predicted
    row_ind, col_ind = linear_sum_assignment(reduced_matrix)

    mcc = [reduced_matrix[row_ind[i], col_ind[i]] for i in range(len(row_ind))]
    print(mcc)
    mcc = np.sum(mcc) / (num_predicted + num_true)
    return mcc


def plot_MCC(mcc_model, mcc_mean, mcc_var):
    """_summary_

    Args:
        mcc_model (list): names of models that MCC was performed on
        mcc_mean (list): the returned value of the MCC function
        mcc_var (list): the variance that corresponds with the mean values given above
    """
    plt.bar(mcc_model, mcc_mean, yerr=mcc_var)


# test MCC
random.seed(10)
true_z = np.random.rand(4, 3)
test_z = np.random.rand(4, 3)
# print(f"true z: {true_z}")
print(MCC(true_z, test_z))
