import numpy as np
from scipy.optimize import linear_sum_assignment
import random


def MCC(true_z, predicted_z):
    """Caluclates the Correlation Coefficient between all pairs of true 
    and recovered latent variables for one environment 

    Uses Pearsons Corr Coef

    Args:
        true_z (numpy array): 2D dimensional numpy array, where columns represent variables
        predicted_z (numpy array): _description_
    """
    # corr_matrix = np.zeros(len(true_z), len(predicted_z))
    num_true = len(true_z[0])
    num_predicted = len(predicted_z[0])
    corr_matrix = np.corrcoef(
        true_z, predicted_z, rowvar=False
    )  # [[np.corrcoef(z1, z2) for z1 in true_z] for z2 in predicted_z]
    # print(corr_matrix)
    # print("indexed")
    # print(f"num predicted {num_predicted}")
    # print(corr_matrix[0:3, -num_predicted:-1])
    # print(corr_matrix[0:num_true, -num_predicted:6])
    # print(f"indices: 0:{num_true}, {num_true+1}")
    # print(corr_matrix[0:num_true, num_true : len(corr_matrix[0]) + 1])
    # print(corr_matrix[:, -1])
    reduced_matrix = corr_matrix[
        0:num_true, num_true : len(corr_matrix[0]) + 1
    ]  # where rows are true and columns are predicted
    row_ind, col_ind = linear_sum_assignment(reduced_matrix)

    mcc = [reduced_matrix[row_ind[i], col_ind[i]] for i in range(len(row_ind))]
    print(mcc)
    mcc = np.sum(mcc) / (num_predicted + num_true)
    return mcc


# test
random.seed(10)
true_z = np.random.rand(4, 3)
test_z = np.random.rand(4, 3)
# print(f"true z: {true_z}")
print(MCC(true_z, test_z))
