import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

split_path = '../data/hw02_splits'
data_path = '../data/RedLights2011_Medium'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

def dot_product(A, B):
    '''
    This function takes two matrices A, B of the same size and returns their
    dot product (sum of the entries in their Hadamard product)
    '''
    return sum(sum(sum(np.multiply(A, B))))

def normalize(A):
    '''
    This function takes an arbitrary matrix A as the input and normalize it
    '''
    A1 = A / 255.0
    coeff = np.sqrt(dot_product(A1, A1))
    B = A1 / coeff
    return B

def compute_convolution_single(I, T, stride=1):
    '''
    This function takes an image <I> and a template <T> (both numpy arrays)
    and returns a heatmap where each grid represents the output produced by
    convolution at each location. You can add optional parameters (e.g. stride,
    window_size, padding) to create additional functionality.
    '''
    I = I.astype(int)
    T = T.astype(int)
    (n_rows, n_cols, n_channels) = np.shape(I)
    (n_rows_T, n_cols_T, n_channels_T) = np.shape(T)
    heatmap = np.zeros((n_rows - n_rows_T + 1, n_cols - n_cols_T + 1))
    '''
    BEGIN YOUR CODE
    '''
    T_transform = normalize(T)

    for i in range(0, n_rows - n_rows_T + 1, stride):
        for j in range(0, n_cols - n_cols_T + 1, stride):
            I_transform = normalize(I[i:i + n_rows_T, j:j + n_cols_T, :])
            value = dot_product(I_transform, T_transform)
            heatmap[i][j] = value
    '''
    END YOUR CODE
    '''

    return heatmap

I_RL_118 = np.asarray(Image.open(os.path.join(data_path,file_names_train[101])))
T2 = I_RL_118[150:170, 337:347, :]
heatmap2 = compute_convolution_single(I_RL_118, T2, stride=1)
heatplt2 = sns.heatmap(heatmap2, cmap="YlGnBu")
heatmapoutput2 = heatplt2.get_figure()
heatmapoutput2.savefig('heatmapoutput2.png')

I_RL_116 = np.asarray(Image.open(os.path.join(data_path,file_names_train[99])))
T1 = I_RL_116[160:180, 357:367, :]
heatmap1 = compute_convolution_single(I_RL_116, T1, stride=1)
heatplt1 = sns.heatmap(heatmap1, cmap="YlGnBu")
heatmapoutput1 = heatplt1.get_figure()
heatmapoutput1.savefig('heatmapoutput1.png')
