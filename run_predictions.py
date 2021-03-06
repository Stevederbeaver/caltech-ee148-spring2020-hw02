import os
import numpy as np
import json
from PIL import Image

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

def compute_convolution(I, T1, T2, stride=1):
    '''
    This function takes an image <I> and a template <T> (both numpy arrays)
    and returns a heatmap where each grid represents the output produced by
    convolution at each location. You can add optional parameters (e.g. stride,
    window_size, padding) to create additional functionality.
    '''
    I = I.astype(int)
    T1 = T1.astype(int)
    T2 = T2.astype(int)
    (n_rows, n_cols, n_channels) = np.shape(I)
    (n_rows_T, n_cols_T, n_channels_T) = np.shape(T1)
    heatmap = np.zeros((n_rows - n_rows_T + 1, n_cols - n_cols_T + 1))
    '''
    BEGIN YOUR CODE
    '''
    T1_transform = normalize(T1)
    T2_transform = normalize(T2)

    for i in range(0, n_rows - n_rows_T + 1, stride):
        for j in range(0, n_cols - n_cols_T + 1, stride):
            I_transform = normalize(I[i:i + n_rows_T, j:j + n_cols_T, :])
            value1 = dot_product(I_transform, T1_transform)
            value2 = dot_product(I_transform, T2_transform)
            heatmap[i][j] = 0.5 * (value1 + value2)
    '''
    END YOUR CODE
    '''

    return heatmap


def predict_boxes(heatmap, T):
    '''
    This function takes heatmap and returns the bounding boxes and associated
    confidence scores.
    '''

    output = []
    thre = 0.87
    (n_rows_T, n_cols_T, n_channels_T) = np.shape(T)
    (n_rows_heatmap,n_cols_heatmap) = np.shape(heatmap)

    for i in range(n_rows_heatmap):
        for j in range(n_cols_heatmap):
            if heatmap[i][j] > thre:
                output.append([i, j, i + n_rows_T, j + n_cols_T, heatmap[i][j]])

    '''
    BEGIN YOUR CODE
    '''

    '''
    As an example, here's code that generates between 1 and 5 random boxes
    of fixed size and returns the results in the proper format.
    '''


    '''

    box_height = 8
    box_width = 6

    num_boxes = np.random.randint(1,5)

    for i in range(num_boxes):
        (n_rows,n_cols,n_channels) = np.shape(I)

        tl_row = np.random.randint(n_rows - box_height)
        tl_col = np.random.randint(n_cols - box_width)
        br_row = tl_row + box_height
        br_col = tl_col + box_width

        score = np.random.random()

        output.append([tl_row,tl_col,br_row,br_col, score])

    END YOUR CODE
    '''

    return output


def detect_red_light_mf(I, T1, T2):
    '''
    This function takes a numpy array <I> and returns a list <output>.
    The length of <output> is the number of bounding boxes predicted for <I>.
    Each entry of <output> is a list <[row_TL,col_TL,row_BR,col_BR,score]>.
    The first four entries are four integers specifying a bounding box
    (the row and column index of the top left corner and the row and column
    index of the bottom right corner).
    <score> is a confidence score ranging from 0 to 1.

    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''

    '''
    BEGIN YOUR CODE
    '''
    heatmap = compute_convolution(I, T1, T2, stride = 1)
    output = predict_boxes(heatmap, T1)

    '''
    template_height = 8
    template_width = 6

    # You may use multiple stages and combine the results
    T = np.random.random((template_height, template_width))

    heatmap = compute_convolution(I, T)
    output = predict_boxes(heatmap)

    END YOUR CODE
    '''

    for i in range(len(output)):
        assert len(output[i]) == 5
        assert (output[i][4] >= 0.0) and (output[i][4] <= 1.0)

    return output

# Note that you are not allowed to use test data for training.
# set the path to the downloaded data:
data_path = '../data/RedLights2011_Medium'

# load splits:
split_path = '../data/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# set a path for saving predictions:
preds_path = '../data/hw02_preds'
'''
os.makedirs(preds_path, exist_ok=True) # create directory if needed
'''

# Set this parameter to True when you're done with algorithm development:
done_tweaking = True

training_unfinished = True

'''
Make predictions on the training set.
'''
I_RL_116 = np.asarray(Image.open(os.path.join(data_path,file_names_train[99])))
T1 = I_RL_116[160:180, 357:367, :]

I_RL_118 = np.asarray(Image.open(os.path.join(data_path,file_names_train[101])))
T2 = I_RL_118[150:170, 337:347, :]
preds_train = {}

if training_unfinished:
    for i in range(len(file_names_train)):
        print(i)

        # read image using PIL:
        I = Image.open(os.path.join(data_path,file_names_train[i]))

        # convert to numpy array:
        I = np.asarray(I)

        preds_train[file_names_train[i]] = detect_red_light_mf(I, T1, T2)

    # save preds (overwrites any previous predictions!)
    with open(os.path.join(preds_path,'preds_train.json'),'w') as f:
        json.dump(preds_train,f)

if done_tweaking:
    '''
    Make predictions on the test set.
    '''
    preds_test = {}
    for i in range(len(file_names_test)):
        print(i)

        # read image using PIL:
        I = Image.open(os.path.join(data_path,file_names_test[i]))

        # convert to numpy array:
        I = np.asarray(I)

        preds_test[file_names_test[i]] = detect_red_light_mf(I, T1, T2)

    # save preds (overwrites any previous predictions!)
    with open(os.path.join(preds_path,'preds_test.json'),'w') as f:
        json.dump(preds_test,f)
