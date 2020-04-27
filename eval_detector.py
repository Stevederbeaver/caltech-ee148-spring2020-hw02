import os
import json
import matplotlib.pyplot as plt
import numpy as np

def compute_iou(box_1, box_2):
    '''
    This function takes a pair of bounding boxes and returns intersection-over-
    union (IoU) of two bounding boxes.
    '''
    x_overlap =  max(min(box_1[2], box_2[2]) - max(box_1[0], box_2[0]), 0)
    y_overlap =  max(min(box_1[3], box_2[3]) - max(box_1[1], box_2[1]), 0)

    area_overlap = x_overlap * y_overlap
    area_1 = (box_1[2]-box_1[0]) * (box_1[3]-box_1[1])
    area_2 = (box_2[2]-box_2[0]) * (box_2[3]-box_2[1])
    area_union = area_1 + area_2 - area_overlap

    iou = area_overlap / area_union

    assert (iou >= 0) and (iou <= 1.0)

    return iou

def compute_counts(preds, gts, iou_thr=0.5, conf_thr=0.5):
    '''
    This function takes a pair of dictionaries (with our JSON format; see ex.)
    corresponding to predicted and ground truth bounding boxes for a collection
    of images and returns the number of true positives, false positives, and
    false negatives.
    <preds> is a dictionary containing predicted bounding boxes and confidence
    scores for a collection of images.
    <gts> is a dictionary containing ground truth bounding boxes for a
    collection of images.
    '''
    TP = 0
    FP = 0
    FN = 0

    '''
    BEGIN YOUR CODE
    '''
    for pred_file, pred in preds.iteritems():
        gt = gts[pred_file]
        number_gt = len(gt)
        detection_map = [0 for i in range(number_gt)]
        number_pred = 0

        for j in range(len(pred)):
            if pred[j][4] >= conf_thr:
                number_pred += 1
                for k in range(number_gt):
                    iou = compute_iou(pred[j][:4], gt[k])
                    if iou >= iou_thr and detection_map[k] == 0:
                        detection_map[k] = 1
                        break

        number_detect = sum(detection_map)

        TP += number_detect
        FP += (number_pred - number_detect)
        FN += (number_gt - number_detect)

    '''
    END YOUR CODE
    '''

    return TP, FP, FN

# set a path for predictions and annotations:
preds_path = '../data/hw02_preds'
gts_path = '../data/hw02_annotations'

# load splits:
split_path = '../data/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# Set this parameter to True when you're done with algorithm development:
done_tweaking = True

training_unfinished = True

'''
Load training data.
'''
with open(os.path.join(preds_path,'preds_train.json'),'r') as f:
    preds_train = json.load(f)

with open(os.path.join(gts_path, 'annotations_train.json'),'r') as f:
    gts_train = json.load(f)

if done_tweaking:

    '''
    Load test data.
    '''

    with open(os.path.join(preds_path,'preds_test.json'),'r') as f:
        preds_test = json.load(f)

    with open(os.path.join(gts_path, 'annotations_test.json'),'r') as f:
        gts_test = json.load(f)

confidence_thrs = []
for pred_file in preds_train:
    pred_list = preds_train[pred_file]
    pred_length = len(pred_list)
    if pred_length > 0:
        confidence_thrs += [pred_list[i][4] for i in range(pred_length)]
confidence_thrs = np.sort(np.array(confidence_thrs,dtype=float)) # using (ascending) list of confidence scores as thresholds

# For a fixed IoU threshold, vary the confidence thresholds.
# The code below gives an example on the training set for one IoU threshold.
if training_unfinished:
    min_iou_thr = 0.25

    precision_train = np.ones((3, len(confidence_thrs)))
    recall_train = np.zeros((3, len(confidence_thrs)))
    for i in range(3):
        for j, conf_thr in enumerate(confidence_thrs):
            TP, FP, FN = compute_counts(preds_train, gts_train, iou_thr= float(i+1) * min_iou_thr, conf_thr=conf_thr)
            precision_train[i][j] = float(TP) / float(TP + FP)
            recall_train[i][j] = float(TP) / float(TP + FN)

    print('Code for plotting train set PR curves.')

    plt.plot(recall_train[0], precision_train[0])
    plt.plot(recall_train[1], precision_train[1])
    plt.plot(recall_train[2], precision_train[2])
    plt.legend(['iou_thr=0.25', 'iou_thr=0.50', 'iou_thr=0.75'], loc = 'best')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR Curves for Training Set')
    plt.savefig('PR_train.png')
    plt.close()

# Plot testing set PR curves

if done_tweaking:
    min_iou_thr = 0.25

    precision_test = np.ones((3, len(confidence_thrs)))
    recall_test = np.zeros((3, len(confidence_thrs)))
    for i in range(3):
        for j, conf_thr in enumerate(confidence_thrs):
            TP, FP, FN = compute_counts(preds_test, gts_test, iou_thr= float(i+1) * min_iou_thr, conf_thr=conf_thr)
            precision_test[i][j] = float(TP) / float(TP + FP)
            recall_test[i][j] = float(TP) / float(TP + FN)

    print('Code for plotting test set PR curves.')

    plt.plot(recall_test[0], precision_test[0])
    plt.plot(recall_test[1], precision_test[1])
    plt.plot(recall_test[2], precision_test[2])
    plt.legend(['iou_thr=0.25', 'iou_thr=0.50', 'iou_thr=0.75'], loc = 'best')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR Curves for Testing Set')
    plt.savefig('PR_test.png')
    plt.close()
