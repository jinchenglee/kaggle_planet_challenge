import numpy as np
from sklearn.metrics import fbeta_score
from keras import backend as K
import matplotlib.pyplot as plt


def fbeta(y_true, y_pred, threshold_shift=0):
    beta = 1

    # just in case of hipster activation at the final layer
    y_pred = K.clip(y_pred, 0, 1)

    # shifting the prediction threshold from .5 if needed
    y_pred_bin = K.round(y_pred + threshold_shift)

    tp = K.sum(K.round(y_true * y_pred_bin), axis=1) + K.epsilon()
    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)), axis=1)
    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)), axis=1)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    beta_squared = beta ** 2
    return K.mean((beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall + K.epsilon()))

def precision(y_true, y_pred, threshold_shift=0):

    # just in case of hipster activation at the final layer
    y_pred = K.clip(y_pred, 0, 1)

    # shifting the prediction threshold from .5 if needed
    y_pred_bin = K.round(y_pred + threshold_shift)

    tp = K.sum(K.round(y_true * y_pred_bin), axis=1) + K.epsilon()
    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)), axis=1)
    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)), axis=1)

    precision = tp / (tp + fp)
    return precision

def recall(y_true, y_pred, threshold_shift=0):

    # just in case of hipster activation at the final layer
    y_pred = K.clip(y_pred, 0, 1)

    # shifting the prediction threshold from .5 if needed
    y_pred_bin = K.round(y_pred + threshold_shift)

    tp = K.sum(K.round(y_true * y_pred_bin), axis=1) + K.epsilon()
    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)), axis=1)
    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)), axis=1)

    recall = tp / (tp + fn)
    return recall

# Precision/Recall/Fbeta
#  - Implemented as "tensor"
def myfbeta(y_true, y_pred, beta = 2, threshold_shift=0):
        # just in case of hipster activation at the final layer
    y_pred = K.clip(y_pred, 0, 1)

    # shifting the prediction threshold from .5 if needed
    y_pred_bin = K.round(y_pred + threshold_shift)

    tp = K.sum(K.round(y_true * y_pred_bin)) + K.epsilon()
    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    beta_squared = beta ** 2
    fbeta_val = (beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall + K.epsilon())
    return fbeta_val, precision, recall

def mlc_confusion_matrix(yt, yp, classes):
    instcount = yt.shape[0]
    n_classes = classes.shape[0]
    mtx = np.zeros((n_classes, 4))
    for i in range(instcount):
        for c in range(n_classes):
            mtx[c,0] += 1 if yt[i,c]==1 and yp[i,c]==1 else 0
            mtx[c,1] += 1 if yt[i,c]==1 and yp[i,c]==0 else 0
            mtx[c,2] += 1 if yt[i,c]==0 and yp[i,c]==0 else 0
            mtx[c,3] += 1 if yt[i,c]==0 and yp[i,c]==1 else 0
    mtx = [[m0/(m0+m1), m1/(m0+m1), m2/(m2+m3), m3/(m2+m3)] for m0,m1,m2,m3 in mtx]
    mtx_transposed = np.array(mtx).transpose()
    plt.figure(num=None, figsize=(15, 5), dpi=100, facecolor='w', edgecolor='k')
    plt.imshow(mtx_transposed, interpolation='nearest',cmap='Wistia')
    plt.title("Confusion matrix for each class")
    tick_marks = np.arange(n_classes)
    plt.yticks(np.arange(4), ['1 - 1 (tp)','1 - 0 (fn)','0 - 0 (tn)','0 - 1 (fp)'])
    plt.xticks(tick_marks, classes, rotation=35)
    for i, j in itertools.product(range(n_classes), range(4)):
        plt.text(i, j, round(mtx[i][j],2), horizontalalignment="center")

    #plt.tight_layout()
    plt.xlabel('labels')
    plt.ylabel('Predicted')
    plt.show()


