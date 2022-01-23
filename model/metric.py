from keras import backend as K

def dice_coef(y_true, y_pred):
    smooth = 0.00001
    y_true_f = K.flatten(y_true)
    y_pred = K.cast(y_pred, 'float32')
    y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.5), 'float32')
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / ((K.sum(y_true_f) + K.sum(y_pred_f)) + smooth)
    return score


# Recall (true positive rate)
def recall(truth, prediction):
    TP = K.sum(K.round(K.clip(truth * prediction, 0, 1)))
    P = K.sum(K.round(K.clip(truth, 0, 1)))
    return TP / (P + K.epsilon())


# Precision (positive prediction value)
def precision(truth, prediction):
    TP = K.sum(K.round(K.clip(truth * prediction, 0, 1)))
    FP = K.sum(K.round(K.clip((1-truth) * prediction, 0, 1)))
    return TP / (TP + FP + K.epsilon())