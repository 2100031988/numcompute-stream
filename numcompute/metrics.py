import numpy as np


# =========================
# ACCURACY
# =========================

def accuracy(y_true, y_pred):                       # accuracy is defined as correct predictions divided by total "predictions" where we convert the input lists to numpy arrays and 
                                                    # then calculate the "mean" of the boolean array where we "compare" the true labels with the predicted labels. We can use it to calculate experiment results, model performance, etc.

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return np.mean(y_true == y_pred)


# =========================
# CONFUSION MATRIX
# =========================

def confusion_matrix(y_true, y_pred):               # a common evaluation metric used in machine learning and it gives matrix showing the counts of true positives, true negatives, false positives 
                                                    # and false negatives for a classification problem. Use case includes disease detection, ail spam detection ,etc.

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    labels = np.unique(np.concatenate([y_true, y_pred]))
    label_to_idx = {label: i for i, label in enumerate(labels)}

    cm = np.zeros((len(labels), len(labels)), dtype=int)

    for t, p in zip(y_true, y_pred):
        cm[label_to_idx[t], label_to_idx[p]] += 1

    return cm


# =========================
# PRECISION
# =========================

def precision(y_true, y_pred, average="binary"):            # precision is defined as true positives divided by the sum of true positives and false positives where we first compute the confusion matrix and then calculate 
                                                            # the "true positives" and "false positives" from the confusion matrix. Use case includes spam detection, fraud detection, etc.
    cm = confusion_matrix(y_true, y_pred)

    tp = np.diag(cm)
    fp = np.sum(cm, axis=0) - tp

    precision_vals = tp / (tp + fp + 1e-12)

    if average == "binary":
        return precision_vals.mean()
    elif average == "macro":
        return np.mean(precision_vals)
    else:
        raise ValueError("Invalid average method")


# =========================
# RECALL
# =========================

def recall(y_true, y_pred, average="binary"):           # recall is defined as true positives divided by the sum of the true positives and false negatives where we first compute the confusion matrix and 
                                                        # then calculate the "true positives" and "false negatives" from the confusion matrix. Use case includes fety systems, medical diagnosis, etc.

    cm = confusion_matrix(y_true, y_pred)

    tp = np.diag(cm)
    fn = np.sum(cm, axis=1) - tp

    recall_vals = tp / (tp + fn + 1e-12)

    if average == "binary":
        return recall_vals.mean()
    elif average == "macro":
        return np.mean(recall_vals)
    else:
        raise ValueError("Invalid average method")


# =========================
# F1 SCORE
# =========================

def f1_score(y_true, y_pred, average="binary"):    # f1 score is defined as the harmonic mean of precision and recall where we first calculate the "precision" and "recall" using the respective functions and then
                                                   # compute the F1 score. Use case includes imbalanced datasets, sentimental analysis in NLP, etc.
    
    p = precision(y_true, y_pred, average)
    r = recall(y_true, y_pred, average)
    return 2 * p * r / (p + r + 1e-12)


# =========================
# MEAN SQUARED ERROR
# =========================

def mse(y_true, y_pred):                           # mean squared error is defined as the "average" of the squared differences between the "true values" and the "predicted values" where we convert the input lists to 
                                                    # np arrays and then calculate the "mean" of the squared differences. Use case includes regression problems, forecasting, etc.    
        
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean((y_true - y_pred) ** 2)


# =========================
# ROC CURVE (binary only)
# =========================

def roc_curve(y_true, y_score):                 # the ROC (Receiver Operating Characteristic) curve is a graphical representation of the "true positive" rate against the "false positive" rate at various threshold settings. Moreover, it help us to 
                                                # understand the trade-off between "sensitivity" and "specificity". Use case includes biomedical applications, signal detection, etc.
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    thresholds = np.sort(np.unique(y_score))[::-1]

    tpr_list = []
    fpr_list = []

    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)

    for t in thresholds:
        y_pred = (y_score >= t).astype(int)

        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))

        tpr = tp / (P + 1e-12)
        fpr = fp / (N + 1e-12)

        tpr_list.append(tpr)
        fpr_list.append(fpr)

    return np.array(fpr_list), np.array(tpr_list)


# =========================
# AUC (Trapezoidal rule)
# =========================

def auc(fpr, tpr):                  # AUC (Area Under the Curve) is a metric used to evaluate the performance of binary classification models where we use the "trapezoidal rule" to calculate the area under the ROC curve.

    fpr = np.asarray(fpr)
    tpr = np.asarray(tpr)

    return np.trapz(tpr, fpr)