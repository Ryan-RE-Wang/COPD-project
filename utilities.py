import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, balanced_accuracy_score
from sklearn.metrics import classification_report, average_precision_score
import matplotlib.pyplot as plt
import tensorflow as tf
import itertools


def scheduler(epoch, lr):
    if epoch % 2 == 0:
        return lr * tf.math.exp(-0.05)
    else:
        return lr

# def binary_focal_loss(y_true, y_pred):
#     return tfa.losses.sigmoid_focal_crossentropy(
#         y_true,
#         y_pred,
#         alpha=1,
#         gamma=2,
#         from_logits=True
#     )

def get_tpr(y_test, preds, thresh):
    tn, fp, fn, tp = confusion_matrix(y_test, np.where(preds >= thresh, 1, 0)).ravel()
    
    return tp/(tp+fn)

def get_thresh(y_test, y_preds, thresh_type='Youden'):
    fprs, tprs, thresh = roc_curve(y_test, y_preds, drop_intermediate=False)
    
    if (thresh_type == 'Youden'):
        value = tprs-fprs
    else: # G-mean
        value = np.sqrt(tprs * (1 - fprs))
                
    idx = np.argmax(value)
    
    return thresh[idx]

def test_CI(y_preds, y_test, thresh):
    
    n_bootstraps = 1000
    rng_seed = 42  # control reproducibility
    auc = []
    pre = []
    rec = []
    f1 = []
    auprc = []
    bal_acc = []

    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_preds), len(y_preds)) # random sample another part and get auc --> get 1000 results 
        if len(np.unique(y_test[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue

        auc.append(roc_auc_score(y_test[indices], y_preds[indices]))
        
        bal_acc.append(balanced_accuracy_score(y_test[indices], np.where(y_preds[indices] >= thresh, 1, 0)))
        
        cr = classification_report(y_test[indices], np.where(y_preds[indices] >= thresh, 1, 0), output_dict=True)
        pre.append(cr['1']['precision'])
        rec.append(cr['1']['recall'])
        f1.append(cr['1']['f1-score'])
        
        auprc.append(average_precision_score(y_test[indices], y_preds[indices], average=None))
        
    
    auc = np.array(auc)
    pre = np.array(pre)
    rec = np.array(rec)
    f1 = np.array(f1)
    auprc = np.array(auprc)
    bal_acc = np.array(bal_acc)
    
    get_CI('AUC      ', auc)
    get_CI('Precision', pre)
    get_CI('Recall   ', rec)
    get_CI('F1-Score ', f1)
    get_CI('AUPRC    ', auprc)
    get_CI('Balanced ACC', bal_acc)

def get_CI(item, arr):
    mean_score = arr.mean()
    std_dev = arr.std()
    std_error = std_dev / np.math.sqrt(1)
    ci =  2.262 * std_error
    lower_bound = mean_score - ci
    upper_bound = mean_score + ci
    
    print(item, ": {:0.2f}, CI: [{:0.2f} - {:0.2f}]".format(mean_score, lower_bound, upper_bound))
    
def plot_cm(y_test, y_preds, thresh):
    cf = confusion_matrix(y_test, np.where(y_preds >= thresh, 1, 0))

    plt.imshow(cf,cmap=plt.cm.Blues,interpolation='nearest')
    plt.colorbar()
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    tick_marks = np.arange(len(set(y_test))) # length of classes
    class_labels = ['0','1']

    plt.xticks(tick_marks,class_labels)
    plt.yticks(tick_marks,class_labels)
    # plotting text value inside cells
    thresh = cf.max() / 2.
    for i,j in itertools.product(range(cf.shape[0]),range(cf.shape[1])):
        plt.text(j,i,format(cf[i,j],'d'),horizontalalignment='center',color='white' if cf[i,j] >thresh else 'black')

    plt.show()