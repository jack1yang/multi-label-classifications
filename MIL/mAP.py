import  numpy as np
#from sklearn.metrics import roc_auc_score

"""
author: ysk
descriptions: calculate ap of each class in multi-label classification
refernces: https://zhuanlan.zhihu.com/p/69747388, http://blog.sina.com.cn/s/blog_9db078090102whzw.html
input: predictions, gts
output: each class ap and mAP
"""


def eval_ap(prec, rec, use_07_metric=False):
    """
    Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(cls, predictions, gts):
    """
    predictions: (batch_size, class_num)
    gts: (batch_size, class_num)
    output: prec and recal and ap
    """
    num_exaples = len(predictions)
    gts = gts[:, cls]
    confindences = predictions[:, cls]

    sorted_inds = np.argsort(-confindences)
    true_positives = gts[sorted_inds] > 0
    false_positives = gts[sorted_inds] == 0

    # calculate every interval true_positives and false_positives
    true_positives = np.cumsum(true_positives)
    false_positives = np.cumsum(false_positives)
    

    rec = true_positives / (np.sum(gts > 0)+0.0)
    eps = 1e-10

    positives = true_positives + false_positives
    prec = true_positives / (positives + (positives == 0.0) * eps)

    ap = eval_ap(prec, rec, True)
    return ap



if __name__=='__main__':
    predictions = np.loadtxt('results/predictions_lateral_224avg7_epoch39.txt')
    gts = np.loadtxt('results/gts_lateral_224avg7_epoch39.txt')
    outs = []
    #print(np.sum(gts[:,-5]))
    for cls in range(14):
        #print(roc_auc_score( gts[:,cls],predictions[:,cls]))
        ap = voc_eval(cls, predictions, gts)
        outs.append(ap)
    print(np.mean(outs))
    print(outs)
    