

def acc_metrics(outputs, labels):
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    # True Positive
    TP += ((outputs == 1) & (labels == 1)).sum()
    # True Negative
    TN += ((outputs == 0) & (labels == 0)).sum()
    # False Negative
    FN += ((outputs == 0) & (labels == 1)).sum()
    # False Positive
    FP += ((outputs == 1) & (labels == 0)).sum()

    p = TP / (TP + FP)
    r = TP / (TP + FN)
    F1 = 2.0 * r * p / (r + p)
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    return p, r, F1, accuracy
    