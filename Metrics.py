import torch

import torch

def calc_eval_metrics(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape

    # TP == INTERSECTION = PRED==1 GT==1
    # FP = PRED==1 GT ==0
    # FN = PRED==0 GT==1
    # TN = PRED==0, Gt==0

    outputs = outputs.int()
    labels = labels.int()
    #PQ = []
    DICE = []
    IOU = []
    confusion_vector = outputs / labels
    for batch in range(outputs.shape[0]):
        for mask in range(outputs.shape[1]):
            TP = torch.sum(confusion_vector[batch,mask,...] == 1).item()
            FP = torch.sum(confusion_vector[batch,mask,...]  == float('inf')).item()
            TN = torch.sum(torch.isnan(confusion_vector[batch,mask,...])).item()
            FN = torch.sum(confusion_vector[batch,mask,...]  == 0).item()
            if FN != 0 or FP != 0 or TP != 0:
              #print("TP: {},  FP: {},  FN: {}".format(str(TP),str(FP),str(FN)) )
              IOU.append((TP ) / (TP + FP + FN))
              #PQ.append(((IOU ) / ((TP + 0.5 * FP + 0.5 * FN) )))
              DICE.append((2*TP) / (2*TP + FP + FN))
    return (sum(DICE)/len(DICE)), (sum(IOU)/len(IOU))
