import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

# Calculate the Quality of the Segmentation
def IOU(Result,Ground_Truth):
    Classes_gt=list(np.unique(Ground_Truth))
    Classes_re=list(np.unique(Result))
    Classes=Classes_gt+Classes_re
    if 20 in Classes: Classes.remove(20)
    Classes=np.unique(Classes)
    iou=np.zeros(len(Classes))
    i=0
    for n in Classes:
        A = Ground_Truth == n
        B = Result == n
        TP = np.sum(B & A)
        FP = np.sum(B & ~A)
        FN = np.sum(~B & A)
        U = ( TP + FP + FN )
        iou[i] = TP / ( TP + FP + FN )
        i=i+1
    return np.mean(iou), iou
# Validation function with Cross Entropy
def Validate_CrossEntropy(net,optimizer,epoch,losses,Val_Score,bsize,lr,VAL_Loader,best_score, filename):
    score=0.0
    loss=0.0
    loss_fn = nn.CrossEntropyLoss()
    for data in VAL_Loader:
        src_img, seg_img, seg_img_ds, _ = data

        Input = Variable(src_img, requires_grad=False).float().cuda()
        Target = Variable(seg_img_ds.long(), requires_grad=False).cuda()
        
        Output = net(Input)
        
        loss = loss_fn(Output, Target)
        score+=(loss.cpu().data)/1449.0
        
    if (score < best_score):
        best_score=score
        filename=filename
        state={
                'epoch'         : epoch,
                'state_dict'    : net.state_dict(),
                'optimizer'     : optimizer.state_dict(),
                'losses'        : losses,
                'Val_Score'     : Val_Score,
                'batch_size'    : bsize,
                'learning_rate' : lr,
                'best_score'    : best_score
            }
        torch.save(state, filename)
        print("New Best Validation Score! Epoch = {} , Best_CrossEntropy = {}, Checkpoint Done!".format(epoch,best_score))
    return best_score,score

# Validation function with IOU
def Validate_IOU(net,optimizer,epoch,losses,Val_Score,bsize,lr,VAL_Loader,best_score, filename):
    score=0.0

    val_set_size = 0
    for data in VAL_Loader:
        src_img, seg_img, seg_img_ds, _ = data
        val_set_size += src_img.shape[0]

        Input = Variable(src_img, requires_grad=False).float().cuda()
        Target = Variable(seg_img.long(), requires_grad=False).cuda()
        
        Output = net(Input)
        upsampler = nn.Upsample(size=(src_img.shape[2],src_img.shape[3]), mode='bilinear', align_corners=True)
        Output_upsampled = upsampler(Output)
        for i in range(Output.shape[0]):
            out=nn.functional.softmax(Output_upsampled[i,:,:,:], dim=0)
            out=out.cpu()
            out=out.detach().numpy()
            Result=np.argmax(out,axis=0)
            GT=Target[i,:,:].cpu().numpy()
            cur_iou , _ = IOU(Result, GT)
            score+= cur_iou
        
    score = score / val_set_size
    if (score > best_score):
        best_score=score
        filename=filename
        state={
                'epoch'         : epoch,
                'state_dict'    : net.state_dict(),
                'optimizer'     : optimizer.state_dict(),
                'losses'        : losses,
                'Val_Score'     : Val_Score,
                'batch_size'    : bsize,
                'learning_rate' : lr,
                'best_score'    : best_score
            }
        torch.save(state, filename)
        print("New Best Validation Score! Epoch = {} , Best_IOU = {}, Checkpoint Done!".format(epoch,best_score))
    return best_score,score



