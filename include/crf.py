import numpy as np
from os import path
import torch
from include.network import SegmentationModel
from torch.nn.modules.upsampling import Upsample
import torch.optim
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional

# Install package via 
# pip3 install git+https://github.com/lucasb-eyer/pydensecrf.git
import pydensecrf.densecrf as dcrf

"""
Computes energy as a sum of unary and pairwise energy. 
This implementation uses numpy arrays as the input and then calls the C++ implementation
by Krähenbühl to perform CRF via efficient message passing
"""
def applyDenseCRF(unary,image, sigma_alpha, sigma_beta, sigma_gamma, w1, w2, iterations=10):
    h = image.shape[0]
    w = image.shape[1]
    L = unary.shape[0]

    d = dcrf.DenseCRF2D(w,h,L)
    d.setUnaryEnergy(unary)
    d.addPairwiseBilateral(sxy=(sigma_alpha,sigma_alpha), srgb=(sigma_beta,sigma_beta,sigma_beta), rgbim=image, compat=w1, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    d.addPairwiseGaussian(sxy=(sigma_gamma,sigma_gamma), compat=w2, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    energy = d.inference(iterations)
    return energy

"""
Performs grid search over parameter space for CRF.
Returns the total improvement in cross entropy for each candidate parameter measure
Make sure the data loader has batch size 1 and does NOT normalize or else it wont work!
"""
def gridSearchCRFParameters(dcnn, inf_loader, normalization_transform, number_of_classes, w2 = 3, s_g = 3, candidate_w1 = np.arange(3,6), candidate_s_a = np.arange(30,100,10), candidate_s_b = np.arange(3,6), iterations=10):
    # Find CRF parameters using grid search
    total_improvement = np.zeros((len(candidate_w1), len(candidate_s_a), len(candidate_s_b)))
    number_of_parameters = len(total_improvement.reshape(-1))


    FP = np.zeros((number_of_classes))
    TP = np.zeros((number_of_classes))
    FN  = np.zeros((number_of_classes))
    FP_afterCRF = np.zeros((number_of_parameters, number_of_classes))
    TP_afterCRF = np.zeros((number_of_parameters, number_of_classes))
    FN_afterCRF = np.zeros((number_of_parameters, number_of_classes))
    for i, data in enumerate(inf_loader, 0):
        input_image, ground_truth, _, raw_input_image = data
        raw_input_image = raw_input_image[0].clone()

        input_image = input_image.cuda().detach()
        ground_truth = ground_truth.detach()

        unary = dcnn(input_image).detach()
        
        upsample = nn.Upsample(size=(input_image.shape[2],input_image.shape[3]), mode='bilinear', align_corners=True)
        unary_upsampled = upsample(unary)
        
        unary_upsampled = nn.functional.softmax(unary_upsampled[0], dim=0)
        Result=np.argmax(unary_upsampled.cpu().numpy(), axis=0)
        GT=ground_truth.numpy()
        for n in range(number_of_classes):
            A = GT == n
            B = Result == n
            TP[n] += np.sum(B & A)
            FP[n] += np.sum(B & ~A)
            FN[n] += np.sum(~B & A)

    
        # Convert to numpy format for CRF
        unary_upsampled_np = unary_upsampled.cpu().numpy()
        # Needs to be C-contiguous, that's what the order=C is for
        input_image_np = (raw_input_image*255).byte().cpu().numpy().copy(order='C') 
        w = unary_upsampled_np.shape[1]
        h = unary_upsampled_np.shape[2]
        L = unary_upsampled_np.shape[0]
        unary_upsampled_np = unary_upsampled_np.reshape((L,-1))
        unary_upsampled_np = - np.log(unary_upsampled_np)
    
        parameter_i = 0
        print("Image %d" % (i))
        for w1_i in range(len(candidate_w1)):
            for s_a_i in range(len(candidate_s_a)):
                for s_b_i in range(len(candidate_s_b)):
                    w1 = candidate_w1[w1_i]
                    s_a = candidate_s_a[s_a_i]
                    s_b = candidate_s_b[s_b_i]
                    

                    after_crf = applyDenseCRF(unary_upsampled_np, input_image_np,s_a, s_b, s_g, w1, w2, iterations)
                    after_crf = np.array(after_crf).reshape(L,w,h)
                    Result_after_crf = np.argmax(after_crf, axis=0)
                    for n in range(number_of_classes):
                        A = GT == n
                        B = Result_after_crf == n
                        TP_afterCRF[parameter_i, n] += np.sum(B & A)
                        FP_afterCRF[parameter_i, n] += np.sum(B & ~A)
                        FN_afterCRF[parameter_i, n] += np.sum(~B & A)
                    parameter_i+=1
                    
    parameter_i = 0
    for w1_i in range(len(candidate_w1)):
        for s_a_i in range(len(candidate_s_a)):
            for s_b_i in range(len(candidate_s_b)):
                IOU = np.mean(TP / (TP+FP+FN))
                IOU_crf = np.mean(TP_afterCRF[parameter_i] / (TP_afterCRF[parameter_i] + FN_afterCRF[parameter_i] + FP_afterCRF[parameter_i]))
                total_improvement[w1_i, s_a_i, s_b_i] += (IOU_crf - IOU)
                parameter_i+=1

    return total_improvement
