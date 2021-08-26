#!/usr/bin/env python

# ./main.py --path "data_FEPS.json" --seed 42

import json
import pickle
import prepping_data
import building_model
import numpy as np
import nibabel as nib
from nilearn.masking import unmask
from sklearn.linear_model import Lasso, Ridge
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from argparse import ArgumentParser

def main():
    parser = ArgumentParser()
    parser.add_argument("--path", type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, choices=["whole-brain","M1","without M1"], default="whole-brain")
    parser.add_argument("--reg", type=str, choices=['lasso','ridge','svr'], default='lasso')
    args = parser.parse_args()

    #Loading the dataset
    data = open(args.path, "r")
    data_feps = json.loads(data.read())

    #Predicted variable
    y = np.array(data_feps["target"])
    
    #Group variable: how the data is grouped (by subjects)
    gr = np.array(data_feps["group"])

    #Convert fmri files to Nifti-like objects
    array_feps = prepping_data.hdr_to_Nifti(data_feps["data"])
    #Extract signal from gray matter
    if args.model == "whole-brain":
        masker, extract_X = prepping_data.extract_signal(array_feps, mask="template", standardize = True)
    elif args.model == "M1":
        mask_M1 = nib.load("mask_BA4.nii")
        extract_X = prepping_data.extract_signal_from_mask(array_feps, mask_M1)
    elif args.model == "without M1":
        mask_NoM1 = nib.load("mask_excluding_BA4.nii")
        extract_X = prepping_data.extract_signal_from_mask(array_feps, mask_NoM1)
    
    #Standardize the signal
    stand_X = StandardScaler().fit_transform(extract_X.T)
    X = stand_X.T
    
    #Compute the model
    if args.reg == "lasso":
        reg = Lasso()
    elif args.reg == "ridge":
        reg = Ridge()
    elif args.reg == "svr":
        reg = SVR(kernel="linear")
    
    X_train, y_train, X_test, y_test, y_pred, model, model_voxel = building_model.train_test_model(X, y, gr,reg=reg)

    if args.model == "whole-brain" :
        for i, element in enumerate(model_voxel):
            (masker.inverse_transform(element)).to_filename(f"coefs_whole_brain_{i}.nii.gz")
        
        model_to_averaged = model_voxel.copy()
        model_averaged = sum(model_to_averaged)/len(model_to_averaged)
        (masker.inverse_transform(model_averaged)).to_filename("coefs_whole_brain_ave.nii.gz")

    else :
        array_model_voxel = []
        if args.model == "M1" :
            unmask_model = unmask(model_voxel, mask_M1)
        if args.model == "without M1": 
            unmask_model = unmask(model_voxel, mask_NoM1)

        for element in unmask_model:
            array_model_voxel.append(np.array(element.dataobj))

        model_ave = sum(array_model_voxel)/len(array_model_voxel)
        model_to_nifti = nib.nifti1.Nifti1Image(model_ave, affine = array_feps[0].affine)
        model_to_nifti.to_filename(f"coefs_{args.model}_ave.nii.gz")
    
    
    for i in range(len(X_train)):
        filename = f"train_test_{i}.npz"
        np.savez(filename, X_train=X_train[i],y_train=y_train[i],X_test=X_test[i],y_test=y_test[i],y_pred=y_pred[i])

    #Saving the model
    filename_model = f"lasso_models_{args.model}.pickle" 
    pickle_out = open(filename_model,"wb")
    pickle.dump(model, pickle_out)
    pickle_out.close()

    #Compute permutation tests
    score, perm_scores, pvalue = building_model.compute_permutation(X, y, gr, random_sedd=args.seed)
    perm_dict = {'score': score, 'perm_scores': perm_scores.tolist(), 'pvalue': pvalue}
    filename_perm = f"permutation_output_{args.model}_{args.seed}.json"
    with open(filename_perm, 'w') as fp:
        json.dump(perm_dict, fp)
    

if __name__ == "__main__":
    main()
