#!/usr/bin/env python

#sudo mount -t drvfs D: /mnt/d
# ./main.py --path "data_FEPS.json" --seed 42

import json
import pickle
import prepping_data
import building_model
import metrics_feps
import numpy as np
import nibabel as nib
from argparse import ArgumentParser

def main():
    parser = ArgumentParser()
    parser.add_argument("--path", type=str)
    parser.add_argument("--seed", type=int)
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
    masker, X = prepping_data.extract_signal(array_feps, mask="template", standardize = True)

    #Compute the model
    X_train, y_train, X_test, y_test, y_pred, model = building_model.train_test_model(X, y, gr)
    for i in range(len(X_train)):
        filename = "train_test_"+str(i)+".npz"
        np.savez(filename, X_train=X_train[i],y_train=y_train[i],X_test=X_test[i],y_test=y_test[i],y_pred=y_pred[i])

    #Saving the model
    filename_model = "lasso_models.pickle" 
    pickle_out = open(filename_model,"wb")
    pickle.dump(model, pickle_out)
    pickle_out.close()
    

    #Compute permutation tests
    score, perm_scores, pvalue = building_model.compute_permutation(X, y, gr, n_permutations=3)
    perm_dict = {'score': score, 'perm_scores': perm_scores.tolist(), 'pvalue': pvalue}
    with open('permutation_output.json', 'w') as fp:
        json.dump(perm_dict, fp)

    #compute bootstrap tests
    resampling_coef = building_model.boostrap_test(X, y, gr, n_resampling=3, random_seed=args.seed)
    filename_bootstrap = "bootstrap_models.pickle"
    pickle_out = open(filename_bootstrap,"wb")
    pickle.dump(resampling_coef, pickle_out)
    pickle_out.close()
    

if __name__ == "__main__":
    main()