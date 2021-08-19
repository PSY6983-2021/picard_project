import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.masking import apply_mask
from nilearn.input_data import NiftiMasker
from nilearn.image import resample_img
from sklearn.preprocessing import FunctionTransformer

def hdr_to_Nifti(files):
    """
    Convert hdr files to Nifti-like objects

    Parameters
    ----------
    files: list of paths to each hdr file

    Returns
    ----------
    array: array of Nifti-like objects
    """
    array = []
    for element in files:
        array = np.append(array, nib.load(element))

    print('array size: ', array.shape, '\narray type: ', type(array))

    return array 


def y_transformer(y, func = np.log1p):
    """
    Transform y using a specified function

    Parameters
    ----------
    y: variable to transform
    func: list of numpy transformations to apply to the variable

    Returns
    ----------
    df_y: DataFrame containing y and the transformed y according to the specified transformations
    """
    df_y = pd.DataFrame(y.tolist(), columns = ["y"])

    for element in func:
        transformer = FunctionTransformer(element, validate=True)
        Y_transformed = transformer.fit_transform(y.reshape(-1,1))
        Y_transformed = Y_transformed[:,0]
        df_y[str(element).replace("<ufunc '", ""). replace("'>","")] = Y_transformed.tolist()

    return df_y


def extract_signal(data, mask="template", standardize = True):
    """
    Apply a mask to extract the signal from the data and save the mask
    in a html format

    Parameters
    ----------
    data: list of Niimg-like objects
    mask: strategy to compute the mask. By default the gray matter is extracted based on the MNI152 brain mask
    standardize: strategy to standardize the signal. The signal is z-scored by default

    Returns
    ----------
    masker_all: mask of the data
    masker_gm: array containing the extracted signal

    See also NifitMasker documentation
    """
    masker_all = NiftiMasker(mask_strategy = mask,standardize=standardize, verbose = 1, reports = True)
    
    masker_gm = masker_all.fit_transform(data)
    print("mean: ", round(masker_gm.mean(),2), "\nstd: ", round(masker_gm.std(),2))
    print("Signal size: ", masker_gm.shape)

    report = masker_all.generate_report()
    report.save_as_html("masker_report.html")

    return masker_all, masker_gm

def extract_signal_from_mask(data, mask):
    """
    Apply a pre-computed mask to extract the signal from the data

    Parameters
    ----------
    data: Niimg-like object
    mask: mask to apply to the data

    Returns
    ----------
    signal: extracted signal from mask

    See also nilearn masking documentation
    """
    affine = data[0].affine
    resample_mask = resample_img(mask,affine)
    signal = apply_mask(data, resample_mask, ensure_finite=True)
    print(signal.shape, type(signal))

    return signal
