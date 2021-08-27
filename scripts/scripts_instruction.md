To use the main.py script, the data need to be in a dictionary format. If they are not already in a dictionary format, please check the data_to_json.py script.

The main.py script contains code to save:
* Regression model coefficients (in .nii.gz format)
* Regression model (in .pickle format)
* Train/test sets and the predicted values (in .npz format)
* Permutation test outputs (in .json format)

Please keep in mind that the scripts were only tested on 3D fmri voxel-level data, not on 4D fmri data timeseries. Also the scripts will need to be adjust to work with ROIs.

If you have any questions about the scripts and the project in general, please contact me at marie-eve.picard.2@umontreal.ca. 
