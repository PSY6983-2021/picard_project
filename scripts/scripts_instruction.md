To use the main.py script, the data need to be in a dictionary format. If they are not already in a dictionary format, please check the data_to_json.py script.

The main.py script can be run from the terminal using that line: ./main.py --path_0 "data_FEPS_0.json" --path_1 "data_FEPS_1.json" --seed 42 --model "whole-brain" --reg "lasso"
<br>You can also see the [submit.sh](https://github.com/PSY6983-2021/picard_project/blob/main/scripts/submit.sh) file for an example of a bash script to run the analysis on Compute Canada.

The main.py takes different arguments:
* path_0: the path to the first dataset .json file
* path_1: the path to the second dataset .json file (could be removed if not needed)
* seed: 
* model: mask strategy to extract the fMRI signal. Either "whole-brain", "M1" or "without M1".
* reg: regression technique to use for the prediction. Either "lasso", "ridge" or "svr". 

Note that the current main.py script has been written to take into account two datasets (data_FEPS_0.json and data_FEPS_1.json). The main.py script could easily be adjusted to to support only one dataset.

The main.py script contains code to save:
* Regression model coefficients (in .nii.gz format)
* Regression model (in .pickle format)
* Train/test sets and the predicted values (in .npz format)
* Permutation test outputs (in .json format)

Please keep in mind that the scripts were only tested on 3D fmri voxel-level data, not on 4D fmri data timeseries. Also the scripts will need to be adjust to work with ROIs.

If you have any questions about the scripts and the project in general, please contact me at marie-eve.picard.2@umontreal.ca. 
