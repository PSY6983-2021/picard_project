import json
import os

def read_files(filesInput = None):
    """
    Load all functional files into Nifti format
    
    Parameters
    ----------
    filesInput: Folder where the functional images can be found

    Returns
    ----------
    files_hdr: list containing the path of each functional images
    array : ndarray containing the Nifti objects
    """
    path, dirs, files = next(os.walk(filesInput))

    files_hdr = list()
    for file in files:
        if file[-3:] == "hdr":
            files_hdr.append(path + "//" + file)
    
    files_hdr.sort(reverse=True)
    files_hdr.sort()

    return files_hdr


def check_order(dataframe, column, filenames, extension=".hdr"):
    """
    Check if the order of the fMRI files match the one in the
    behavioral file.
    
    Parameters
    ----------
    dataframe (pandas dataFrame) : behavioral dataframe
    column (String): name of the column containing the participants' ID
    filenames (list): list of paths where the fMRI data can be found
    extension (String): extension of the fMRI files
    
    Returns
    ----------
    order: either or not the behavioral file and the functional images are in the same order (boolean)
    """
    order = True
    ls_ID = dataframe[column].tolist()
    
    for i, filename in enumerate(filenames):
        if os.path.basename(filename[:-len(extension)]) != ls_ID[i]:
            print("Error in the files order")
            print(os.path.basename(filename[:-len(extension)]))
            print(ls_ID[i])
            order=False
            break
            
    return order


def save_to_json(dataframe, ID, target, group = None, files, save_name):
    """
    Create and save a json file containing the relative path of the hdr fmri files,
    the target variable and the group variable

    Parameters
    ----------
    dataframe: behavioral dataframe
    ID (string): name of the column in dataframe containing the ID variable
    target (string): name of the column in dataframe containing the target variable
    group (string): name of the column in dataframe containing the group variable 
    files: list of hdr file paths
    save_name (string): name of the file to create
    """
    if check_order(dataframe, ID, files):
        idx = files[0].find("//")
        for i in range(len(files)):
            files[i]=files[i][idx+1:]
            files[i]=root+files[i]
        if group == None:
            FEPS_data = {"target": dataframe[target].to_list(), "group": group, "data": files}
        else:
            FEPS_data = {"target": dataframe[target].to_list(), "group": dataframe[group].to_list(), "data": files}
        
        with open(save_name, 'w') as fp:
                json.dump(FEPS_data, fp)
    else: 
        print("Cannot save to json: fmri and behavioral data are not in the same order")
