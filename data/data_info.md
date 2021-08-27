The data are organized in a dictionary saved in json format. The dictionary contain three keys:
* target: predicted variable
* group: grouping variable to use when there are multiple observations per participant. If a grouping variable is not necessary, this key takes `None` as value
* data: relative path of the .hdr files

Two dictionaries are provided in this folder, one for each dataset used in this project. 

To create dictionary in that format from fmri and behavioral data, see the [data_to_json.py](https://github.com/PSY6983-2021/picard_project/blob/main/scripts/data_to_json.py) script.
