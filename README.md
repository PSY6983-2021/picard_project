# Picard_Marie-Eve_project

## Summary

Education:
* B.Sc. in cognitive neuroscience
* Currently a Master's student in Psychology at Université de Montréal

I am interested in the neural correlates of pain and the communication of pain. The goal of my master project is to predict the occurrence and intensity of the facial expression of pain from brain activity (fMRI data) in response to a tonic painful stimuli. 

I'm also interested in the different methods used to analyze neuroimaging data.

<a href="https://github.com/me-pic">
   <img src="https://avatars.githubusercontent.com/u/77584086?v=4?s=100" width="100px;" alt=""/>	
</a>

# Project definition

## Background

Facial expression of pain is an important non-verbal informative element of communication, signaling an immediate threat and a need for help. Facial movements related to the experience of pain are associated not only with the sensory dimension of pain (i.e., pain intensity) but also with its affective dimension (i.e., the unpleasantness of pain). The facial expression can be measured with the Facial Action Coding System (FACS, Ekman & Friesen, 1978), which is based on the possible action of muscles or groups of muscles, called action units. We can compute a composite score (FACS score) taking into account the frequency and the intensity of the contraction of these action units. 

<p align="center">
<img src="https://github.com/PSY6983-2021/picard_project/blob/main/images/goal.png" width="400px"/>
</p>
   
**Personal goals**
* Apply machine learning to neuroimaging data
* Learn visualization tools
* Improve my coding skills
* Learn how to run analysis on compute canada and don't be afraid of it

## Tools

The completion of this project will require the following tool: 
* **Python scripts** to write the code for the analysis
* Several python modules: 
   * Pandas: to manipulate the behavioral dataframe
   * Numpy: to manipulate arrays
   * Nibabel: to load the fMRI contrast images (from .hdr files to Nifti objects)
   * Scikit-learn: for machine learning stuff
   * Nilearn: to extract and to visualize the fMRI data
   * Matplotlib, seaborn, plotly: to plot some figures
* The analysis will be run on **compute canada**
* The python scripts and figures will be added to the **github** repository 

## Data

The dataset that will be used for this project is a secondary private dataset (not open access yet). It includes 55 participants (women = 28, men = 27) aged between 18 and 53 years old. Moderately painful stimuli were applied while they where in the MRI scanner. Their facial expression was recorded during this time by the use of a MRI-compatible camera. Each participant completed 2 runs of 8 painful trials, resulting to 797 observations (after removing data with movement artefacs). 

Illustration of the experimental paradigme.
<p align="center">
<img src="https://github.com/PSY6983-2021/picard_project/blob/main/images/exp_paradigme.png" width="300px"/>
</p>

## Deliverables

By the end of this course, I will provide 
* python scripts for:
   - [ ] Prepping the dataset for the analysis
   - [ ] Machine learning pipeline
* jupyter notebook for:
   - [ ] Visualizing the results
* Static and interactive visualizations of the results
* MarkDown README.md documenting all important information about the project
* Github repository containing the scripts, graphs, report and relevant information

# Results

## Overview

## LASSO-PCR models

## Conclusion

## Acknowledgement

## References

Ekman, P. et Friesen, W. V. (1978). Facial action coding systems. Consulting Psychologists Press.

Kunz, M., Chen, J.-I., Lautenbacher, S., Vachon-Presseau, E. et Rainville, P. (2011). Cerebral regulation of facial expressions of pain. Journal of Neuroscience, 31(24), 8730-8738. https://doi.org/10.1523/JNEUROSCI.0217-11.2011

Vachon-Presseau, E., Roy, M., Woo, C.-W., Kunz, M., Martel, M.-O., Sullivan, M. J., Jackson, P. L., Wager, T. D. et Rainville, P. (2016). Multiple faces of pain: effects of chronic pain on the brain regulation of facial expression. Pain, 157(8), 1819-1830. https://doi.org/10.1097/j.pain.0000000000000587
