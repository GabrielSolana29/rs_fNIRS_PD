
## Integrating fNIRS and Machine Learning: Shedding Light on Parkinson's Disease Detection
---
About the project
---
This project presents a methodology for the analysis of fNIRS signals with Machine learning for assisting in the diagnosis of Parkinson's disease(PD).

The following steps are performed with this code:
1. Dataset creation
2. Feature extraction 
3. Feature selection and finding the best subset
4. Classification

---
**Getting Started**


**Requirements**
- python 3.9.18
- pandas 2.1.2
- scikit-learn 1.3.2
- numpy 1.23.5
- xgboost 2.0.1
- statsmodels 0.14.0
- scipy 1.11.3
- seaborn 0.13.0
- feature-selection-ga 0.1.3


**Dataset**

For ease of use a CSV file named "complete_dataset.csv" with the fNIRS data from PD and controls can be found in this repository. After downloading the file, store it in the CSV folder.

- https://app.box.com/s/qt238sdt8udozm29vf8rp3jo6146fk2r

Original fNIRS resting-state signals from patients with PD and controls can be found in:

- Guevara, E., Rivas-Ruvalcaba, F. J., Kolosovas-Machuca, E. S., Ramírez-Elías, M. G., Díaz de Leon Zapata, R., Ramirez-GarciaLuna, J. L., & Ildefonso Rodriguez-Leyva. (2023). Functional Near-Infrared Spectroscopy Reveals Delayed Hemodynamic Changes in the Primary Motor Cortex During Fine Motor Tasks and Decreased Interhemispheric Connectivity in Parkinson's Disease Patients (1.0.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.7966830

**Pre-processing**

***Load the dataset***

*Recomended*
Download the CSV "complete_dataset.csv" as described in the previous section and store it in the CSV folder.

*Not recommended*
Create a new dataset CSV file with the original data, download all the signals from (Guevara, E., Rivas-Ruvalvaba, F.F., Kolosova-Machuca, et al. 2023) in .tsv format and add them to the CSV folder. Finally, run the "create_dataset.py script".

***Parameters***
The number of folds for the cross-validation of the classifiers and training algorithms can be modified by changing the variable *cv* in the main.py in the main program.

***Execution***

The main script performs **Feature extraction**, **Feature selection**,**Training of model**, and **Tests the models**. A performance report will be generated after running the script with the performance metrics from the algorithms by using different feature selection methods.