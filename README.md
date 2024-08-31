# COPD-project

## Paper
This work is published on International Journal of Medical Informatics (https://www.sciencedirect.com/science/article/abs/pii/S1386505623002290).


## Abstract

# Purpose
Chronic obstructive pulmonary disease (COPD) is one of the most common chronic illnesses in the world. Unfortunately, COPD is often difficult to diagnose early when interventions can alter the disease course, and it is underdiagnosed or only diagnosed too late for effective treatment. Currently, spirometry is the gold standard for diagnosing COPD but it can be challenging to obtain, especially in resource-poor countries. Chest X-rays (CXRs), however, are readily available and may have the potential as a screening tool to identify patients with COPD who should undergo further testing or intervention. In this study, we used three CXR datasets alongside their respective electronic health records (EHR) to develop and externally validate our models.

# Method
To leverage the performance of convolutional neural network models, we proposed two fusion schemes: (1) model-level fusion, using Bootstrap aggregating to aggregate predictions from two models, (2) data-level fusion, using CXR image data from different institutions or multi-modal data, CXR image data, and EHR data for model training. Fairness analysis was then performed to evaluate the models across different demographic groups.

#Results
Our results demonstrate that DL models can detect COPD using CXRs with an area under the curve of over 0.75, which could facilitate patient screening for COPD, especially in low-resource regions where CXRs are more accessible than spirometry.

# Conclusions
By using a ubiquitous test, future research could build on this work to detect COPD in patients early who would not otherwise have been diagnosed or treated, altering the course of this highly morbid disease.
