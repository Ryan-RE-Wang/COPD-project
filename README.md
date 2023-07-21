# COPD-project

## Paper
This work is currently available on arXiv (https://arxiv.org/abs/2211.06925).

If you find this work useful, we would appreciate you citing this paper.
```
@article{wang2022early,
  title={Early Diagnosis of Chronic Obstructive Pulmonary Disease from Chest X-Rays using Transfer Learning and Fusion Strategies},
  author={Wang, Ryan and Chen, Li-Ching and Moukheiber, Lama and Moukheiber, Mira and Moukheiber, Dana and Zaiman, Zach and Moukheiber, Sulaiman and Litchman, Tess and Seastedt, Kenneth and Trivedi, Hari and others},
  journal={arXiv preprint arXiv:2211.06925},
  year={2022}
}
```

## Introduction
Chronic obstructive pulmonary disease (COPD) is one of the most common chronic illnesses in the world and the third leading cause of mortality worldwide but is often underdiagnosed. COPD is often not diagnosed until later in the disease course because spirometry tests are not sensitive to patients in the early stage. Currently, no research applies deep learning (DL) algorithms to detect COPD in early-stage patients using only chest X-rays (CXRs). To prevent diagnostic delays and underdiagnosis, we aim to develop DL algorithms to detect COPD using CXRs. We use three CXR datasets in our study, CheXpert to pre-train models, MIMIC-CXR, and Emory-CXR to develop and validate our fusion models. The CXRs from patients in the early stage of COPD and not on mechanical ventilation are selected for model training and validation. Our motivation for this project is to use fusion strategies for COPD prediction as there has not been any prospective validation that the model accurately screens for COPD. This study is more of a first step to demonstrate what is possible. We also released our code to the community. Although we did not find significant differences between the fusion strategies, future researchers working on similar clinical problems can easily apply those different strategies. 
