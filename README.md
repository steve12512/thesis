The purpose of this project was to find a biased dataset, and try different methods to mitigate its bias.
That is done under the patients directory, in the patients.ipynb file.
The dataset used is this; https://www.kaggle.com/datasets/majdmustafa/diabetes-hospital-readmission-dataset
We firstly train a Random Forest Classifier on the dataset, in order to predict whether or not a patient will be readmitted to a hospital.
We then notice a disparity between the outcomes, and the true positive/negative, false positive/negative results for different genders, and especially races.
We then try to mitigate such unfair-biased predictions using different algorithms.
We first use preprocessing techniques, such as reweighting and resampling.
We then use in processing techniques, such as fairness constraints(demographic parity and equalized odds), and an Adversial Debiasing Model.
Lastly, we use post processing techniques, such as a Threshold Optimizer.
We then compare the resuts, and the different trade offs they induce between accuracy and fairness.

The rest of the directories contain different attempts to find bias.
The greek directory contains the training of an nlp model(word2vec) on a corpus on classical Greek literature.
