# Heart Failure Prediction
This project aims to predict whether a heart failure patient will die during their follow-up period

**Huggingface** : https://huggingface.co/spaces/ahmadluay/deploy

# File Explanation on Github
This repository consists of several files :

- **Folder deployment** = Contains files used for deployment to HuggingFace (contains models, python applications etc.)
- **Dataset.csv** = dataset used for this project
- **inference.csv** = Dataset used for model inference
- **Notebook.ipynb** = This file is the main notebook used to explore dataset and built model
- **inference.ipynb** = - Notebook used for testing inference. Inferencing is done on a separate notebook to prove that the model can run on a notebook that is clean of variables
- **Drop_Columns**.txt = list of the columns to be dropped
- **best_model.pkl** = Saved model to be loaded and used later for making predictions on new data without the need to retrain
- **url.txt** = Deployment URL to HuggingFace

# Brief Summary of Project
The flow of this project, first EDA (Exploratory Data Analysis) to find out the basic picture of the dataset. Second, cleaning and preprocessing of the dataset. Third, Built Classification Models using 7 algorithms Random Forest, AdaBoost Classifier. These algorithms are tested based on their baseline/default parameters, and then cross-validation will be applied to evaluate each model based on mean and also standard deviation. Next hyperparameter tuning is carried out using the selected algorithm. The Selected model is AdaBoost Classifier model has been improved with Hyperparameter Tuning using GridSearch.

# Project Conclusion
1. Based on Exploratory Data Analysis:
  - Number of deaths after the following days are different, where __Non-Death are 36% greater than Death__.
  - The number of male patients with heart failure is more than female patients. Where about 32% die during the follow-up period.
  - Male patients who have smoking habits have a higher chance of dying during follow up periods than any other conditions.

2. Based on Model Evaluation:
  - Based on the model evaluation, the best model is AdaBoost Classifier that has been improved with hyperparameter tuning using GridSearch. The model achieves high accuracy (87%) in correctly classifying whether the patient died during the follow-up period or not. For the "Dead" class (1.0), the recall is 0.69. This means the model identifies around 69% of the actual "Dead" cases. This suggests that the model is not as sensitive when it comes to correctly identifying cases that are truly positive for the "Dead" class. It misses a significant portion of actual "Dead" cases, which could indicate that the model has a higher rate of false negatives for the "Dead" class. 

3. 
  - The model seems to perform well in correctly identifying cases of the "Not Dead" class, but it has room for improvement in correctly identifying cases of the "Dead" class. Depending on the application and the consequences of false negatives or positives, further analysis and potential adjustments to the model could be considered to enhance its overall performance