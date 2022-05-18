# Credit-Risk-Analysis

## 1. Overview of the Analysis

### Background Information
Traditional banking relies on credit scores, income, and collateral assets to assess the risk of a borrower. However, the rise of FinTech (Financial Technology) has allowed lenders to use machine learning to assess risks to decide whether or not to approve a loan application.

FastLending, a lending services company, wants to use machine learning to predict credit risk. The company believes that it will provide a quicker and more reliable loan experience to its clients. Machine learning will provide more accurate identification of good borrower candidates, which will lead to lower default rates.  

### Purpose
The purpose of this project is to assist the company in implementing this plan. In this project, I will build and evaluate several machine learning models to predict credit risk. I will use techniques such as resampling and boosting to make the most of our models and data. Once we designed and implemented these algorithms, we will evaluate their performance and see how well our models predict data

### Dataset 
- The company provided us with their loan dataset in a CSV format located [here](https://github.com/Aigerim-Zh/Credit-Risk-Analysis/blob/main/Resources/LoanStats_2019Q1.csv).
- In the original dataset, there are 68,817 observations corresponding to 86 variables. 
- The target variable we need to predict is a dummy variable, ```loan_status```, for high- and low-risk applications. 
- The feature variables include many variables that the company records for an applicant such as, but not limited to, loan amount, annual income, homeownership, verification status, hardship flag, etc..

### Resources Used
- Python 3.7 machine learning environment. 
- ```Scikit-learn``` and ```imbalanced-learn``` libraries.

## 2. Machine Learning Implementation Results
Credit risk data has an obvious class imbalance as there are many more good applications than risky ones. Machine learning, when dealing with unbalanced data, will produce results biased toward the majority class. That is why I will compare how **six different machine learning models** work and compare their strengths and weaknesses.

**The first four machine learning models are focused on resampling data**. The code is available [here](https://github.com/Aigerim-Zh/Credit-Risk-Analysis/blob/main/credit_risk_resampling.ipynb). 
1. The features and target sets are defined. 
2. String variables are converted to numeric ones using the ```get_dummies()``` method. 
3. The data is split into training and testing sets (75% and 25%, respectively).
4. The data is resampled. 
5. ```LogisticRegression()``` binary classifier is applied to make predictions.
6.  The model's performance is evaluated using appropriate metrics. 

```imbalanced-learn``` and ```scikit-learn``` libraries were used to build and evaluate models using resampling. 

### Deliverable 1
I will oversample data using the Naive Random Oversampling ```RandomOverSampler``` and ```SMOTE``` algorithms. 

**Model 1. Naive Random Oversampling**
In random oversampling, instances of the minority class are randomly selected and added to the training set until the majority and minority classes are balanced.  

* **Balanced Accuracy Score**: 0.65
* **Precision Score**: 
    * High Risk: 0.01
    * Low Risk: 1.00
* **Recall Score**:
    * High Risk: 0.62
    * Low Risk: 0.68

![](https://github.com/Aigerim-Zh/Credit-Risk-Analysis/blob/main/Results/Naive_Random_Oversampling_Results.png)

**Model 2. SMOTE Oversampling**
In SMOTE, like naive random oversampling, the size of the minority is increased. The key difference with random oversampling is how the minority class is increased in size. 

However, un SMOTE, new instances are interpolated. That is, for an instance from the minority class, new values are generated based on its distance from its neighbors.

**Random oversampling draws from existing observations, whereas SMOTE generates synthetic observations**.

* **Balanced Accuracy Score**: 0.63
* **Precision Score**: 
    * High Risk: 0.01
    * Low Risk: 1.00
* **Recall Score**:
    * High Risk: 0.59
    * Low Risk: 0.67

![](https://github.com/Aigerim-Zh/Credit-Risk-Analysis/blob/main/Results/SMOTE_Results.png)

**Model 3. ClusterCentroids Undersampling**
Akin to SMOTE, the algorithm identifies clusters of the majority class, then generates synthetic data points, called centroids, that are representative of the clusters. The majority class is then undersampled down to the size of the minority class. 

* **Balanced Accuracy Score**: 0.52
* **Precision Score**: 
    * High Risk: 0.01
    * Low Risk: 1.00
* **Recall Score**:
    * High Risk: 0.60
    * Low Risk: 0.44

![](https://github.com/Aigerim-Zh/Credit-Risk-Analysis/blob/main/Results/ClusterCentroids_Results.png)


### Deliverable 2
**Model 4. SMOTEENN Combinational Approach of Over- and Undersampling** 
A downside of oversampling with SMOTE is its reliance on the immediate neighbors of a data point. Because the algorithm doesn't see the overall distribution of data, the new data it creates can be heavily influenced by outliers. 

A downside of undersampling is that it involves loss of data and is not an option when the dataset is small. 

**SMOTEEN** combines the SMOTE and Edited Nearest Neighbors (ENN) algorithms. It includes the following steps: 
1. Oversample the minority class with SMOTE.
2. Clean the resulting data with an undersampling strategy. If the two of the nearest neighbors of a data point belong to different classes, the data point is dropped.

* **Balanced Accuracy Score**: 0.61
* **Precision Score**: 
    * High Risk: 0.01
    * Low Risk: 1.00
* **Recall Score**:
    * High Risk: 0.67
    * Low Risk: 0.55

![](https://github.com/Aigerim-Zh/Credit-Risk-Analysis/blob/main/Results/SMOTEENN_Results.png)


### Deliverable 3
**The last two models are focused on ensemble learning techniques**. The code is available [here](https://github.com/Aigerim-Zh/Credit-Risk-Analysis/blob/main/credit_risk_ensemble.ipynb).

Ensemble learning builds on the idea that two is better than one. A single tree may be prone to errors, but many of them can be combined to form a stronger model. 

I will use two new machine learning models that reduce bias, ```BalancedRandomForestClassifier``` and ```EasyEnsembleClassifier```, to predict credit risk. 

**Model 5. Balanced Random Forest Classifier**

A random forest model combines many decision trees into a forest of trees. Random forest models:
- Are robust against overfitting because all of those weak learners are trained on different pieces of the data.
- Can be used to rank the importance of input variables in a natural way.
- Can handle thousands of input variables without variable deletion.
- Are robust to outliers and nonlinear data.
- Run efficiently on large datasets. 


* **Balanced Accuracy Score**: 0.79
* **Precision Score**: 
    * High Risk: 0.03
    * Low Risk: 1.00
* **Recall Score**:
    * High Risk: 0.71
    * Low Risk: 0.88

![](https://github.com/Aigerim-Zh/Credit-Risk-Analysis/blob/main/Results/Balanced_Random_Forest_Classifier.png)


**Model 6. Easy Ensemble AdaBoost Classifier**
In AdaBoost, a model is trained and then evaluated. After evaluating the errors of the first model, another model is trained. 

The model gives extra weight to the errors from the previous model. The purpose of this weighting is to minimize similar errors in subsequent models. Then, the errors from the second model are given extra weight for the third model. This process is repeated until the error rate is minimized.

* **Balanced Accuracy Score**: 0.93
* **Precision Score**: 
    * High Risk: 0.08
    * Low Risk: 1.00
* **Recall Score**:
    * High Risk: 0.91
    * Low Risk: 0.94

![](https://github.com/Aigerim-Zh/Credit-Risk-Analysis/blob/main/Results/Easy_Ensemble_AdaBoost_Classifier_Results.png)

## 3. Summary
- **Accuracy Score** is the percentage of predictions the model predicts right. Looking at the accuracy score alone is not enough as it does not account for unbalanced data. 
- **Balanced Accuracy Score** is necessary to account for class imbalance.  
- **Precision Score** measures how reliable a positive classification is: ```TP/(TP+FP)```
- **Recall Score** is the ability of the classifier to find all positive samples: ```TP/(TP+FN)```

Based on these metrics, **Model 6, Easy Ensemble AdaBoost Classifier, showed the strongest performance**. It would be my recommendation for the following reasons:
- Model 6 has the largest balanced accuracy score (0.93) compared to all other tested models.  
- The precision score for low-risk applications is 1.00 in all models. 
- The precision score for high-risk applications is the highest for Model 6 (0.08), indicating the lowest number of false positives. 
- Model 6 has the highest recall score for both high- and low-risk applications (0.91 and 0.94, respectively). 