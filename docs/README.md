# DEFAULTER PREDICTION SYSTEM FOR ANALYZING RISKS FOR BANKING AGENCIES

## Project Description

Loan default occurs when the borrower fails to repay as per the initial agreement. The borrower is called a defaulter in this scenario. We have a dataset that holds the details about loan applications. This dataset will be cleaned and analyzed to determine if the current applicant will be a good candidate to receive a loan or not.

The reason why the bank analyzes or tries to predict whether the current applicant is a good applicant is to reduce the risk of loss and increase the profits. If the bank rejects an applicant that is capable of repaying the money, it will be a loss of business. Hence, we will be developing a system that helps banks identify whether the bank should approve the loan or not for any particular applicant. The system will find the patterns in the data to determine whether a person will be capable or repaying the borrowed money or not.

## Data Description

The dataset “Loan Defaulter” from Kaggle contains the complete loan data for real business scenarios in banking and financial services. The features include the type of loan (cash or revolving), credit amount of loan, loan annuity, details of user (gender, income, employment details, age, contact details, etc.), down payment details, rate of interest, contract status and many more. The dataset contains a total of 307,501 observations with 122 distinct attributes. We are planning to use around 50,000 to 100,000 records for the training and testing purposes of the model. These figures may tend to change as we move ahead in our project. Using this data, we will be trying to predict if a person is able to repay a loan or not.

## Data Cleansing & Sampling

We analyzed all the attributes to choose the most important features. We picked the following 26 features:SK_ID_CURR,TARGET,NAME_CONTRACT_TYPE,CODE_GENDER, FLAG_OWN_CAR,FLAG_OWN_REALTY,CNT_CHILDREN,AMT_INCOME_TOTAL, AMT_CREDIT,AMT_ANNUITY,AMT_GOODS_PRICE,NAME_INCOME_TYPE, NAME_EDUCATION_TYPE,NAME_FAMILY_STATUS,NAME_HOUSING_TYPE, DAYS_EMPLOYED, DAYS_ID_PUBLISH, OCCUPATION_TYPE, CNT_FAM_MEMBERS, REGION_RATING_CLIENT, REGION_RATING_CLIENT_W_CITY, ORGANIZATION_TYPE, OBS_30_CNT_SOCIAL_CIRCLE,DEF_30_CNT_SOCIAL_CIRCLE, OBS_60_CNT_SOCIAL_CIRCLE,DEF_60_CNT_SOCIAL_CIRCLE.

Moreover, we also identified TARGET as the response variable. TARGET is a binary attribute that only accepts 0 and 1 as values. 0 means that the client received a loan and 1, that he/she did not.
 
After choosing the features, we made a few changes. We transformed all the negative values in the DAYS_EMPLOYED and DAYS_ID_PUBLISH attributes by taking their absolute value. Moreover, the OCCUPATION_TYPE attribute had numerous empty cells. We filled those empty cells with the value “Missing.” After making these changes, we extracted a random sample of 100,000 rows for classification.

## Data Analysis & Visualization

**Target Class Distribution**

From the figure shown below, we can see that the class 0 and class 1 are unbalanced as observed in the real world scenario. Class 0 contains around 90% of the records whereas class 1 contains remaining 10% records.

![Fig1](https://github.com/AnkitaKhadsare95/Data-Science/blob/main/Bank%20Defaulter%20Prediction%20System/Images/Fig1.png?raw=true)

**Analysis of quantitative predictors**

From the analysis of the figures shown below, we understand the type of data, the various categories it consists of and the frequency of the values. The quantitative predictors such as, name_contract_type, tells us the type of loan, whether a cash loan or a revolving loan, code_gender tells us the number of male versus female applicants. Flag_own_car tells us the number of applicants that own a car and those that do not own a car, similarly for all the other qualitative data, we observe the categories and the frequency of the values. 

![Fig2](https://github.com/AnkitaKhadsare95/Data-Science/blob/main/Bank%20Defaulter%20Prediction%20System/Images/Fig2.png?raw=true)

![Fig3](https://github.com/AnkitaKhadsare95/Data-Science/blob/main/Bank%20Defaulter%20Prediction%20System/Images/Fig3.png?raw=true)

![Fig4](https://github.com/AnkitaKhadsare95/Data-Science/blob/main/Bank%20Defaulter%20Prediction%20System/Images/Fig4.png?raw=true)

**Correlation Matrix**

We can analyze from the figure shown below that, name_family_status is strongly correlated (inversely correlated) with cnt_family_members, obs_30_cnt_social_circle is strongly correlated with obs_60_cnt_social_circle. Amt_credit is strongly correlated with amt_goods_price and mildly correlated with amt_annuity. Thus, these attributes may be important in further process.

![Fig5](https://github.com/AnkitaKhadsare95/Data-Science/blob/main/Bank%20Defaulter%20Prediction%20System/Images/Fig5.png?raw=true)

**Amt_credit versus the Name_contract_type**

As seen in the figure below, we are comparing the credit amount of the loan with the type of loan. It can be inferred that, the average cash loans is approximately $500000, and revolving loans is roughly $250000, that is half of the cash loans amount.

![Fig6](https://github.com/AnkitaKhadsare95/Data-Science/blob/main/Bank%20Defaulter%20Prediction%20System/Images/Fig6.png?raw=true)

**Amt_income_total versus Name_income_type**

It is observed from this figure that, the average income is approximately, slightly less than $200,000 for a commercial associate, almost $100,000 for those on Maternity leave, slightly greater that $100,000 for a pensioner, $150,000 for a State Servant, slightly less than $200,000 for a student, roughly $150,000 for unemployed and similarly approximately $150,000 for working.

![Fig7](https://github.com/AnkitaKhadsare95/Data-Science/blob/main/Bank%20Defaulter%20Prediction%20System/Images/Fig7.png?raw=true)

**Histograms of some attributes**

Below are the histograms of some potential attributes. Here, we are visualizing the numeric attributes like the credit amount of the loan, loan annuity, and price of goods for which loan is sanctioned. Through this we try to understand the distribution of the data. Also, we summarize each of these attributes for a detailed understanding of the attributes.

![Fig8](https://github.com/AnkitaKhadsare95/Data-Science/blob/main/Bank%20Defaulter%20Prediction%20System/Images/Fig8.png?raw=true)

![Fig9](https://github.com/AnkitaKhadsare95/Data-Science/blob/main/Bank%20Defaulter%20Prediction%20System/Images/Fig9.png?raw=true)

![Fig10](https://github.com/AnkitaKhadsare95/Data-Science/blob/main/Bank%20Defaulter%20Prediction%20System/Images/Fig10.png?raw=true)

![Fig11](https://github.com/AnkitaKhadsare95/Data-Science/blob/main/Bank%20Defaulter%20Prediction%20System/Images/Fig11.png?raw=true)

![Fig12](https://github.com/AnkitaKhadsare95/Data-Science/blob/main/Bank%20Defaulter%20Prediction%20System/Images/Fig12.png?raw=true)

![Fig13](https://github.com/AnkitaKhadsare95/Data-Science/blob/main/Bank%20Defaulter%20Prediction%20System/Images/Fig13.png?raw=true)

# Feature selection

Feature selection is an important part of the process. All the features that were irrelevant to the response or had over 50% of the blank data were eliminated. Few of the columns had 100% data but the level distribution was unbalanced. It was clear that such features had a default value and very few applications had received any value for it. Such features could throw off our model, hence they were eliminated. From the rest of the features, the ones with good correlation coefficients were retained. To balance out the distribution of the data, for few of the classification algorithms Feature Scaling was applied. Feature scaling helps in standardizing the highly varying magnitude of the values of the independent features. Feature selection in our implementation did not improve the performance/accuracy drastically, but minor improvement was seen in Naive bayes and KNN classifications.

# Classification Algorithms

**a.	Decision Trees**

The decision tree is a supervised machine learning algorithm that works with categorical as well as numerical data. It is a better way to solve the supervised classification problem in machine learning. It is made up of nodes, leaves, and branches. A Node represents the attribute/feature and a branch represents the decision rule that divides the data. The leaves are the prediction of particular classes. The decision tree divides the data into different parts (can be 2 or more) at each level. This division is decided using the most informative category or attribute at the top level and becomes less informative as the level progresses.

i.	Advantages

Decision Tree is easy to interpret and provide human-friendly visualization. It is fast to implement and training takes less time. DTs provide inbuilt feature selection. They can handle both numerical and categorical data. Decision tree algorithms are not affected by highly correlated data.

ii.	Disadvantages

Decision Tree could overfit the data if the records are not sufficient for each category. They are highly prone to the variance of data. They could generate biased results if the data has an unbalanced class frequency

**b.	Logistic Regression**

Logistic regression is a binary classification method used when the response variable is categorical. It is used to describe data and to explain the relationship between one dependent binary variable and one or more nominal, ordinal, interval or ratio-level independent variables.

i.	Advantages

Logistic regression is designed for classification and is most useful for understanding the influence of several independent variables on a single outcome variable. It’s also simple and easy to implement

ii.	Disadvantages

It works only when the predicted variable is binary and assumes that all predictors are independent of each other and that data is free of missing values. Moreover, it does not handle a large number of categorical variables well and requires transformation of
non-linear features.
 
**c.	K-Nearest Neighbors**

K-Nearest Neighbor is a supervised learning algorithm to predict the label of the target variable. It doesn’t make any assumptions about the target data, it rather choses the label for it according to the labels of k neighbouring data points. The K neighbouring points are chosen according to their distance from the target variable. It can choose from different distance calculation methodologies like Euclidean, Hamming, Manhattan and Minkowski distance.

i.	Advantages

Major advantage of KNN is that it is simple to implement and doesn’t need and set up for training. As the algorithm is used, it learns and constantly keeps evolving. Unlike other algorithms, KNN handles multiclass problems pretty easily. It handles changes for multiclass problems without a lot of modifications. KNN can be used for both Classification and regression.

ii.	Disadvantages

It is very sensitive to changes in the data. The noise, outliers and missing values in the data can throw off the algorithms prediction accuracy. KNN doesn’t necessarily work well with high dimensional data. As the dimensions of the dataset increases, the complexity of the distance calculation between the points also increases. Choosing an optimal K value and speed are two most critical problems with KNN. If the K value is chosen too small or too large, it will lead to false predictions and the algorithms can slow down pretty quickly with increasing dataset size.

**d.	Naive Bayes**
Naive Bayes is a classification technique based on Bayes’ Theorem that assumes the predictors are independent. Therefore, the presence of a particular feature in a class is unrelated to the presence of any other feature. A Naive Bayes model is easy to build and useful for very large data sets. There are three types of Naive Bayes models: Gaussian, Multinomial, and Bernoulli. Gaussian assumes that features follow a normal distribution. Multinomial is used for discrete counts, and Bernoulli is a binomial model useful if the features are binary.

**i.	Advantages**

It is easy and fast to predict the class of the test data set. It also performs well in multi-class prediction. When the assumption of independence holds, a Naive Bayes classifier performs better compared to other models like logistic regression and you need less training data. Moreover, it performs well on categorical input variables compared to numerical variables. For numerical variables, the normal distribution is assumed.

ii.	Disadvantages

If a categorical variable has a category in the test set, which was not observed in the training set, then the model will assign a zero probability and will be unable to make a prediction. Naive Bayes can also be a bad estimator, so the probability outputs from predict_proba are not to be taken too seriously. Another limitation of Naive Bayes is the assumption of independent predictors. In real life, it is almost impossible that we get a set of predictors which are completely independent.

**e.	Support Vector Machines (SVMs)**

SVMs are a set of supervised learning methods used for classification, regression and outliers detection. In the SVM algorithm, each data item is plotted as a point in n-dimensional space (where n is number of features) with the value of each feature being the value of a particular coordinate. Then, classification is performed by finding the hyper-plane that differentiates the two classes very well.

i.	Advantages

SVMs have numerous advantages. They work well with a clear margin of separation, are effective in high dimensional spaces and when the number of dimensions is greater than the number of samples. It is also memory efficient because it uses a subset of the training points in the decision function and is versatile due to its several kernel functions.

ii.	Disadvantages

It doesn’t perform well when the data set is large because the required training time is higher. SVMs also do not directly provide probability estimates, which are calculated using an expensive five-fold cross-validation.

# Test & Evaluation

**a.	Decision Trees**

While implementing the decision tree model, we have decided to make use of randomly picked 70% of the data for training the model and kept remaining 30% data for testing purposes. We have made use of gini for the impurity measure of the decision tree. Below are the results obtained from the decision tree model.
 
**Training Confusion Matrix**
| | Predicted 0  | Predicted 1 |
|-------------| ------------- | ------------- |
| **Actual 0** | 64307  | 0  |
| **Actual 1** | 0 |5693  |

Accuracy: 100%

**Testing Confusion Matrix**
| | Predicted 0  | Predicted 1 |
|-------------| ------------- | ------------- |
| **Actual 0** | 24964  | 2550  |
| **Actual 1** | 2171 | 315  |

Accuracy: 84.21%

![Fig14](https://github.com/AnkitaKhadsare95/Data-Science/blob/main/Bank%20Defaulter%20Prediction%20System/Images/Fig14.png?raw=true)

**b.	K-Nearest Neighbors**

The algorithm was run using 70% of the total data to train the model and 30% was used to test it. Error rates were calculated between k=1 and k=20. Below is the plot of K values against the error rate. We see a significant drop in the error rate at k=4, after which it approximately remains the same. Hence k value was chosen to be 4.
 
![Fig15](https://github.com/AnkitaKhadsare95/Data-Science/blob/main/Bank%20Defaulter%20Prediction%20System/Images/Fig15.png?raw=true)
 
We also calculated the confusion matrix and accuracy score for Training and testing data.

**Training Confusion Matrix**
| | Predicted 0  | Predicted 1 |
|-------------| ------------- | ------------- |
| **Actual 0** | 64244  | 63  |
| **Actual 1** | 5522 | 171  |

Accuracy: 92.02%

**Testing Confusion Matrix**
| | Predicted 0  | Predicted 1 |
|-------------| ------------- | ------------- |
| **Actual 0** | 27431  | 83 |
| **Actual 1** | 2477 | 9  |

Accuracy: 91.47%

The testing and training sets have similar accuracies.

**Classification Report**
| | Precision |	Recall | F1-score |	Support |
|-------------|-------------|------------|-----------|-----------|
| **0 (Gets loan)**	| 0.93 | 1.00	| 0.96 | 27514 |
| **1 (No loan)**	| 0.10 | 0.00 |	0.01 | 2486 |
| **Accuracy**	|	| |	0.91 | 30000 |
| **Macro avg** |	0.51 | 0.50 |	0.48 | 30000 |
| **Weighted avg** | 0.85 |	0.91 | 0.88	| 30000 |

**c.	Logistic Regression**

**Training Confusion Matrix**
| | Predicted 0  | Predicted 1 |
|-------------| ------------- | ------------- |
| **Actual 0** | 64370  | 0  |
| **Actual 1** | 5630 | 0  |

Accuracy: 91.96%

**Testing Confusion Matrix**
| | Predicted 0  | Predicted 1 |
|-------------| ------------- | ------------- |
| **Actual 0** | 27507  | 0 |
| **Actual 1** | 2493 | 0  |

Accuracy: 91.69%

The testing and training sets have similar accuracies. Logistic Regression and SVMs provided the same results and had the highest accuracy.

**Classification Report**
| | Precision |	Recall | F1-score |	Support |
|-------------|-------------|------------|-----------|-----------|
| **0 (Gets loan)**	| 0.92 | 1.00	| 0.96 | 27507 |
| **1 (No loan)**	| 0.00 | 0.00 |	0.00 | 2493 |
| **Accuracy**	|	| |	0.92 | 30000 |
| **Macro avg** |	0.46 | 0.50 |	0.48 | 30000 |
| **Weighted avg** | 0.84 |	0.92 | 0.88	| 30000 |

**d.	Naive Bayes**

We used the Gaussian model. We split the data into training and test sets. 70% of the data went into the training set and 30% was used for the test set. We ran the model and then calculated the confusion matrix and accuracy score of both the training and test sets.

**Training Confusion Matrix**
| | Predicted 0  | Predicted 1 |
|-------------| ------------- | ------------- |
| **Actual 0** | 52243  | 12127  |
| **Actual 1** | 3697 | 1933  |

Accuracy: 77.39%

**Testing Confusion Matrix**
| | Predicted 0  | Predicted 1 |
|-------------| ------------- | ------------- |
| **Actual 0** | 22280  | 5227 |
| **Actual 1** | 1682 | 811  |

Accuracy: 76.97%

The testing and training sets have similar accuracies.

**Classification Report**
| | Precision |	Recall | F1-score |	Support |
|-------------|-------------|------------|-----------|-----------|
| **0 (Gets loan)**	| 0.93 | 0.81	| 0.87 | 27507 |
| **1 (No loan)**	| 0.13 | 0.33 |	0.19 | 2493 |
| **Accuracy**	|	| |	0.77 | 30000 |
| **Macro avg** |	0.53 | 0.57 |	0.53 | 30000 |
| **Weighted avg** | 0.86 |	0.77 | 0.81	| 30000 |

We also created a plot to show the relationship between different the client’s income and the amount of loan he/she received. We see that people with higher incomes are more likely to receive a bigger loan.

![Fig16](https://github.com/AnkitaKhadsare95/Data-Science/blob/main/Bank%20Defaulter%20Prediction%20System/Images/Fig16.png?raw=true)

**e.	Support Vector Machines**

We ran the same operations as above for the other algorithm. The only specificity is that we chose rbf as the kernel function because it gave better results than the others.
 
**Training Confusion Matrix**
| | Predicted 0  | Predicted 1 |
|-------------| ------------- | ------------- |
| **Actual 0** | 64370  | 0  |
| **Actual 1** | 5630 | 0  |

Accuracy: 91.96%

**Testing Confusion Matrix**
| | Predicted 0  | Predicted 1 |
|-------------| ------------- | ------------- |
| **Actual 0** | 27507  | 0 |
| **Actual 1** | 2493 | 0  |

Accuracy: 91.69%

The testing and training sets have similar accuracies. SVMs had the highest accuracy among the classification algorithms.

**Classification Report**
| | Precision |	Recall | F1-score |	Support |
|-------------|-------------|------------|-----------|-----------|
| **0 (Gets loan)**	| 0.92 | 1.00	| 0.96 | 27507 |
| **1 (No loan)**	| 0.00 | 0.00 |	0.00 | 2493 |
| **Accuracy**	|	| |	0.92 | 30000 |
| **Macro avg** |	0.46 | 0.50 |	0.48 | 30000 |
| **Weighted avg** | 0.84 |	0.92 | 0.88	| 30000 |

We also created a plot to show the relationship between different the client’s income and the amount of loan he/she received. We see that people with higher incomes are more likely to receive a bigger loan.
 
**8.	Core Algorithm & Fine Tuning**

![Fig17](https://github.com/AnkitaKhadsare95/Data-Science/blob/main/Bank%20Defaulter%20Prediction%20System/Images/Fig17.png?raw=true)

Initially, as we can see from the above graph, the target class distribution was not balanced. The majority class (class 0) was having more than 250K records and minority class (class 1) was having less than 25k records. But, training the models on such highly unbalanced data might result in highly biased models towards the majority class. Hence, to avoid such bias we have tried to balance out the data by considering all records from the minority class and randomly selecting the ~24k records from the majority class. The obtained balanced data can be observed in the figure shown above. This balanced data contains total ~50k records with equal distribution of both the classes.

Before applying the models, we randomly selected a sample of 70% of the data for training purposes and kept the rest 30% of the data for testing. Hence, we have selected around 35K records for training and 15K records for testing.

Upon applying the decision tree model on the balanced dataset, we obtained the following results.

**Testing Confusion Matrix**
| | Predicted (Non-Defaulter) 0 | Predicted (Defaulter) 1 |
|-------------| ------------- | ------------- |
| **Actual (Non-Defaulter) 0** | 2645  | 2116 |
| **Actual (Defaulter) 1** | 2150 | 2657  |

Accuracy: 55.41 %

Predicting that the user will not be a defaulter even though there is a high chance of him being a defaulter in real life is not a good practice as it may result in high revenue loss for the loan lending companies. Hence, along with accuracy, we should consider other evaluation metrics too while analyzing the performance of the model.

By looking at the results of both models, we can clearly observe that the model trained on the unbalanced data produces the higher accuracy as compared to the model trained on the balanced dataset. However, if we take a closer look at the results, we can notice that the model trained on the unbalanced dataset is highly biased towards predicting the majority class. The probability of predicting class 0 out of all actual class 0 is 91.73 % but the probability of predicting class 1 out of all actual class 1 is only 12.67%. Hence, this model is not very good for banking applications. This model is favoring false positives (considering non-defaulters as a positive class), i.e., it is predicting defaulters are non-defaulters more often. Hence it is not a good model.

On the other hand, the model trained on the balanced dataset produces more unbiased results. Although the accuracy of this model is comparatively low (55.41 %), the probability of predicting class 0 when it is actually class 0 is 55.55% and the probability of predicting class 1 when it is actually class 1 is 55.27 %. This model is maintaining a good balance between false positives and false negatives (considering non-defaulters as a positive class), i.e., it is not frequently predicting defaulters as non-defaulters and vice-versa. Hence it is a better model than a model trained on unbalanced data.

**Choosing the best value for min_samples_split**

The parameter min_samples_splits is taken as the minimum number of samples required to split an internal node. We tried to fine tune the model by calculating the accuracy of the model with different values for min_sample_split. Based on this, we get the best accuracy when min_samples_split = 0.05. The graph below shows the accuracy of the model on changing the value of min_samples_split from 0.01 to 1 with an increment of 0.05 for each iteration.
 
![Fig18](https://github.com/AnkitaKhadsare95/Data-Science/blob/main/Bank%20Defaulter%20Prediction%20System/Images/Fig18.png?raw=true)

The Test Confusion Matrix after choosing the best min_samples_values=0.5 can be seen below.

**Testing Confusion Matrix**
| | Predicted (Non-Defaulter) 0 | Predicted (Defaulter) 1 |
|-------------| ------------- | ------------- |
| **Actual (Non-Defaulter) 0** | 2859  | 1902 |
| **Actual (Defaulter) 1** | 1943 | 2864  |

Accuracy: 59.81 %

![Fig19](https://github.com/AnkitaKhadsare95/Data-Science/blob/main/Bank%20Defaulter%20Prediction%20System/Images/Fig19.png?raw=true)

**Classification Report**
| | Precision |	Recall | F1-score |	Support |
|-------------|-------------|------------|-----------|-----------|
| **0 (Non-Defaulter)**	| 0.60 | 0.60	| 0.60 | 4761 |
| **1 (Defaulter)**	| 0.60 | 0.60 |	0.60 | 4807 |
| **Accuracy**	|	| |	0.60 | 9568 |
| **Macro avg** |	0.60 | 0.60 |	0.60 | 9568 |
| **Weighted avg** | 0.60 |	0.60 | 0.60	| 9568 |

Based on the confusion matrix and the classification report above, we can see that the performance of the model has increased from 55.41% to 59.81%. Also, it maintains a good balance between the F-1 score for each class, sensitivity (60.05%) and specificity (59.57%) considering 0 i.e., Non-Defaulters as positive class. Upon performing the 5-fold cross validation on the decision tree, we got the average accuracy of 60.27%.

**Cross Validation Plot**

![Fig20](https://github.com/AnkitaKhadsare95/Data-Science/blob/main/Bank%20Defaulter%20Prediction%20System/Images/Fig20.png?raw=true)
 
**Random Forest**

Random Forest is another supervised machine learning technique used for classification problems. It constructs a large number of decision trees using randomly selected data  samples from training dataset and gets the prediction from each of those trees. Finally, chooses the best solution which has the maximum votes. Random forest is commonly known to outperform the decision tree algorithm. Hence, we have tried to apply this model in the effort of further improving the results of our model. The table below shows the results obtained using Random forest classifier.

**Testing Confusion Matrix**
| | Predicted (Non-Defaulter) 0 | Predicted (Defaulter) 1 |
|-------------| ------------- | ------------- |
| **Actual (Non-Defaulter) 0** | 819  | 409 |
| **Actual (Defaulter) 1** | 429 | 735  |

Accuracy: 64.92 %

![Fig21](https://github.com/AnkitaKhadsare95/Data-Science/blob/main/Bank%20Defaulter%20Prediction%20System/Images/Fig21.png?raw=true)

**Classification Report**
| | Precision |	Recall | F1-score |	Support |
|-------------|-------------|------------|-----------|-----------|
| **0 (Non-Defaulter)**	| 0.66 | 0.67	| 0.66 | 1228 |
| **1 (Defaulter)**	| 0.64 | 0.63 |	0.64 | 1164 |
| **Accuracy**	|	| |	0.65 | 2392 |
| **Macro avg** |	0.65 | 0.65 |	0.65 | 2392 |
| **Weighted avg** | 0.65 |	0.65 | 0.65	| 2392 |

With random forest we improved the accuracy of our model to 64.92 %. We got the similar average accuracy (64.10%) upon performing a 5-fold cross validation on the random forest model. As per above results, our model’s sensitivity is 66.69% and specificity of 63.14% considering 0 i.e., Non-Defaulters as positive class.

**Cross Validation Plot**

![Fig22](https://github.com/AnkitaKhadsare95/Data-Science/blob/main/Bank%20Defaulter%20Prediction%20System/Images/Fig22.png?raw=true)

# Lessons Learned

| Model	| Sensitivity	| Specificity |	Accuracy |
|-----------------------|----------------------|----------------------|---------------------|
| **Logistic Regression (Unbalanced data)**	| 100.00 | 00.00 | 91.69 |
| **K-Nearest Neighbors (Unbalanced data)**	| 99.69	| 00.36	| 91.47 |
| **Naive Bayes (Unbalanced data)** |	80.99 |	32.53	| 76.97 |
| **Support Vector Machine (Unbalanced data)** | 100.00 |	00.00	| 91.69 |
| **Decision Tree (Unbalanced data)**	| 90.7 | 12.67 | 84.21 |
| **Decision Tree (Fine Tuning)**	| 60.05 |	59.57 |	59.81 |
| **Random Forest (Fine Tuning)** |	66.69	| 63.14 |	64.92 |

Naive Bayes was very fast to run, but it had the worst accuracy. KNN took a lot longer to produce results than the other algorithms, but gave the most accurate results. The Decision tree model gave fairly decent results in terms of accuracy. However, regarding other metrics such as sensitivity and specificity, the obtained results seem to be highly biased towards the majority class in all the models before fine tuning. To try and reduce this bias, we tried to train the model with balanced data where there are equal amounts of class 1 (Defaulters) and class 0 (Non Defaulters) responses in the training records. This approach helped us to improve the performance of our model by achieving a good balance between sensitivity and specificity. To further improve the performance, we tried fine tuning the decision tree model.
 
Also, we tried applying the Random Forest model as it is known to produce better results as compared to the decision tree model, which resulted in improved accuracies with a better balance between sensitivity and specificity.

# Conclusion

After analyzing the data set we performed feature selection to get rid of non-correlated features. Only 24 features which contributed towards the response were kept. The achieved results from all five algorithms have shown very different results. The computation time for Naive Bayes was the lowest, whereas, KNN and Support Vector Machine took the longest to get the results. The accuracy of the prediction for K-Nearest Neighbor, Support Vector Machine and Logistic Regression were above 90%. Naive Bayes classifier produced an accuracy of around 60%. But all these models were highly biased towards the majority class. They were producing the worst results for classifying if the person is a defaulter i.e., very low specificity. The accuracy achieved for the Decision Tree algorithm was approximately 60%. The decision tree algorithm was more reliable than other models to predict if the person is a defaulter or not as it was maintaining a good balance between sensitivity and specificity.

For the real world applications, even though some of these algorithms are faster, they wouldn’t give accurate results. There is still scope to improve the performance of the Decision Tree algorithm. For future, we may use different boosting, binning or bagging methods to fine tune the Decision tree. It is important to achieve good accuracy as mistakes in Loans/Banking can get very expensive. We must prevent monetary loss not just by predicting whether the person is capable of replaying the loan, but also by preventing the loss of business caused by falsely predicting that the applicant may be able to repay the loan even when he can not.
