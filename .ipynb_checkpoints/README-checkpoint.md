# Supervised Learning Project - Diabetes Diagnosis

## Brief Overview
Using data from the National Institute of Diabetes and Digestive and Kidney Diseases, the objective of this project is to predict whether a patient has diabetes based on provided diagnostic measurements.

Given that our dataset is labelled – a patient's diabetes status is labelled either '1' or '0' in our 'Outcome' column – we aim to build a supervised learning model to make these predictions.

## Initial Insights
An initial pairplot of all variables, with the data colored by Outcome, shows some significant stratification (particularly for Glucose, where higher levels are clearly correlated with a positive diagnosis). Even the clearest cases, however, are not absolute.

A heatmap intended to show the correlation between variables suggests a reasonably strong correlation between Glucose and Insulin. It further indicates a correlation between Glucose and Outcome. This suggests that there's a correlation between Insulin and Outcome, which is also borne out by the heatmap.

## Preprocessing & Feature Engineering Rationale

### Zero Values
I'm inclined to omit the pregnancies category out of hand, as a relationship between the two factors would almost certainly have an extraneous cause. Nonetheless, the data is included for the sake of completeness. Especially owing to the fact that 0 is a valid and repeated value in our data set, the significant range in number of pregnancies makes analysis difficult without removing the upper end of outliers so they are removed.

Blood Glucose levels below 70mg/dL are indicative of Hypoglycemia, which can be life threatening. Blood Glucose levels of 0 would be an immediate, life-threatening emergency. We can reasonably assume none of our patients have 0 Blood Glucose. 

Blood Pressure, Skin Thickness, Insulin, and BMI are also not capable of being 0, so we must exclude the 0 values from these columns.

Diabetes Pedigree Function is not clearly defined but contains no 0 values.

### Outliers

Pregnancies are addressed above – 0 values are valid, high outliers are trimmed given their proclivity to skew results and the serious unlikelihood that a positive correlation would have medical merit.

Our Q1 value for Blood Glucose is 99 mg/dL and our Q3 value is 140.25 mg/dL. Our IQR is 41.25, making our outlier boundaries 37.125 mg/dL and 202.125 mg/dL. Only our 0 values fall outside of this range, so in the end we haven't excluded any data.
On the other hand, we should consider being more restrictive – the rough boundaries for Hypoglycemia and Hyperglycemia are 70mg/dL and 180mg/dL, and we have patients who meet the criteria for these conditions but fall within our outlier boundaries. These may be medical outliers, but they may also be inaccurate – our patient with the 44mg/dL Blood Glucose level would have likely already been in life-threatening condition when her blood was taken.

Blood Pressure is difficult to interpret because it's typically measured using two figures, one describing systolic and one describing diastolic blood pressure. The single value here doesnt appear to reasonably describe either one, though the source cited by the providers of the dataset claims it describes diastolic blood pressure.
After removing our 0 values we're left with a min of 24mmHg and a max of 122mmHg, both of which are likely to represent severe and immediate risk to life.
Based on distribution alone and using IQR to calculate, our outlier boundaries are 35mmHg and 107mmHg, both of which carry risk but are legitimate values for diastolic blood pressure.
We ought remove high and low outliers here.

Skin Thickness measurement is absent in roughly 30% of patients. Consideration was given to eliminating this column. Any data imputation method would have to be performed on such a significant portion of the sample as to render the analysis likely useless anyways.

Regarding insulin, it is disappointing that so many values are missing but outliers should likely not be touched given the extreme variability that individuals exhibit.

BMI values appear broadly accurate, outliers likely ought not to have been replaced.

Diabetes Pedigree Function may or may not be effective as a real predictor (it was described in this paper in 1988 https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2245318/?page=2, where it was described as "not yet validated"). Regardless, these figures should likely be left as is.

Our minimum Age is 21 – consistent with the data. Outliers needn't be removed.

## Model Training
Given the number of possible machine learning models, our recent introduction to pipelines and ensemble models, and the fact that I'm a beginner to the development of models, I'm not sure how to best select a model.

We have been repeatedly given the advice that there's no hard and fast way to distinguish that problem x should be met with model y – only that as we build and deploy models, we'll become more familiar with which respond well to various data structures and nuances.

My aim here was to create several different options in terms of model pipelines, effectively tune the hyperparameters of each, and display a comparison of the results.

I'll briefly outline my rationale in selecting and combining various components:

Firstly, each model begins by scaling the data – because our variables are measured in different units, their relative absolute sizes wouldn't be likely to represent meaningful relationships. Data with a larger associated numeric value would be overweighted in the result.

Secondly, I incorporated PCA in an attempt to find hidden correlations between variables which could be later considered as their own variables – like feature engineering, but incorporating machine learning, was my thought. I no longer believe this is an especially effective way to address this problem.

Thirdly, I incorporated Logistic Regression because I understood it to be an effective method of predicting a binary result – in this case, the 'Outcome' variable is a binary diagnosis.

I reasoned that combining these methods would normalize my data, attempt to find variables even more effective at predicting the outcome than the existing variables, and then compare/combine the preexisting with the newly created variables in predicting an outcome.

A similar rationale held for the second pipeline, but I considered that maybe I was missing a step initially. By running PCA and SelectKBest side by side, now I would be generating new variables and assessing existing variables separately, and then using a classifier to compare their efficacy.

The second approach felt more reasonable, so I tried the same feature union of PCA and SelectKBest with two different classifiers – a Ridge Classifier and Logistic Regression.

Eventually, I elected to leave all models in my submission to show the attempts I had made. Given my level of experience, I was concerned that presenting a single model would increase my chances of handing in garbage while deleting something I wasn't aware was useful.

## Conclusion
The pipeline featuring a StandardScaler, followed by PCA with 6 components, followed by a RidgeClassifier with alpha 0.1, ended up performing best following recommended updates to data cleaning processes.

This model provided a test set accuracy of roughly 0.76, an improvement of roughly 3-5% over my other models.

In future, I would begin by attempting to conceptualize a simpler model more directly suited to the task, optimize that model as extensively and creatively as possible, and later consider adding additional steps.