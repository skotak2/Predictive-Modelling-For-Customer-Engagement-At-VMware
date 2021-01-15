# Predictive-Modelling-For-Customer-Engagement-At-VMware
Built a supervised multi-class predictive model to bucket customers based on the events and actions recorded during their interactions with the VMWare's customer engagement portals 

### TABLE OF CONTENTS
* [Objective](#objective)
* [Technologies](#technologies)
* [Algorithms](#algorithms)
* [Data](#data)
* [Implementation](#implementation)
* [Results](#results)

## OBJECTIVE 
# Improve Customer engagement with website portals of VM Ware

1. Suggest business on come up with a set of segment rules to identify top individuals for a digital asset and to target them with personalization on the website.
2. Have substantiated marketing and sales implications.
3. Fine tune predictive models with appropriate parameters to predict for customer segments

## TECHNOLOGIES
R programming - *SMOTE, LiblineaR, ggplot2, randomForest, RRF, gbm, xgboost*

## ALGORITHMS
* SMOTE
* Random Forest
* LASSO
* Ridge Regression
* XGBoost

## DATA

The dataset has 700+ variables and 50K+ records of customer interactions. The data set is confidential and could be bought from [Harvard Business Publication](https://hbsp.harvard.edu/product/IMB623-PDF-ENG). 

## DATA PRE PROCESSING

Removed null valued rows and performed SMOTE to balance the dataset for customer converts vs non customer converts. Loaded the datas set for modeling. The code can be accessed [here](https://github.com/skotak2/Predictive-Modelling-For-Customer-Engagement-At-VMware/blob/main/Code/Models.R).

## Random Forest Model - Variable importance 

Post exploratory analysis, we decide on important variables as predictors for our model. We use the mean decrease in Gini Index to pick on the important variables and reduce the number of dimensions in the feature set.

![GitHub Logo](https://github.com/skotak2/Predictive-Modelling-For-Customer-Engagement-At-VMware/blob/main/Images/RF_model.png)

Post running this model, we use the mean decrease in gini index to list the important variables. Below is the list of variables which are potential predictors for a model.

![GitHub Logo](https://github.com/skotak2/Predictive-Modelling-For-Customer-Engagement-At-VMware/blob/main/Images/significant_variables.png)

We built a Lasso regression model using the top 200 variables that came out significant from the Random Forest model. We performed Cross-validation to get the best cost paramater for the LASSO regression.

![GitHub Logo](https://github.com/skotak2/Predictive-Modelling-For-Customer-Engagement-At-VMware/blob/main/Images/LASSO_Model.png)

We also built XGBoost model using the top 200 variables from the Random Forest model. The XGBoost model outperformed the LASSO regression in terms of accuracy and recall by 6% and 4% . However the model lacks interpretability in understanding what veriables influence the marketing and sales of products on the digital portals.

![GitHub Logo](https://github.com/skotak2/Predictive-Modelling-For-Customer-Engagement-At-VMware/blob/main/Images/Xgboostresult.png)


## RESULTS

1. Interms of factors influencing the conversion of a vistor to customer, - "product page views, first data of download, top resources and pdf downloads" are top variables with high importance.

2. Vistors who view the product page more than average views for the page, should be priortized and personalization is required.

3. Vistors downloading more PDFs of various products are interested in understanding the details further, hence persuing them will add up for conversion rate.
