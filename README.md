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

Removed null valued rows and performed SMOTE to balance the dataset for customer converts vs non customer converts. Loaded the datas set for modeling. The code can be accessed [here]().






![GitHub Logo](https://github.com/skotak2/Seq2Seq-Machine-Translation-Model-Kannada-to-English/blob/main/Images/Picture2.png)

* Now we use the cloud functions capability to deploy the code the code for the Flask API and access the weights and vocabulary dictionary from the storage. 

For creating the cloud function, browse for it on the GCP platform and use the options highlighted to below to create a function,


![GitHub Logo](https://github.com/skotak2/Seq2Seq-Machine-Translation-Model-Kannada-to-English/blob/main/Images/Picture3.png)

![GitHub Logo](https://github.com/skotak2/Seq2Seq-Machine-Translation-Model-Kannada-to-English/blob/main/Images/Picture4.png)

![GitHub Logo](https://github.com/skotak2/Seq2Seq-Machine-Translation-Model-Kannada-to-English/blob/main/Images/Picture5.png)

*Allocation of 1 GiB memory is recommended. Once set, click on ‘Next’ and deploy the code on the cloud function console. 

To deploy the code, first configure the console with the below highlighted settings and prepare the environment using the requirements file (this is equivalent to pip install {library}) as described below, 

![GitHub Logo](https://github.com/skotak2/Seq2Seq-Machine-Translation-Model-Kannada-to-English/blob/main/Images/Picture6.png)

* Once Requirement is set with the above libraries, prepare the main.py script for deployment. The script has the api_request(x) function defined for returning the desired output given the input – ‘x’, from an external source. The code is uploaded above with name “main.py”. Once the code is arranged click on deploy.

![GitHub Logo](https://github.com/skotak2/Seq2Seq-Machine-Translation-Model-Kannada-to-English/blob/main/Images/Picture7.png)

![GitHub Logo](https://github.com/skotak2/Seq2Seq-Machine-Translation-Model-Kannada-to-English/blob/main/Images/Picture8.png)

* Once deployment is complete, click on the cloud function, using TESTING option to debug for deployment errors. Once the input is passed in the below format, test the function, and look for the desired output.

![GitHub Logo](https://github.com/skotak2/Seq2Seq-Machine-Translation-Model-Kannada-to-English/blob/main/Images/Picture9.png)

![GitHub Logo](https://github.com/skotak2/Seq2Seq-Machine-Translation-Model-Kannada-to-English/blob/main/Images/Picture10.png)


## RESULTS
The deployed model can be accessed from the url from any system to translate kannada sentences to english. 

## REFERENCES
* https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
* https://pytorch.org/tutorials/beginner/saving_loading_models.html


