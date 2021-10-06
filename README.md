# Predicting eBay Delivery Times 
## Millie Mince, Meghna Lohia, Hannah Mandell, Nate Dailey, Indiana Huey

## Update 1
#### Software we will use and type of neural network
Thus far, we have trained a naive network that uses simple linear regression. Expectedly, this model does not perform well on our complex dataset. We want to explore different frameworks that may deal with the complexity of our dataset better:
- [XGBoost](https://xgboost.readthedocs.io/en/latest/): We read about XGBoost in one of our articles in our literature review and are interested in trying it out. XGBoost provides an implementation of gradient boosted decision trees that perform at a high level. What makes XGBoost unique is that instead of training a single model on a dataset, XGBoost combines many models together to perform the final one, training each model after the previous one and correcting errors as it goes.
- [SqueezeNet](https://arxiv.org/abs/1602.07360): We also read about different squeeze frameworks and wanted to explore squeezenet. Although it is traditionally used for image classification, we may be interested in exploring the location relationship for the zip codes in our data. We’d like to read more on CNN’s and how they work as well. 
- [Fastai.tabular](https://docs.fast.ai/tabular.learner.html): We are interested in trying out a tabular data focussed solution that works on creating embeddings for categorical variables. It allows for the user to specify the layers in the model and can be used for classification and regression. The fastai tabular learner is the same as a normal learner but it “implements a predict method specific to work on a row of data”

#### Data we will use 
eBay has provided a training dataset containing 15 million shipment records and a test dataset containing 2.5 million shipment records. Because these records are confidential, we cannot link the entire dataset here. However, the features for each shipment are outlined [here](https://github.com/milliemince/eBay-shipping-predictions/blob/main/eBay_ML_Challenge_Dataset_2021/Annexure_2021.pdf).
We have created visualizations to help us better understand the dataset. They can be found below (STILL NEED TO INSERT THESE!)

#### Shape and type of input/output
Our input is an *m x n* matrix where *n* = number of training instances = 15 million and *m* = number of features = 19.
The output of our network will be a single value: the prediction for the number of days the shipment will take.

#### To consider
 1. Some features are given as strings (i.e. package_size) that must be converted to discretely numbered values
 2. How should we deal with missing values? Many features, including weight, declared_handling_days, carrier_min_estimate, and carrier_max_estimate, have missing values. We could remove these instances entirely, but that is not ideal because the training instance is still valuable. We could replace missing data with averaged data from other, similar instances.
 3. How should we deal with outliers?
 4. How can we engineer zip codes (both from the buyer and the seller) to meaningul data? Should we create a new feature that represents the distance between both zipcodes?
 5. How can we incorporate seller_id and category_id? 


## Literature Review
#### Predicting Shipping Times with Machine Learning
Jonquais, A.C. (2019). Predicting Shipping Time with Machine Learning. Diplome d'Ingenieur, Logistique, Universite le Harve Normandie, dspace.mit.edu/bitstream/handle/1721.1/121280/Jonquais_Krempl_2019.pdf?sequence=1&isAllowed=y.

This project attempts to predict shipping times for the compnay A.P Moller-Maersk. Different models, including linear regression, random forest, and a neural network with 1 hidden layer of 50 neurons was used. Engineers trained different models for different segments of a given shipment: from booking date to receipt by shipment service, from receipt by shipment service to the departure from origin site, from the departure from origin site to arrival at point of destination, and finally from arrival at point of destination to the time the shipment was received by the buyer. 

#### End-to-End Prediction of Parcel Delivery Time with Deep Learning for Smart-City Applications
Araujo, A.C., & Etemad, A. (2020). End-to-End Prediction of Parcel Delivery Time with Deep Learning for Smart-City Applications. ArXiv, abs/2009.12197.

This paper documents how the authors went about solving the problem of last-mile parcel delivery time estimation using data from Canada Post, the main postal service in Canada. They formalize the problem as an Origin-Destination Travel Time Estimation problem and compare several neural networks in order to generate the best results. Their findings indicate that a ResnNet with 8 residual convolutional blocks has the best performance, but they also explore VGG and Multi-Layer Perceptron models.

#### DeepETA: A Spatial-Temporal Sequential Neural Network Model for Estimating Time of Arrival in Package Delivery System
Wu, F., & Wu, L. (2019). DeepETA: A Spatial-Temporal Sequential Neural Network Model for Estimating Time of Arrival in Package Delivery System. Proceedings of the AAAI Conference on Artificial Intelligence, 33(01), 774-781. doi.org/10.1609/aaai.v33i01.3301774.

This article covers the many difficulties of predicting package delivery time such as multiple destinations, time variant delivery status, and time invariant delivery features. It then describes how DeepETA, the proposed framework, is a spatial-temporal sequential neural network model that uses a latest route encoder to include the location of packages and frequent pattern encoder to include historical data. There are 3 different layers to this model and through experimenting on real logistics data, the authors show that their proposed method outperforms state of the art methods. 


#### How to Predict Shipments’ Time of Delivery with Cloud-based Machine Learning Models
Sancricca, M., & Basford, P. (2021, March 23). How to Predict Shipments’ Time of Delivery with Cloud-based Machine Learning Models. Amazon. /aws.amazon.com/blogs/industries/how-to-predict-shipments-time-of-delivery-with-cloud-based-machine-learning-models/.

The article discusses how the logistics supplier Aramex uses machine learning to predict delivery times, claiming that its model has increased the accuracy of delivery predictions by 74%. It describes a multi-leg approach in which the shipping process is divided into multiple steps to be estimated individually. For instance, the time from seller to processing hub is estimated separately from the time from customs to buyer in international transactions. The architecture integrates systems such as Amazon Redshift clustering and Amazon SageMaker to process its data.


#### Predicting Package Delivery Time For Motorcycles In Nairobi
Magiya, J. (2020). Predicting Package Delivery Time For Motorcycles In Nairobi. ResearchGate, www.researchgate.net/publication/344871967_Predicting_Package_Delivery_Time_For_Motorcycles_In_Nairobi.

This study uses XGBoost, a supervised regression model, to predict the estimated time of delivery of a motorcycle-transported delivery in Nairobi. The author decided to use input variables including the client’s past orders information, the rider’s past orders information, weather, delivery distance, drop off and pickup location, and time of day. It describes the author’s approach to determining feature importance, particularly through graphics and XGBoost tree visuals. There is also a discussion of examining the results through the specific lens of a delivery date – it is better to predict late than early – and thus an optimized model should account for this by reprimanding a model harsher for predicting an early time, as opposed to a late time.


#### How to Set Realistic Delivery Dates in High Variety Manufacturing Systems
Mezzogori D., Romagnoli G., & Zammori F. (2019). How to Set Realistic Delivery Dates in High Variety Manufacturing Systems. International Federation of Automatic Control. www.sciencedirect.com/science/article/pii/S2405896319314983.

This paper discusses how neural networks can be used to predict workload (called WLC, workload control) and ultimately delivery date in manufacturing plants. The researchers set out to design a neural network that could streamline workload order (in order to minimize queues and optimize delivery dates). Their neural network had the following structure: an input layer with 12 inputs, a single output neuron delegated to make the forecast of the final delivery time, three hidden layers, each one with 32 neurons (plus a bias) with the Relu activation function, and batch normalization after each hidden layer. The authors found significant optimizations of manufacturing logistics and delivery times with their model.


## Introduction Outline
1. Intro: We seek to build a neural network that can accurately predict shipping times for eBay. Users of eBay buy and sell various products, use various shipping providers, reside in various locations, etc. making this an interesting and challenging problem with many features.
2. Background: eBay reports that machine learning has not yet been applied to, and could be useful in, the prediction of shipping times.
3. Transition paragraph: Because predicted values are on a discrete scale (number of days), we used an ___ architecture suitable for a discrete response rather than a continuous one.
4. Details paragraph: 
Despite the enormous number of data records provided for testing, 15 million shipping records still might not be representative of the enormous number of shipments that go through Ebay every day (25 million). 
5. Assessment paragraph: Our model was able to predict eBay shipping times with an accuracy of __ and we found that the most important features in predicting shipping times were X, Y, and Z.
6. Ethics Component
  - Environmental Implications of Fast Shipping: If we begin to predict fast shipping times, the shipping infrastructure will aim to complete them in a sort of self-fulfilling model. Rushing shipping procedures often results in a slew of non-sustainable methods such as shipping half-full boxes, a method that wastes shipping materials (boxes, foam peanuts, tape) and fuel for the transportation method. 
  - Potential Bias in Dataset: Our full data set could, at most, provide one data point from 60% of eBay users. We do not know how eBay generated their dataset; if eBay pulled from a specifc region, then our model would likely be poor at predicting shipping times from other, not represented regions in the dataset.


### Project Description
Accurate predictions of product delivery times is a basic a crucial aspect of customer service for any company that deals with the transpoortation of goods from seller to buyer. For eBay, predicting delivery times is especially difficult because shipments are made by over 25 million individual sellers across the globe who have different preferred carrieres, as well as different levels of proactiveness to package and ship good shortly thereafter purchase. The range of goods sold on eBay along with the variety of sellers and buyers makes this problem interesting and challenging. The objective is to estimate the number of calendar days it will take for a buyer to receive the product after payment is made:

### payment date + handling time + transit time = delivery date 

The team will enter eBay's [evalAI Machine Learning competition](https://eval.ai/web/challenges/challenge-page/1205/overview) to get access to a training dataset of over 15 million of eBay's shipping records.

## Goals
1. Compare different NN architectures to determine which best fits problem.
2. Determine most important features for prediction (1) handling time and (2) transit time
3. Build an accurate model that can realistically be used by eBay!


