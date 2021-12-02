# Predicting eBay Delivery Times 
## Millie M, Meghna L, Hannah M, Nate D, Indiana H

First Draft

## Abstract

In this paper, we attempted to predict shipment delivery times for eBay from a shipment dataset that they provided. We explore four models (Linear Regression, Fully Connected Neural Network, XGBoost and CatBoost) to compare the effectiveness of the models on shipment data. We also examine different strategies to clean our dataset, engineer new features, and fine-tune each model. Overall, our CatBoost model proved to be the most successful. Looking forward, we suggest investigating models that work best on categorical data or tabular data. 

## Introduction

The economy of online shopping is bigger than it has ever been and continues to grow each year. It follows that the market for being able to deliver goods quickly and reliably is becoming more and more competitive. In addition to improving the overall flow of transactions, knowing when a package will be delivered is a major factor in customer satisfaction, making the ability to accurately predict delivery dates essential to companies such as eBay.

The process of achieving this, however, poses many challenges. In the case of eBay, the majority of transactions are carried out between individual sellers and buyers, often resulting in the data for goods and their delivery being inconsistently recorded. This, in addition to packages being processed by a variety of different delivery services, means that data labels are frequently missing or incorrect. Further, the shipment date is largely left to the sole decision of each individual seller, resulting in a high degree of variability.

![image of shipping process](/images/diagram.png)

Our team will provide a solution to these problems and provide a model to enable the accurate prediction of delivery dates using machine learning. Because predicted values are in the range of the number of days between shipment and delivery, we will employ an architecture suitable for a discrete response rather than a continuous one.

To implement our model, a number of decisions will have to be made and tested, including deciding the optimal means by which to clean the data (for instance, whether to omit training data that has missing or incorrect features or to assign it average values based on correctly labeled training data), deciding whether to compute the estimation holistically from shipment to delivery date or to compute multiple separate estimates on separate legs of the delivery process, and deciding which features to include and in which leg.

Should our model have some error, it is important that it produces random rather than systematic error. Specifically, we want to avoid creating a model which might consistently predict early delivery dates, which could lead to sellers and delivery services rushing packages and resulting in the employment of more non-sustainable methods, such as shipping half-full boxes, as well as increasing the pressure on employees to have to work faster and faster.

Ultimately, our ideal model will be able to accurately predict the exact day of delivery 100% of the time and demonstrate which features are the most important in estimating delivery dates.


## Literature Review

The current state of research for the delivery time estimation problem shows how varied and nuanced delivery data and time estimation can be. There have been studies that focus on solving complex delivery problems such as having multiple legs of shipment, but also studies that are smaller in scale and detail how delivery time estimation is calculated for food delivery services. In this section, we will describe the relevant studies that cover delivery time estimation and discuss how each of their findings are applicable to our project. 

To solve a problem that is similar to ours, logistics supplier [Aramex](https://aws.amazon.com/blogs/industries/how-to-predict-shipments-time-of-delivery-with-cloud-based-machine-learning-models/) uses machine learning to predict delivery times, claiming that its model has increased the accuracy of delivery predictions by 74%. They describe a multi-leg approach in which the shipping process is divided into multiple steps to be estimated individually. For instance, the time from seller to processing hub is estimated separately from the time from customs to buyer in international transactions. The architecture integrates systems such as Amazon Redshift clustering and Amazon SageMaker to process its data. For our project, we also explore the benefits of splitting the different legs of shipment into smaller segments; handling time and shipment time. 

Another study, [Predicting Package Delivery Time For Motorcycles In Nairobi](https://www.researchgate.net/publication/344871967_Predicting_Package_Delivery_Time_For_Motorcycles_In_Nairobi) details how modifying the input variables for the model can help the model narrow in on important features. This study uses XGBoost, a supervised regression model, to predict the estimated time of delivery of a motorcycle-transported delivery in Nairobi. The researchers used input variables such as the client’s past orders information, the rider’s past orders information, weather, delivery distance, drop off and pickup location, and time of day. It describes the author’s approach to determining feature importance, particularly through graphics and XGBoost tree visuals. There is also a discussion of examining the results through the specific lens of a delivery date – it is better to predict late than early – and thus an optimized model should account for this by reprimanding a model harsher for predicting an early time, as opposed to a late time. We are using XGBoost as well because of its ability to determine feature importance when given a complex set of input variables.

Although it is a slightly modified problem, the researchers [Araujo and Etemad (2020)](https://arxiv.org/abs/2009.12197) document how they went about solving the problem of last-mile parcel delivery time estimation using data from Canada Post, the main postal service in Canada. They formalize the problem as an Origin-Destination Travel Time Estimation problem and compare several neural networks in order to generate the best results. Their findings indicate that a ResnNet with 8 residual convolutional blocks has the best performance, but they also explore VGG and Multi-Layer Perceptron models. We used the findings from this model to guide our choices when experimenting with different models.

Another version of the delivery time estimation problem is predicting workload (called WLC, workload control) and can be applied to predicting the delivery date in manufacturing plants. 
A paper by [Romagnoli and Zammori (2019)](https://www.researchgate.net/publication/345008117_Defining_accurate_delivery_dates_in_make_to_order_job-shops_managed_by_workload_control) describes how researchers set out to design a neural network that could streamline workload order (in order to minimize queues and optimize delivery dates). Their neural network had the following structure: an input layer with 12 inputs, a single output neuron delegated to make the forecast of the final delivery time, three hidden layers, each one with 32 neurons (plus a bias) with the Relu activation function, and batch normalization after each hidden layer. The authors found significant optimizations of manufacturing logistics and delivery times with their model. Initial trials to solve our problem used a neural network with a similar model, but we also modified the number of layers, inputs, and neurons to explore different structural possibilities.

Finally, there are numerous challenging elements that must be considered when solving this problem, and [Wu, F., and Wu, L., (2019)](https://ojs.aaai.org//index.php/AAAI/article/view/3856) cover the many difficulties of predicting package delivery time such as multiple destinations, time variant delivery status, and time invariant delivery features. Their article describes how DeepETA, their proposed framework, is a spatial-temporal sequential neural network model that uses a latest route encoder to include the location of packages and frequent pattern encoder to include historical data. There are 3 different layers to this model and through experimenting on real logistics data, the authors show that their proposed method outperforms start of the art methods. This paper is useful in identifying the challenges of predicting delivery time and how we may go about solving them.

## Methods

### Overview
To create our neural network model and attain our results, we used a number of tools for the different stages of our project. To construct our model, we used [Pytorch, sklearn, and Jupyter notebooks](https://github.com/milliemince/eBay-shipping-predictions) for most of our development. We ran our code on the Pomona High Performance Computing servers to utilize more GPU power. We used a linear regression model with regularization penalties, as well as [XGBoost](https://xgboost.readthedocs.io/en/stable/), to infer which features were the most important for predicting (a) the handling time, and (b) the shipment time. We also compared our findings to a fully connected network. Finally, we used CatBoost to identify the hyperparameters to fine tune for our final model. Ultimately, we used CatBoost for our final predictions as it outperformed the other models.

### Dataset
Our dataset was provided by [eBay](https://eval.ai/web/challenges/challenge-page/1205/overview). It contains 15 million shipment records and a validation dataset containing 2.5 million shipment records. Each shipment record contains 19 features. To visualize our dataset, we used pandas and matplotlib. This allowed us to generate graphs for each feature in our dataset. Some of the images we generated include the category ID (which broadly classifies the type of item purchased), the item price, and the shipment method ID (which signifies the type of shipment service declared by the seller to be used for the item). 

![image of shipping process](/images/CategoryID.png)
![image of shipping process](/images/ItemPrice.png)
![image of shipping process](/images/ShipmentMethodID.png)

### Goal
Our goal was to use this dataset to create a model that can accurately predict the delivery time (in days) of an eBay shipment. To accomplish this goal, we broke our problem down into a few steps:

<ol>
 <li>Cleaning our data </li>
 <li>Creating new features </li>
 <li>Determining feature importance </li>
 </ol>


### Cleaning our data
To clean our dataset, we first needed to handle irregularities and missing data. To do so, we omitted some examples, and replaced other examples with averages from similar training instances. Many of the data points contained missing entries, so to account for this, we either replaced the entries or deleted them. 
For example, one of our features, the minimum/maximum transit time that the carrier provided, had many missing entries, so we replaced the missing entries with an estimate that we calculated by obtaining an average transit time based on the shipment_method_id feature associated with that item. We did this for the declared handling days feature as well, using an average from the seller id feature. Finally, we also did this for any missing entries in the weight feature by replacing them with the average weight for shipments that have the same category as the item.
We also had to clean the data so that all entries were able to be used by the neural network. We converted features that were represented as strings, such as b2c_c2c and package_size, into discrete numeric encodings. We also converted all of the entries in the weight_units feature to the same unit, instead of having a weight feature and a separate unit feature.



### Creating new features
Once we had cleaned the data, we also needed to create a new feature that accounts for the zipcode data we were provided with. We created a new feature:` zip_distance`, using the `item_zip` and `buyer_zip`. The feature `zip_distance` quantifies the haversine distance between the two zip codes (distance across the surface of a sphere). We utilized the package [uszipcode](https://pypi.org/project/uszipcode/) to retrieve a latitude and longitude for zip codes in the packages database and a package called mpu to calculate the haversine distance between the two points. For US zip codes that were not in the database and returned NaNs, we temporarily decided to calculate `zip_distance` as the distance between the middle of the two states involved in the sale. For zip codes that were not in the database since they were from non-US countries _______. 

### Feature engineering
Although we created new features that we thought to be important, we wanted to ensure that those features were necessary for the model. To proceed with this process, we looked at feature importance in both a linear model and a decision tree through XGBoost.

In the linear regression model, we analyzed the coefficients assigned to each input variable. The larger the assigned coefficient, the more ‘important’ the input is for determining the output. We additionally ran a lasso regression for this same purpose, hoping to identify any highly unnecessary inputs and remove over-dependency on highly weighted inputs. However, despite scaling our data, we ran into the issue of having all but one feature assigned a non-zero coefficient. 

We also used XGBoost to delineate the most important features. The features nearest the top of the tree are more important than those near the bottom. We ran multiple trees with multiple combinations of variables to see which ones repeatedly showed up at the top of the tree, despite being paired with other variables. 

## Models

After our data was clean, we began trying different models to see what was most effective and gave the best results.

### Linear Regression Model
We decided to start with a naive linear regression. Despite knowing that a linear model could probably not correctly learn this large, complex, real-world data set, linear regression is a powerful algorithm that could at least learn relative feature importance. We used three types of linear regression: (1) standard regression without weight decay, and (2) lasso regression (using L2 penalty). Lasso regression has a an L1 weight decay penalty, and thus tends to push irrelevant features’ coefficients to 0. The results of the 3 linear regression models are as follows:

|       | Test Set Loss |
| ----------- | ----------- |
| Standard Linear Regression  | 0.82    |
| Lasso Regression | 0.89        |

The following table shows the coefficient values learned for each feature in the 3 regression models:

|    | b2c_c2c | shipping_fee | carrier_min_estimate | carrier_max_estimate |
|----|---------|--------------|----------------------|----------------------|
| SR | 0.2635  | 0.0030       | 0.2217               | 0.4032               | 
| LR | 0.0     | 0.0          | 0.0                  | 0.0                  | 

|    | category_id | item_price | quantity | weight   |
|----|-------------|------------|----------|----------|
| SR | 0.0183      | -0.0001    | 0.003    | -9.5e-06 |
| LR |0.005        | 0.0001     | 0        | 1.9e-06  |

Standard linear regression suggests that carrier_max_estimate, carrier_min_estimate, and b2c_c2c are the most important features. It also learned that shipping_fee and quantity are of minimal, but non-zero importance. The remainder of the features are found to have no importance. However, these models do not perform well relative to the benchmark random classifier loss provided by eBay: 0.75. Since the model cannot perform better than random guessing, the feature importances learned from it should be taken with a grain of salt.

### Fully Connected Model 
TODO 

Num Layers
Type of Layers
Learning rate
Num epochs

### XGBoost
XGBoost is a popular gradient boosting decision tree algorithm that has been featured in the winning submissions of many machine learning competitions. Essentially, XGBoost is an ensemble method that combines several weak learners (the individual decision trees) into a strong one. Gradients come into play when each new tree is built; subsequent trees are fit using the residual errors made by predecessors. Another advantage of XGBoost is its interpretability. Trees are easy to understand, and they are great models for discerning feature importance. The higher up a feature is on the various decision trees, the more important that feature is. As each tree is built, splits are decided based on what “decision” (i.e. is b2c_c2c true or false?) best evenly partition the data. This is why features high up in the tree are indicators of the most important features. XGBoost performs well on various datasets, and we wanted to explore how it would perform on the eBay dataset. 

Feature importances learned by XGBoost:

| Feature                | Feature Importance  |
|------------------------|---------------------|
| b2c_c2c                | 0.015656            |
| seller_id              | 0.047507            |
| declared_handling_days | 0.042284            |
| shipment_method_id     | 0.035131            |
| shipping_fee           | 0.028294            |
| carrier_min_estimate   | 0.044301            |
| carrier_max_estimate   | 0.055198            |
| item_zip               | 0.059736            |
| buyer_zip              | 0.028275            |
| category_id            | 0.022790            |
| item_price             | 0.042802            |
| quantity               | 0.047650            |
| weight                 | 0.057305            |
| package_size           | 0.020258            |

By far the most importance feature learned by XGBoost is handling days, a feature we engineered from the difference in acceptance_carrier_timestamp and payment_date (i.e. the number of days between the time the package was accepted by the shipping carrier and the time that the order was placed). Following this, it seems that weight, item_zip, buyer_zip and carrier_max_estimate are also important. The only features that XGBoost has learned to have lower or minimal importance are buyer_zip_median_income, item_zip_pop_density, package_size, category_id, buyer_zip, shipping_fee, and b2c_c2c. We can, to some extent, trust that the feature importances learned by XGBoost carry some weight, because the model was able to achieve a loss of 0.50, a large improvement from the linear regression models.
 
### CatBoost
Another tool we used is a random forest/decision tree package called Catboost. This is similar to XGBoost, however it focuses on categorical data (hence the “cat” in Catboost). This is discrete data that represents categories rather than continuous numerical data. This tool was useful to us because several of our important features are categorical, such as zipcodes, package size (small, medium, large), and item type.

According to Catboost documentation, the best parameters to fine tune are: learning rate, L2 regularization (coefficient of the L2 regularization term of the cost function), random strength (the amount of randomness to use for scoring splits when the tree structure is selected), bagging temperature (which impacts how random weights are assigned), border count (the number of splits for numerical features), and tree growing policy (which impacts the symmetry of the trees). We used this data to fine tune these parameters in order to obtain better performance on our data. 

Upon a preliminary test of a Catboost model on a subset of our data (20,000 rows) with default parameters, we were able to achieve a loss of 0.49 (using the loss function provided by eBay). This is a promising number, considering the small subset of data used and lack of fine tuning. The contrast of Catboost’s performance shows how a categorically tailored package seems to be a better choice. Catboost also includes an overfitting-detection tool. This stops more trees from being created if the model begins to perform worse. 


### Training
Once we had cleaned and processed our data, we trained 4 models to compare the results from each one. These models were:

1. Linear Regression
2. Fully Connected Neural Network
3. XGBoost (Decision Tree Gradient Boosting Algorithm)
4. CatBoost (Another Decision Tree Gradient Boosting Algorithm that deals better with categorical features)

### Loss function
After training our models, we used the loss function provided by eBay of which the baseline (random guessing) loss is 0.75. This loss function is essentially an average of how many days the prediction was off by, where late predictions are weighted more heavily than early predictions. Our goal was to obtain a loss that is significantly lower than 0.75 for our model. 


## Results

In our project, we trained four models to try to find the best architecture to predict shipping times. We ran our cleaned data through each model and compared effectiveness. We compared these models by evaluating the loss for each of them. 

Comparing these models, we can see that Catboost performed the best. 

With XGBoost, after fine tuning, our loss was 0.51 using eBays provided loss function. 
Catboost had a loss of ___ after fine tuning. 

| Model      | Loss |
| ----------- | ----------- |
| Linear Regression NN Model      | __       |
| Fully Connected Neural Network   | __        |
| XGBoost   | __        |
| CatBoost   | __        |



## Discussion

### Model Evaluation: Feature Importance
To evaluate our models, we first looked at the features that each of the models deemed to be the most important. This is so we can determine whether the models performed well on certain subsets of the data, analyzing whether some subsets had bias, or if different models prioritized certain features over others. 

’We learned it is possible to determine the most important features in predicting delivery times for the eBay dataset using Random Forests with boosting (using XGBoost). We can look at the weights for each of the features in our dataset after using XGBoost’s decision trees to determine the highest weighted features. 

TODO
*insert table maybe for feature importance once we complete model runs*

XGBoost learned many boosted decision trees on random subsets of the data. Features that are evaluated higher up in the trees correspond to the features which the model has learned are more important. We have found that the most important features are `handling_days` and `carrier_max_estimate`, followed by some mid-tier importance features such as `weight`, `zip_distance`, and `shipment_method_id`.

### Model Evaluation: Predictions

We also wanted to compare how the models might differ in effectiveness or use case. To do so, we can look at the kinds of predictions that each model made to see if there are any key differences. We created a scatter plot of predicted vs actual shipping dates, categorized by product category or price to look at our predictions for each model.

*insert scatter plot once we complete our model runs*

Our dataset has many categorical features, like `shipment_method_id`, `item_zip`, `buyer_zip`, and `b2c_c2c`. XGBoost performed relatively well, which motivated finding a similar algorithm that could better handle these categorical features. Our findings (even before fine tuning CatBoost) show that this is true.


### Comparison
TODO
To see how our work compares to others, we looked at the other competitors in the eBay competition and the leaderboard. 
Comparing our models to the other teams, we found that our models performed ____
*insert image of the other competitors' scores?*  


### Ethics Discussion
One of the key implications of fast shipping is the effect that it has on the environment. The environmental effects of shipping are widespread, affecting everything from air pollution and water pollution to oil spills and excess waste. The increase of fast delivery times and same-day delivery options provided by companies has led to a greater environmental cost such as more trucks on the road, air quality issues, and package waste. Furthermore, there is also a high human labor cost associated with delivery and shipment. The delivery drivers who are responsible for delivering packages to the correct recipients face long shifts without stops, and some companies even encourage fewer stops to prioritize fast shipping. This can lead to detrimental effects as delivery drivers may be incentivized or even forced to provide fast delivery no matter the impact on their wellbeing. It is incredibly important to consider the human impact of fast shipping and prioritize the safety of those who work in the shipping industry.  If eBay were to implement a shipment prediction algorithm, it would be incredibly important to ensure that the model was not predicting shipping times that are too fast, or else it may lead to a greater environmental and human impact.  Finally, there is a positive impact of shopping on eBay compared to other alternatives (such as Amazon). eBay offers many second-hand goods and products, so consumers buying second hand are positively contributing to the environment by creating less waste. By offering pre-owned items, eBay saves items from ending up in the landfill and promotes sustainable shopping. 

## Reflection

This project illustrated to us the difficulty of working with large datasets and the many complexities that come with them. Although the data set was already created for us to use, it was still difficult to clean the data and identify the most important features. When we were cleaning, we decided on replacing missing values with averages, however looking back on our cleaning process, this may have added a lot of noise to our model and hindered accurate predictions. An alternative to this would be to discard missing entries altogether. Next time, we would have tried this instead of using averages to see if it helps our predictions. 

To extend this work, we would try to use a tabular model such as fast.ai Tabular. Because our data was in a tabular format, this model may be better suited for the specific kind of data we are using. We could also extend our work by implementing some of the newer algorithms that we read about through our literature review. For example, [Wu, F., and Wu, L., (2019)](https://ojs.aaai.org//index.php/AAAI/article/view/3856) created DeepETA, a new model that works well on time-estimation problems. We would take this model and adapt it for our problem to see if it applies in the case of delivery time estimation. 

