# Predicting eBay Delivery Times 
## Millie Mince, Meghna Lohia, Hannah Mandell, Nate Dailey, Indiana Huey

### Introduction Outline
1. Intro: We seek to build a neural network that can accurately predict shipping times for eBay. Users of eBay buy and sell various products, use various shipping providers, reside in various locations, etc. making this an interesting and challenging problem with many features.
2. Background: eBay reports that machine learning has not yet been applied to, and could be useful in, the prediction of shipping times.
3. Transition paragraph: Because predicted values are on a discrete scale (number of days), we used an ___ architecture suitable for a discrete response rather than a continuous one.
4. Details paragraph: 
Despite the enormous number of data records provided for testing, 15 million shipping records still might not be representative of the enormous number of shipments that go through Ebay every day (25 million). 
5. Assessment paragraph: Our model was able to predict eBay shipping times with an accuracy of __ and we found that the most important features in predicting shipping times were X, Y, and Z.

### Project Description
Accurate predictions of product delivery times is a basic a crucial aspect of customer service for any company that deals with the transpoortation of goods from seller to buyer. For eBay, predicting delivery times is especially difficult because shipments are made by over 25 million individual sellers across the globe who have different preferred carrieres, as well as different levels of proactiveness to package and ship good shortly thereafter purchase. The range of goods sold on eBay along with the variety of sellers and buyers makes this problem interesting and challenging. The objective is to estimate the number of calendar days it will take for a buyer to receive the product after payment is made:

### payment date + handling time + transit time = delivery date 

The team will enter eBay's [evalAI Machine Learning competition](https://eval.ai/web/challenges/challenge-page/1205/overview) to get access to a training dataset of over 15 million of eBay's shipping records.

## Goals
1. Compare different NN architectures to determine which best fits problem.
2. Determine most important features for prediction (1) handling time and (2) transit time
3. Build an accurate model that can realistically be used by eBay!


