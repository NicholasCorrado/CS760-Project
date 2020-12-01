# CS760-Project
This repo is an application of machine learning models to predict the microplastic content in marine/fresh water areas.

## Depedency
Python 3

## Introduction

The study of microplastic pollution is still in its infancy, and not much is understood surrounding itsimpacts, hazards, and risks to the environment and human health.  Growing concern among researchershas prompted extensive data collection.Using this data, we seek to predict the amount of microplasticpollution in marine and freshwater systems globally.  We use three machine learning models: linear regression,decision trees, and nearest neighbors models to predict.


## Datasets
We select the data from .  The dataset contains sampling date, sampling latitude, sampling longitude, sea surface temperature, Sample_Volume, Total Microplastic Pieces and Average Microplastic Pieces.  In this project, we select sampling date, sampling latitude, sampling longitude and sea surface temperature as indepedent variables and Total Microplastic Pieces as dependent variable. (Marine_Clean.csv)
We also generate testing datasets. We choose several samples across 10 different marine areas in the earth for the microplastic content prediction based on three different models. These areas include North Atlantic Ocean, Caribbean Sea, West Pacific Ocean, North Pacific Ocean and Central Pacific Ocean. The predicted results will be compared and evaluated in our models. (TestCase.csv)

## Datasets Accessibility
Please contact authors to get access to related csv files.

## Models

*  Linear Regression Model

*  Regression Decision Tree Model

*  Neareast Neighbors Model

## Linear Regression Model

## Regression Decision Tree Model
1. tree = regressionTreeConstruct(filename, depth = 8, split=False, printTree = True)
We can built the regression tree by import the csv data file. 
The default max depth of tree is setup to 8, and user can adjust it to any reasonable depth.
The user can choose to further split the data into training and testing for validation.
The user can select printTree to print the split rules to console.

2. evalTree(filename)
Evaluate the tree generated for different depth. 
The goodness of fit (r2 = 1 - sse/sst) would be plotted with training and testing dataset based on 9:1 ratio.
The user can choose the best fit decision tree models with optimal tree depth.

3. predictions = testCase(filename,tree)
Can use the selected regression tree model to predict the microplastic content in test cases.
The predicted content would be ploted.


## Nearest Neighbor Model


## Main Function
1. All three models related python files need to be imported in main.py.
2. Run sample python code in main.py to apply three models to predict the total pieces of microplastic in certain sea area with sampling time and temperature.

## Authors
1. Nicholas Corrado, ncorrado@wisc.edu 
2. John Li, zli769@wisc.edu
3. Dan Kiel, dkiel2@wisc.edu
