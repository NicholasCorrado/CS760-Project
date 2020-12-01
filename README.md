# CS760-Project
This repo is using machine learning models to predict the microplastic content in marine areas.

## Depedency

Python 3

## Introduction

The study of microplastic pollution is still in its infancy, and not much is understood surrounding itsimpacts, hazards, and risks to the environment and human health.  Growing concern among researchershas prompted extensive data collection.Using this data, we seek to predict the amount of microplasticpollution in marine and freshwater systems globally.  We use three machine learning models: linear regression,decision trees, and nearest neighbors models to predict.


## Datasets
We select the data from ..............  The dataset contains sampling date, sampling latitude, sampling longitude, sea surface temperature, Sample_Volume, Total Microplastic Pieces and Average Microplastic Pieces.  In this project, we select sampling date, sampling latitude, sampling longitude and sea surface temperature as indepedent variables and Total Microplastic Pieces as dependent variable. (Marine_Clean.csv)
We also generate testing datasets. We choose several samples across 10 different marine areas in the earth for the microplastic content prediction based on three different models. These areas include North Atlantic Ocean, Caribbean Sea, West Pacific Ocean, North Pacific Ocean and Central Pacific Ocean. The predicted results will be compared and evaluated in our models. (TestCase.csv)

## Datasets Accessibility
Please contact authors to get access to related csv files.

## Models

*  Linear Regression Model

*  Regression Decision Tree Model

*  Neareast Neighbors Model

## Main Function
1. All three models related python files need to be imported in main.py.
2. Run sample python code to apply three models to predict.

## Authors
1. Nicholas Corrado,ncorrado@wisc.edu 
2. John Li, zli769@wisc.edu
3. Dan Kiel, dkiel2@wisc.edu
