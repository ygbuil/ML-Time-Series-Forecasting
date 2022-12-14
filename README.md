# ML-Time-Series-Forecasting

## Introduction

This project is a Time Series Forecasting code that uses XGBoost to do multi-step forecasting. Provided a Time Series, the code generates various time and statistical features in order to produce a forecast based on the information of the Time Series itself. The code is meant to serve as a generic starting point that can be useful in many business problems, and from here you can add additional features to the model depending on your needs.

## Relevant code features:
* Multi-step forecasting.
* Supports multivariate forecasting (external features).
* Multiple Time Series can be forecasted in parallel.
* Modular and easy to customize. Switch on and off different code features from the "constants/constants.py" file.

## Project components:
* <b>data/example_dataset.csv</b>: An example dataset to try out the code. Contains 7 Time Series of historical demand of products in different stores.
* <b>pipeline/pipeline.py</b>: Main pipeline. This is the file you will need to run.
* <b>objects</b>: Contains all functions and classes used in the pipeline.
* <b>constants/constants.py</b>: This is where you can tune different model parameters. Every parameter is explained inside "objects/constants.py".

## How to use it:
* <b>Step 1</b>: Clone "/ML-Time-Series-Forecasting" to "/path/in/your/local/machine".
* <b>Step 2</b>: Inside "pipeline/pipeline.py", in line 6, define path = "/path/in/your/local/machine".
* <b>Step 3</b>: Run "pipeline/pipeline.py".

Disclaimer: Parameters in "constants/constants.py" are not optimized for the given dataset, they are set for demonstration purposes.

Example of the resulting forecast:

![alt_file](https://github.com/ygbuil/ML-Time-Series-Forecasting/blob/master/forecast_result_example.png)
