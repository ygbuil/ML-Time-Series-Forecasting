# ML-Time-Series-Forecasting

## Introduction

This project is a Time Series Forecasting code that uses XGBoost to do multi-step forecasting. Provided a Time Series, the code generates various time and statistical features in order to produce a forecast based on the information of the Time Series itself. The code is meant to serve as a generic starting point that can be useful in many business problems, and from here you can add additional features to the model depending on your needs.

## Relevant code features:
* Multi-step forecasting.
* Supports multivariate forecasting (external features).
* Multiple Time Series can be forecasted in parallel.
* Modular and easy to customize. Switch on and off different code features from the `src/constants/constants.py` file.

## Project components:
* `src/main.py`: Main pipeline. This is the file you will need to run.
* `src/constants/constants.py`: This is where you can tune different model parameters. Every parameter is documented in `src/objects/constants.py`.
* `src/objects`: Contains all functions and classes used in the pipeline.
* `src/data/products_sales.csv`: An example dataset to try out the code. Contains 6 Time Series of historical demand of products in different stores.
* `src/tests`: Unit and integration tests. They can be run using the `pytest` console command.

## How to use it:
* <b>Step 1</b>: In `src/constants/constants.py`, set the `root_path` variable to the apropiate root path in your machine. The root path is considered to be the `src` folder.
* <b>Step 2</b>: Run `src/main.py`.

_Disclaimer: Parameters in "constants/constants.py" are not optimized for the given dataset, they are set for demonstration purposes._

Example of the resulting forecast:

![alt_file](https://github.com/ygbuil/ML-Time-Series-Forecasting/blob/master/images/forecast_result_example_1.png)

![alt_file](https://github.com/ygbuil/ML-Time-Series-Forecasting/blob/master/images/forecast_result_example_2.png)
