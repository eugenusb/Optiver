Optiver Realized Volatility Prediction
======================================

Description
-----------

This is an approach for the recent Optiver's Kaggle [competition](https://www.kaggle.com/c/optiver-realized-volatility-prediction). 
The goal is to predict the realized volatility during the next ten minutes for several stocks departing from certain information about the order and trade book of each stock for the ten minutes window right before the target period.

Quite simply, the idea is to learn the parameters of a certain probability distribution called inverse Gamma to fit the realized volatility through a neural network. This is achieved by a straightforward modification of MDNs. A more detailed exposition can be found [here](https://eugenusb.github.io/machine/learning/2021/10/23/Forecasting-volatility.html).

Project Organization
-----------

    ├── data                 <- folder holding train and test data    
    ├── src                  <- folder holding the necessary scripts to pre-process the data and generate the models
    ├── Starter.ipynb        <- a Jupyter notebook that shows how to run the model and features some basic EDA

Usage
------

In order to run the notebook, you would need first to download the data from [here](https://www.kaggle.com/c/optiver-realized-volatility-prediction/data) and save it to the `data` folder.
I strongly recommend to run the data pre-processing step only once and saving afterwards the resulting datasets in parquet format. You will find a cell to save the generated data and another one to load it in the first section of the Jupyter notebook, just remove the comments as needed.