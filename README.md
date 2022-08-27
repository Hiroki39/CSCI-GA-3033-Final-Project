# CSCI-GA 3033 Final Project

## Introduction

This project aims to provide 5-minutes short-term price prediction based on market data and sentiment data.

## Usage

First of all, run cells in `EDA.ipynb`, which conducts some exploratory data analysis and generates some plots. It gives user a clear idea about how the model is performing.

Then, run

```python
python pipeline.py regularization_search
```

in the terminal. The `pipeline.py` conducts the hyperparameter search based on the information in `searches.py`. This framework is expandable: user could try edit the code in `get_rolling_data` function in `pipeline.py` and define new hyperparameter search in `pipeline.py` to conduct other kinds of hyperparameter search.

Then open the notebook `ResultAnalysis.ipynb` to conduct the result analysis, select the best combinations, and visualize.
# CSCI-UA-3033-Final-Project
