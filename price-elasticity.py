# %%
from __future__ import print_function
from statsmodels.compat import lzip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols

beef_data = pd.read_csv('beefdata.csv')
beef_data.head(10)

# Ordinary Least Squares Estimation
beef_model = ols("Quantity ~ Price", data=beef_data).fit()
print(beef_model.summary())

# %%
# Graphs to interpret regression analysis in interactice cell
fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_partregress_grid(beef_model, fig=fig)

if __name__ == '__main__':
    pass

