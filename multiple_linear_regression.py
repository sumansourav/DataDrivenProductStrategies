# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_excel('IA1-PanasonicLCDTV.xlsx')
dataset.head()
#Dropping the categorical variables as we already have created dummy variables for them.
dataset.drop(['Screen Size', 'Pixel', 'Motion Rate', 'Selling Price'],axis = 1, inplace=True)

# Put data into a X matrix and a y vector
X = dataset.iloc[:,1:-1]
y = dataset.iloc[:,-1]

# Add an intercept to X
import statsmodels.formula.api as sm
X['Intercept'] = pd.DataFrame(np.ones((704,1),dtype='int'))

# Build a OLS regression model
regressor_ols = sm.OLS(endog=y, exog=X).fit()
regressor_ols.summary()
