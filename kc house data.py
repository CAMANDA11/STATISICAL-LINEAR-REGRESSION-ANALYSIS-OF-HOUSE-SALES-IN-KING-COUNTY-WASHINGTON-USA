# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 00:19:23 2021

@author: amand
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 10:16:49 2021

@author: amand
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.decomposition import PCA
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.offline import plot
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "svg"
from sklearn.svm import SVR


# Importing dataset and examining it
dataset = pd.read_csv("KcHouseData.csv")
pd.set_option('display.max_columns', None) # to make sure you can see all the columns in output window
print(dataset.head())
print(dataset.shape)
print(dataset.info())   
print(dataset.describe())


# Plotting Correlation Heatmap
corrs = dataset.corr()
figure = ff.create_annotated_heatmap(
    z=corrs.values,
    x=list(corrs.columns),
    y=list(corrs.index),
    annotation_text=corrs.round(2).values,
    showscale=True)
figure.show()



# Dividing dataset into label and feature sets
X = dataset.drop (['price', 'id','date'],axis = 1) # Features
Y = dataset['price'] # Labels
print(type(X))
print(type(Y))
print(X.shape)
print(Y.shape)
print(dataset.info())
print(dataset.describe())

# Normalizing numerical features so that each feature has mean 0 and variance 1
feature_scaler = StandardScaler()
X_scaled = feature_scaler.fit_transform(X)
print(pd.DataFrame(X_scaled).head())

# Tuning the random forest parameter 'n_estimators' and implementing cross-validation using Grid Search
# rfr = RandomForestRegressor(criterion='mse', max_features='sqrt', random_state=1)
# grid_param = {'n_estimators': [600,650,700,750,800]}

# gd_sr = GridSearchCV(estimator=rfr, param_grid=grid_param, scoring='r2', cv=10)

# gd_sr.fit(X_scaled, Y)

# best_parameters = gd_sr.best_params_
# print(best_parameters)

# best_result = gd_sr.best_score_ # Mean cross-validated score of the best_estimator
# print(best_result)

#Building random forest using the tuned parameter
# rfr = RandomForestRegressor(n_estimators=650, criterion='mse', max_features='sqrt', random_state=1)
# rfr.fit(X_scaled,Y)
# featimp = pd.Series(rfr.feature_importances_, index=list(X)).sort_values(ascending=False)
# print(featimp)
 

# # Selecting features with higher sifnificance and redefining feature set
# X_ = dataset[['sqft_living', 'grade', 'lat', 'sqft_living15','sqft_above']]

# feature_scaler = StandardScaler()
# X_scaled_ = feature_scaler.fit_transform(X_)

# # Tuning the random forest parameter 'n_estimators' and implementing cross-validation using Grid Search
# rfr = RandomForestRegressor(criterion='mse', max_features='sqrt', random_state=1)
# grid_param = {'n_estimators': [600,650,700,750,800]}

# gd_sr = GridSearchCV(estimator=rfr, param_grid=grid_param, scoring='r2', cv=10)

# gd_sr.fit(X_scaled_, Y)

# best_parameters = gd_sr.best_params_
# print(best_parameters)

# best_result = gd_sr.best_score_ # Mean cross-validated score of the best_estimator
# print(best_result)


# ############################################################
#Linear Regression
#Tuning the SGDRegressor parameters 'eta0' (learning rate) and 'max_iter' using Grid Search
# sgdr = SGDRegressor(random_state = 1)
# grid_param = {'eta0': [.0001, .001, .01, .1, 1], 'max_iter':[1000,2000,3000,5000]}

# gd_sr = GridSearchCV(estimator=sgdr, param_grid=grid_param, scoring='r2', cv=10)

# gd_sr.fit(X_scaled, Y)

# best_parameters = gd_sr.best_params_
# print(best_parameters)

# best_result = gd_sr.best_score_ # Mean cross-validated score of the best_estimator
# print(best_result)

# ############################################################
#Principal Component Regression
#Implementing PCA - figuring out the optimal number of principal componenets
# for pc in range(1, 18):
#   pca = PCA(n_components = pc)
#   pca.fit(X_scaled)
#   X_pca = pca.transform(X_scaled)
#   print("Total variance explained by the {} components: ".format(pc),sum(pca.explained_variance_ratio_))

# # # Implementing PCA for optimal number of principal components
# pca = PCA(n_components = 9)
# pca.fit(X_scaled)
# X_pca = pca.transform(X_scaled)

# # Tuning the SGDRegressor parameters 'eta0' (learning rate) and 'max_iter' using Grid Search
# sgdr = SGDRegressor(random_state = 1)
# grid_param = {'eta0': [.0001, .001, .01, .1, 1], 'max_iter':[1000,2000,3000,4000,5000]}

# gd_sr = GridSearchCV(estimator=sgdr, param_grid=grid_param, scoring='r2', cv=10)

# gd_sr.fit(X_pca, Y)

# best_parameters = gd_sr.best_params_
# print(best_parameters)

# best_result = gd_sr.best_score_ # Mean cross-validated score of the best_estimator
# print(best_result)

##Implementing Linear Regression
###Tuning the SGDRegressor parameters 'eta0' (learning rate) and 'max_iter' using Grid Search
sgdr = SGDRegressor(penalty=None, random_state = 1)
grid_param = {'eta0': [.0001, .001, .01, .1, 1], 'max_iter':[1000,2000,3000,4000,5000]}

gd_sr = GridSearchCV(estimator=sgdr, param_grid=grid_param, scoring='r2', cv=10)

gd_sr.fit(X_scaled, Y)

best_parameters = gd_sr.best_params_
print(best_parameters)

best_result = gd_sr.best_score_ # Mean cross-validated score of the best_estimator
print(best_result)

# # Building SGDRegressor using the tuned parameters
# sgdr = SGDRegressor(eta0=.0001, max_iter=1000 , penalty=None, random_state=1)
# sgdr.fit(X_scaled,Y)
# print('Intercept', sgdr.intercept_)
# print(pd.DataFrame(zip(X.columns, sgdr.coef_), columns=['Features','Coefficients']).sort_values(by=['Coefficients'],ascending=False))
# #


###############################################################################
#Implementing L2 Regularization (Ridge Regression)
#Tuning Regularization parameter alpha
# sgdr = SGDRegressor(eta0=.0001, max_iter=1000, penalty='l2', random_state=1)
# grid_param = {'alpha': [.0001, .001, .01, .1, 1,10,50,80,100]}

# gd_sr = GridSearchCV(estimator=sgdr, param_grid=grid_param, scoring='r2', cv=10)

# gd_sr.fit(X_scaled, Y)

# best_parameters = gd_sr.best_params_
# print(best_parameters)

# best_result = gd_sr.best_score_ # Mean cross-validated score of the best_estimator
# print(best_result)

# # Building SGDRegressor using the tuned parameters
# sgdr = SGDRegressor(eta0=.0001, max_iter=1000, penalty='l2', alpha=0.01, random_state=1)
# sgdr.fit(X_scaled,Y)
# print('Intercept', sgdr.intercept_)
# print(pd.DataFrame(zip(X.columns, sgdr.coef_), columns=['Features','Coefficients']).sort_values(by=['Coefficients'],ascending=False))

# # # ################################################################################
# #Implementing L1 Regularization (Lasso Regression)
# #Tuning Regularization parameter alpha
# sgdr = SGDRegressor(eta0=.0001, max_iter=1000, penalty='l1', random_state=1)
# grid_param = {'alpha': [.0001, .001, .01, .1, 1,10,50,80,100]}

# gd_sr = GridSearchCV(estimator=sgdr, param_grid=grid_param, scoring='r2', cv=10)

# gd_sr.fit(X_scaled, Y) 

# best_parameters = gd_sr.best_params_
# print(best_parameters)

# best_result = gd_sr.best_score_ # Mean cross-validated score of the best_estimator
# print(best_result)

# #Building SGDRegressor using the tuned parameters
# sgdr = SGDRegressor(eta0=.0001, max_iter=1000, penalty='l1', alpha=10, random_state=1)
# sgdr.fit(X_scaled,Y)
# print('Intercept', sgdr.intercept_)
# print(pd.DataFrame(zip(X.columns, sgdr.coef_), columns=['Features','Coefficients']).sort_values(by=['Coefficients'],ascending=False))

# ################################################################################
# # # Implementing Elastic Net Regularization (Elastic Net Regression)
# # # Tuning Regularization parameter alpha and l1_ratio
# sgdr = SGDRegressor(eta0=.0001, max_iter=1000, penalty='elasticnet', random_state=1)
# grid_param = {'alpha': [.0001, .001, .01, .1, 1,10,50,80,100],'l1_ratio':[0, 0.1, 0.3,0.5,0.7,0.9,1]}

# gd_sr = GridSearchCV(estimator=sgdr, param_grid=grid_param, scoring='r2', cv=10)
   
# gd_sr.fit(X_scaled, Y)

# best_parameters = gd_sr.best_params_
# print(best_parameters)

# best_result = gd_sr.best_score_ # Mean cross-validated score of the best_estimator
# print(best_result)

# # # Building SGDRegressor using the tuned parameters
# sgdr = SGDRegressor(eta0=.0001, max_iter=1000, penalty='elasticnet', alpha=0.01, l1_ratio=0.1, random_state=1)
# sgdr.fit(X_scaled,Y)
# print('Intercept', sgdr.intercept_)
# print(pd.DataFrame(zip(X.columns, sgdr.coef_), columns=['Features','Coefficients']).sort_values(by=['Coefficients'],ascending=False))

# ###############################################################################
# #Implementing Support Vector Regression
svr = SVR()
grid_param = {'kernel': ['rbf','sigmoid','poly','linear'], 'epsilon':[.001,.01,.1,1], 'C':[.01, .1, 1, 10, 50, 100]}

gd_sr = GridSearchCV(estimator=svr, param_grid=grid_param, scoring='r2', cv=10)

gd_sr.fit(X_scaled, Y)

best_parameters = gd_sr.best_params_
print(best_parameters)

best_result = gd_sr.best_score_ # Mean cross-validated score of the best_estimator
print(best_result)

