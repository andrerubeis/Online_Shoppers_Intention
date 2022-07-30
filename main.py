# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 11:13:38 2022

@author: andre
"""

#%% IMPORTS

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import seaborn as sns

#%% LOAD DATASET

path = r'C:/Users/andre/OneDrive/Documenti/GitHub/Online Shoppers Intention/online_shoppers_intention.csv'
df = pd.read_csv(path)#.sort_values(["ICUSTAY_ID","offset"])

#%% 2. DATASET
categorical_features = ['Month', 'OperatingSystems', 'Browser', 'Region', 
                        'TrafficType', 'VisitorType', 'Weekend', 'Revenue']

descriptions = ['Month of the visit',
                'Operating system of the user',
                'Browser used by the user',
                'Region where the session started by the user',
                'Traffic source',
                'Type of visitor',
                'whter visit occured during weekend or not',
                'class label indicating the occurnece of transaction']

nans = [df.Month.isna().sum(),
                  df.OperatingSystems.isna().sum(),
                  df.Browser.isna().sum(), 
                  df.Region.isna().sum(), 
                  df.TrafficType.isna().sum(),
                  df.VisitorType.isna().sum(),
                  df.Weekend.isna().sum(), 
                  df.Revenue.isna().sum()]

distinct_values =[df.Month.unique(),
                  df.OperatingSystems.unique(),
                  df.Browser.unique(), 
                  df.Region.unique(), 
                  df.TrafficType.unique(),
                  df.VisitorType.unique(),
                  df.Weekend.unique(), 
                  df.Revenue.unique()]

count_values =  [df.Month.nunique(),
                 df.OperatingSystems.nunique(),
                 df.Browser.nunique(), 
                 df.Region.nunique(), 
                 df.TrafficType.nunique(),
                 df.VisitorType.nunique(),
                 df.Weekend.nunique(), 
                 df.Revenue.nunique()]

df_categorical_features_description = pd.DataFrame()

df_categorical_features_description['Feature Name'] = categorical_features
df_categorical_features_description['Description'] = descriptions
df_categorical_features_description['NaNs'] = nans
df_categorical_features_description['Distinct Values'] = distinct_values
df_categorical_features_description['Distinct Count'] = count_values

df_categorical_features_description = df_categorical_features_description.sort_values('Feature Name')

del descriptions, distinct_values, count_values

#Same with numerical features
numerical_features = list(set(df.columns)-set(categorical_features))

descriptions = ['Average page value of the pages visited by the user',
                'n. pages related to the product visited by the visitor',
                '#seconds spent by the visitor on product related pages',
                '#seconds spent by the visitor on account management related pages',
                '#informational pages visited by the visitor',
                'Average exit rate value of the pages visited by the visitor',
                'Average bounce rate value of the pages visited by the visitor',
                '#pages visited by the visitor about account management',
                'Closeness of the site visiting time to a special day',
                '#seconds spent by the visitor on informational pages']
nans = [df.PageValues.isna().sum(),
                  df.ProductRelated.isna().sum(),
                  df.ProductRelated_Duration.isna().sum(), 
                  df.Administrative_Duration.isna().sum(), 
                  df.Informational.isna().sum(),
                  df.ExitRates.isna().sum(),
                  df.BounceRates.isna().sum(), 
                  df.Administrative.isna().sum(),
                  df.SpecialDay.isna().sum(),
                  df.Informational_Duration.isna().sum()]

mins =[df.PageValues.min(),
                  df.ProductRelated.min(),
                  df.ProductRelated_Duration.min(), 
                  df.Administrative_Duration.min(), 
                  df.Informational.min(),
                  df.ExitRates.min(),
                  df.BounceRates.min(), 
                  df.Administrative.min(),
                  df.SpecialDay.min(),
                  df.Informational_Duration.min()]

maxs =[df.PageValues.max(),
                  df.ProductRelated.max(),
                  df.ProductRelated_Duration.max(), 
                  df.Administrative_Duration.max(), 
                  df.Informational.max(),
                  df.ExitRates.max(),
                  df.BounceRates.max(), 
                  df.Administrative.max(),
                  df.SpecialDay.max(),
                  df.Informational_Duration.max()]

means =[df.PageValues.mean(),
                  df.ProductRelated.mean(),
                  df.ProductRelated_Duration.mean(), 
                  df.Administrative_Duration.mean(), 
                  df.Informational.mean(),
                  df.ExitRates.mean(),
                  df.BounceRates.mean(), 
                  df.Administrative.mean(),
                  df.SpecialDay.mean(),
                  df.Informational_Duration.mean()]

stds =[df.PageValues.std(),
                  df.ProductRelated.std(),
                  df.ProductRelated_Duration.std(), 
                  df.Administrative_Duration.std(), 
                  df.Informational.std(),
                  df.ExitRates.std(),
                  df.BounceRates.std(), 
                  df.Administrative.std(),
                  df.SpecialDay.std(),
                  df.Informational_Duration.std()]

count_values =  [df.PageValues.nunique(),
                  df.ProductRelated.nunique(),
                  df.ProductRelated_Duration.nunique(), 
                  df.Administrative_Duration.nunique(), 
                  df.Informational.nunique(),
                  df.ExitRates.nunique(),
                  df.BounceRates.nunique(), 
                  df.Administrative.nunique(),
                  df.SpecialDay.nunique(),
                  df.Informational_Duration.nunique()]

df_numerical_features_description = pd.DataFrame()

df_numerical_features_description['Feature Name'] = numerical_features
df_numerical_features_description['Description'] = descriptions
df_numerical_features_description['NaNs'] = nans
df_numerical_features_description['Min'] = mins
df_numerical_features_description['Max'] = maxs
df_numerical_features_description['Mean'] = means
df_numerical_features_description['Std'] = stds
df_numerical_features_description['Distinct Count'] = count_values

df_numerical_features_description=df_numerical_features_description.sort_values('Feature Name')

#%% 3. DATA EXPLORATION

#We need to perform different data exploration analysis depending on the data 
#type, indeed histograms are more suggested for categorical features and 
# boxplots for numeric features for example
import datetime
import time
df.Weekend = df.Weekend.astype(int)
df.Revenue = df.Revenue.astype(int)
df['Month'].replace('June', 'Jun', inplace=True)
df.Month = sorted(df.Month, key=lambda x: pd.to_datetime(x, format="%b"))

path_save_plots = r'C:\Users\andre\OneDrive\Documenti\DATA SCIENCE ENGINEERING\I ANNO\MATHEMATICS IN MACHINE LEARNING\Project\images\3. Data Exploration\Categorical'

colors = iter(cm.rainbow(np.linspace(0, 2, len(categorical_features))))
sns.color_palette("Paired")
for feature, color in zip(categorical_features, colors):
    plt.figure()
    n_bins = df[feature].nunique()
    sns.histplot(df, x = feature, hue='Revenue', color = color, multiple='stack', bins=n_bins,discrete=True, palette='Paired')
    plt.xticks(range(0,n_bins))
    plt.title(feature)
    plt.savefig(path_save_plots + f"\{feature}_Revenue.png")
#%% BOXPLOTS FOR NUMERICAL FEATURES
numerical_df = df.drop(categorical_features[:-1], axis=1) #numerical df contiene tutte le numerical features più revenue
# numerical_features = numerical_df.columns
# numerical_features = pd.Series(list(set(df.columns)-set(categorical_features)))
# numerical_features = set((df.columns)-(categorical_df.columns))
#sns.boxplot(x = 'Revenue', y = 'Informational', data = df,palette='Paired')

colors = iter(cm.rainbow(np.linspace(0, 2, len(categorical_features))))
path_save_plots_numerical = r'C:\Users\andre\OneDrive\Documenti\DATA SCIENCE ENGINEERING\I ANNO\MATHEMATICS IN MACHINE LEARNING\Project\images\3. Data Exploration\Numerical'
 


sns.set_style("whitegrid")
for feature in numerical_features: #-1 perchè così feature non assume 'Revenue'
    plt.figure()
    sns.boxplot(x= 'Revenue', y = feature, hue='Revenue', data = df,palette='Paired')
    plt.title(feature)
    plt.legend(title='Revenue')
    plt.show()
    plt.savefig(path_save_plots_numerical+f"\{feature}_Revenue.png")
    
#%% CORRELATION MATRIX
matrix = numerical_df.corr()
fig, ax = plt.subplots(figsize=(12,12))
sns.heatmap(matrix, annot=True, ax=ax, vmin=-1, vmax=1,center=0, fmt='.2g',cmap='GnBu')
plt.title("Correlation Matrix")
plt.show()
plt.savefig(path_save_plots_numerical+"\correlation_matrix.png")
#%%
df = df.drop(columns=['Administrative_Duration', 'Informational_Duration','ProductRelated_Duration'], axis = 1)
#%%
sns.relplot(x = 'BounceRates', y = 'ExitRates', hue='Revenue', data=df)
plt.title("Correlation between BounceRates and ExitRates")

#%% DATA CLEANING

df3=df.copy()
df3['Weekend'] = df3['Weekend'].astype('int64')
df3['Revenue'] = df3['Revenue'].astype('int64')

# One hot encoding 
dummy_columns = ['OperatingSystems','Browser','Region','TrafficType','VisitorType', 'Weekend']

for column in dummy_columns:
    df_dummies = pd.get_dummies(df3[column], drop_first = True, prefix = column+"_")    
    df3 = pd.concat([df3, df_dummies], axis = 1)
    
df3 = df3.drop(columns = dummy_columns)

# Accounting for all months in the calendar
months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

for mx in months[1:]:    # drop_first = True
    df3['Month__'+mx] = (df3['Month'] == mx).astype('int64')

df3 = df3.drop(columns = ['Month'])
#%% DATA CLEANING

#Convert categorical features with one-hot encoding
original_df = df.copy()
df2 = original_df.copy()
for column in categorical_features[:-1]: #-1: neglect Revenue
    
    #Create for each value of categorical feature a distinct column
    #ex. Month = [Feb, Mar, ..., Dec] will be Month=Feb, Month=Mar, ..., Month = Dec
    
    df_onehot = pd.get_dummies(df2[column], drop_first=True, prefix=column+'_', prefix_sep= '_')
    
    #Cocat together the one_hot df just created with original one
    df2 = pd.concat([df2, df_onehot], axis = 1)
    
    #Remove features that are no no longer needed
    #ex. since we now have Month=Feb, Month = Mar, ..., Month = Dec we can drop Month column
    df2 = df2.drop(column, axis = 1)
    

#Remove highly correlated features to reduce the #dimensions
#Handle missing data
#Handle outliers
