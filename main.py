#%% md

# Online Shoppers Intention

## 1. Problem Statement
# The aim of this project is to predict study the consumer's behaviour
# about the possibility to

#TODO: # (NOT) Accounting for all months in the calendar except Jan and Apr not present (modifica poi per vedere che cambia se consideri tutti)

#%% IMPORTS

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from IPython.core.display import display
from matplotlib.pyplot import cm
import seaborn as sns

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

#%% LOAD DATASET

path = r'C:/Users/andre/OneDrive/Documenti/GitHub/Online Shoppers Intention/online_shoppers_intention.csv'
df = pd.read_csv(path)#.sort_values(["ICUSTAY_ID","offset"])
df_copy = df.copy()
# #%% md

# ## 2. Dataset
# The dataset includes feature vectors from 12,330 sessions.

# The dataset was created so that each session would be associated with a different user throughout the course of a year in order to prevent any inclination to a particular campaign, noteworthy day, user profile, or time frame.

# The dataset is made by **10 numerical features** and **8 categorical features**.

# **Attributes Information**

# * **Administrative**: This is the number of pages of this type (administrative) that the user visited.
# * **Administrative Duration**: This is the amount of time spent in this category of pages.

# * **Informational**: This is the number of pages of this type (informational) that the user visited.
# * **Informational Duration**: This is the amount of time spent in this category of pages.
# * **Product Related**: This is the number of pages of this type (product related) that the user visited.
# * **Product Related Duration** : This is the amount of time spent in this category of pages.

# * **Bounce Rate**: The percentage of visitors who enter the website through that page and exit without triggering any additional tasks.
# * **Exit Rate**: The percentage of pageviews on the website that end at that specific page.
# * **Page Value**: The average value of the page averaged over the value of the target page and/or the completion of an eCommerce transaction.
# * **Special Day**: indicates the closeness of the site visiting time to a specific
# special day (e.g. Mother’s Day, Valentine's Day) in which the sessions are more
# likely to be finalized with transaction.
# * **Month**: Contains the month the pageview occurred, in string form.
# * **Operating system**: An integer value representing the operating system that the user was on when viewing the page.
# * **Browser**: An integer value representing the browser that the user was using to view the page.
# * **Region**: An integer value representing which region the user is located in.
# * **Traffic Type**: An integer value representing what type of traffic the user is categorized into.
# * **Visitor Type**: A string representing whether a visitor is New Visitor, Returning Visitor, or Other.
# * **Weekend**: A boolean representing whether the session is on a weekend.
# * **Revenue**: A boolean representing whether or not the user completed the purchase.

#%% md

### Categorical Features

#%% Categorical Features

categorical_features = ['Month', 'OperatingSystems', 'Browser', 'Region',
                        'TrafficType', 'VisitorType', 'Weekend', 'Revenue']

descriptions = ['string indicating month the pageview occurred.',
                'integer value that represents the user’s operating system at the time the page was viewed.',
                'integer value that represents the user’s browser at the time the page was viewed.',
                'region of the user is indicated by an integer value.',
                'category of traffic the user falls under is represented by an integer value.',
                'string indicating whether a visitor is a New Visitor, a Returning Visitor, or Other',
                'boolean value indicating whether or not the session is on a weekend',
                'class label, boolean indicating whether or not the user completed the purchase.']

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
df_categorical_features_description.style.set_properties(**{'text-align': 'left'})
display(df_categorical_features_description.style.hide_index())
#%% md

### Numerical Features

#%% Numerical Features

numerical_features = list(set(df.columns)-set(categorical_features))

descriptions = ['average page value over the value of the target page and/or the successful completion of an online purchase',
                'how many product related pages the user accessed',
                '#seconds spent on product related pages.',
                '#seconds spent on administrative pages',
                'how many pages of informational type the user accessed',
                'percentage of website pageviews actually end on that particular page',
                'proportion of users that arrive on that page of the website and leave without performing any further actions',
                'how many pages of administrative type the user accessed.',
                'closeness of the site visiting time to a special day',
                '#seconds spent on informational pages.']
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
df_numerical_features_description.style.set_properties(**{'text-align': 'left'})
display(df_numerical_features_description.style.hide_index())
#print(df_numerical_features_description.to_string(index=False))
#%% md

## 3. Data Exploration

#Let's see what are the distinct values assumed by each feature
df.Administrative.hist()

#%%

print("First 5 rows of the dataset")
df[50:55]

#%%

print("Dataset information")
df.info()
print("")

#%% md

### Countplots Categorical Features

#%% 3. DATA EXPLORATION

#We need to perform different data exploration analysis depending on the data
#type, indeed histograms are more suggested for categorical features and
# boxplots for numeric features for example
import datetime
import time


path_save_plots = r'C:\Users\andre\OneDrive\Documenti\DATA SCIENCE ENGINEERING\I ANNO\MATHEMATICS IN MACHINE LEARNING\Project\images\3. Data Exploration\Categorical'

# colors = iter(cm.rainbow(np.linspace(0, 2, len(categorical_features))))
# sns.color_palette("Paired")
# for feature, color in zip(categorical_features, colors):
#     plt.figure()
#     n_bins = df[feature].nunique()
#     sns.histplot(df, x = feature, hue='Revenue', color = color, multiple='stack', bins=n_bins,discrete=True, palette='Paired')
#     plt.xticks(range(0,n_bins))
#     plt.title(feature)
#     plt.savefig(path_save_plots + f"\{feature}_Revenue.png")

#%% md

### BoxPlots Numerical Features

#%% BOXPLOTS FOR NUMERICAL FEATURES

numerical_df = df.drop(categorical_features[:-1], axis=1) #numerical df contiene tutte le numerical features più revenue
# numerical_features = numerical_df.columns
# numerical_features = pd.Series(list(set(df.columns)-set(categorical_features)))
# numerical_features = set((df.columns)-(categorical_df.columns))
#sns.boxplot(x = 'Revenue', y = 'Informational', data = df,palette='Paired')

colors = iter(cm.rainbow(np.linspace(0, 2, len(categorical_features))))
path_save_plots_numerical = r'C:\Users\andre\OneDrive\Documenti\DATA SCIENCE ENGINEERING\I ANNO\MATHEMATICS IN MACHINE LEARNING\Project\images\3. Data Exploration\Numerical'

# sns.set_style("whitegrid")
# for feature in numerical_features: #-1 perchè così feature non assume 'Revenue'
#     plt.figure()
#     sns.boxplot(x= 'Revenue', y = feature, hue='Revenue', data = df,palette='Paired')
#     plt.title(feature)
#     plt.legend(title='Revenue')
#     plt.show()
#     plt.savefig(path_save_plots_numerical+f"\{feature}_Revenue.png")

#%% DISTRIBUTION OF NUMERICAL FEATURES
# for feature in numerical_features:
#     print(df[feature].value_counts())
#     sns.displot(df, x=feature, kind="kde", bw_adjust=2)
#     plt.title(feature+" distribution")

#%%
numerical_features = list(set(df.columns)-set(categorical_features))
numerical_features.append('Revenue') #aggiungo 'Revenue' in numerical features solo per fare la correlation matrix
df_numerical = df[numerical_features]
df_numerical.Revenue = df_numerical.Revenue.astype(int)
#sns.pairplot(df_numerical, hue='Revenue', palette = 'Paired')

#%% SCATTER PLOTS

# sns.relplot(x = 'Administrative', y = 'Administrative_Duration', hue='Revenue', data=df, palette = 'Paired')
# plt.title("Correlation between Administrative_Duration and Administrative")

# sns.relplot(x = 'Informational', y = 'Informational_Duration', hue='Revenue', data=df, palette = 'Paired')
# plt.title("Correlation between Informational_Duration and Informational")

# sns.relplot(x = 'ProductRelated', y = 'ProductRelated_Duration', hue='Revenue', data=df, palette = 'Paired')
# plt.title("Correlation between ProductRelated_Duration and ProductRelated")

# sns.relplot(x = 'ExitRates', y = 'BounceRates', hue='Revenue', data=df, palette = 'Paired')
# plt.title("Correlation between ExitRates and BounceRates")

#%% md

### Correlation Matrix

#%% CORRELATION MATRIX

# matrix = numerical_df.corr()
# fig, ax = plt.subplots(figsize=(12,12))
# sns.heatmap(matrix, annot=True, ax=ax, vmin=-1, vmax=1,center=0, fmt='.2g',cmap='GnBu')
# plt.title("Correlation Matrix")
# plt.savefig(path_save_plots_numerical+"\correlation_matrix.png")
# plt.show()

#%%

# sns.relplot(x = 'BounceRates', y = 'ExitRates', hue='Revenue', data=df, palette = 'Paired')
# plt.title("Correlation between BounceRates and ExitRates")


#%% md

## 4. Data Cleaning

#%% DATA CLEANING
df = df_copy.copy()
df.Weekend = df.Weekend.astype(int)
df.Revenue = df.Revenue.astype(int)
df['Month'].replace('June', 'Jun', inplace=True)
df.Month = sorted(df.Month, key=lambda x: pd.to_datetime(x, format="%b"))

#%% DROP CORRELATED FEATURES TO REDUCE DATASET DIMENSIONALITY

features_dropped = ['BounceRates','Administrative_Duration', 'Informational_Duration','ProductRelated_Duration']
numerical_features = list(set(numerical_features)-set(features_dropped))
numerical_features.remove('Revenue')
df = df.drop(columns=features_dropped, axis = 1)
# One hot encoding
dummy_columns = ['OperatingSystems','Browser','Region','TrafficType','VisitorType', 'Weekend']

for column in categorical_features[1:-1]: #non considero Month nel ciclo perchè lo gestico a parte (1) e mi fermo prima di considerare Revenue (-1)
    df_dummies = pd.get_dummies(df[column], drop_first = True, prefix = column)
    df = pd.concat([df, df_dummies], axis = 1)



# (NOT) Accounting for all months in the calendar except Jan and Apr not present (modifica poi per vedere che cambia se consideri tutti)
months = ['Feb','Mar','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

for mx in months[1:]:    # drop_first = True
    df['Month_'+mx] = (df['Month'] == mx).astype('int64')

df = df.drop(columns = categorical_features[:-1]) #droppo tutte le categorical features tranne Revenue

#%% DATA CLEANING

#Convert categorical features with one-hot encoding
# tmp = df_copy.copy()
# tmp = tmp.drop(columns=['BounceRates','Administrative_Duration', 'Informational_Duration','ProductRelated_Duration'], axis = 1)

# original_df = tmp.copy()
# df2 = original_df.copy()

# df2['Weekend'] = df2['Weekend'].astype('int64')
# df2['Revenue'] = df2['Revenue'].astype('int64')
# df2['Month'].replace('June', 'Jun', inplace=True)
# df2.Month = sorted(df2.Month, key=lambda x: pd.to_datetime(x, format="%b"))

# for column in categorical_features[:-1]: #-1: neglect Revenue

#     #Create for each value of categorical feature a distinct column
#     #ex. Month = [Feb, Mar, ..., Dec] will be Month=Feb, Month=Mar, ..., Month = Dec

#     df_onehot = pd.get_dummies(df2[column], drop_first=True, prefix=column)
#     #df2 = pd.get_dummies(df2, drop_first=True, prefix=column)
    
#     #Cocat together the one_hot df just created with original one
#     df2 = pd.concat([df2, df_onehot], axis = 1)

#     #Remove features that are no no longer needed
#     #ex. since we now have Month=Feb, Month = Mar, ..., Month = Dec we can drop Month column
#     df2 = df2.drop(columns=column, axis = 1)


# #Remove highly correlated features to reduce the #dimensions
# #Handle missing data
# #Handle outliers

#%% STANDADIZATION
import sklearn
from sklearn.preprocessing import StandardScaler


ss = StandardScaler()





numerical_df = numerical_df[numerical_features]

scaled_numerical_df =  pd.DataFrame(ss.fit_transform(numerical_df), columns=numerical_df.columns)

categorical_features = list(set(df.columns)-set(numerical_features))
scaled_df = pd.concat([df[categorical_features], scaled_numerical_df], axis=1)

y = scaled_df['Revenue'].copy()
X = scaled_df.drop('Revenue',axis=1)

#Replace 0 values in categorical features with -1 to have mean = 0
scaled_df[categorical_features] = scaled_df[categorical_features].mask(scaled_df[categorical_features] == 0, -1)

# features_to_add = [e for e in numerical_features if e not in features_dropped]
# scaled_df_categorical = df.drop(columns=features_to_add, axis=1)
# scaled_df = pd.concat([scaled_numerical_df, scaled_df_categorical], axis = 1)
#%% PCA
from sklearn.decomposition import PCA
PCA_df = PCA().fit(scaled_df)
pca_df = PCA_df.transform( scaled_df)

cumvar = np.cumsum(PCA_df.explained_variance_ratio_)

n_comp =  np.argmax(cumvar > .9)

#Plotting cumulative variance
plt.plot(cumvar)
plt.title('Cumulative variance')
plt.xlabel('Number of components')
plt.ylabel('Variance explained')
print(f"{cumvar[n_comp]} expressed by {n_comp} components")

#%%
X_pca = np.dot(scaled_df, PCA_df.components_[:n_comp,:].T)
X_pca = pd.DataFrame(X_pca, columns=["PC%d" % (x + 1) for x in range(n_comp)])
X_pca.shape


#%% SMOTE TO BALANCE LABEL
import imblearn
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RepeatedStratifiedKFold, StratifiedKFold

seed = 13
sm = SMOTE(random_state = seed)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

X_train_smote, y_train_smote = sm.fit_sample(X_train, y_train.ravel())
#pd.Series(y_train_smote).value_counts().plot.bar()
y_train_smote = pd.Series(y_train_smote)
#%% MODELS

from sklearn.model_selection import (
    KFold,
    ShuffleSplit,
    StratifiedKFold,
    GroupShuffleSplit,
    GroupKFold,
    StratifiedShuffleSplit,
    StratifiedGroupKFold,
)
rng = np.random.RandomState(1338)
cmap_data = plt.cm.Paired
cmap_cv = plt.cm.coolwarm
n_splits = 4



#%%
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix,roc_curve, roc_auc_score, precision_score, recall_score, precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
# define model
lr = LogisticRegression(random_state=seed, class_weight=None)
accuracies=[]
def plot_cv_indices(cv, X, y, group, ax, n_splits, lw=10):
    """Create a sample plot for indices of a cross-validation object."""

    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y, groups=group)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0
        
        lg1.fit(X.iloc[tr], y.iloc[tr])
        y_pred = lg1.predict(X.iloc[tt])
        
        print(f'Accuracy Score: {accuracy_score(y.iloc[tt],y_pred)}')
        accuracies.append(accuracy_score(y.iloc[tt],y_pred))
        # Visualize the results
        ax.scatter(
            range(len(indices)),
            [ii + 0.5] * len(indices),
            c=indices,
            marker="_",
            lw=lw,
            cmap=cmap_cv,
            vmin=-0.2,
            vmax=1.2,
        )

    # Plot the data classes and groups at the end
    ax.scatter(
        range(len(X)), [ii + 1.5] * len(X), c=y, marker="_", lw=lw, cmap=cmap_data
    )


    # Formatting
    yticklabels = list(range(n_splits)) + ["class"]
    ax.set(
        yticks=np.arange(n_splits + 1) + 0.5,
        yticklabels=yticklabels,
        xlabel="Sample index",
        ylabel="CV iteration",
        ylim=[n_splits + 1.2, -0.2],
        xlim=[0, X.shape[0]],
    )
    ax.set_title("{}".format(type(cv).__name__), fontsize=15)
    return ax


fig, ax = plt.subplots()
n_splits=8
cv = StratifiedKFold(n_splits, shuffle=False)
X_train_smote, y_train_smote = shuffle(X_train_smote, y_train_smote)
plot_cv_indices(cv, X_train_smote, y_train_smote, y_train_smote, ax, n_splits)



#%% MODELS HYPERPARAMETERS

models = {}
models['LR'] = {}
models['LR']['hyperparams'] = {}

models['LR']['hyperparams']['C'] = [100, 10, 1.0, 0.1, 0.01]
models['LR']['hyperparams']['solvers'] = ['newton-cg', 'lbfgs', 'liblinear']

#%%

# example of grid searching key hyperparametres for logistic regression
from sklearn.datasets import make_blobs
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

# define models and parameters
model = LogisticRegression()
solvers = ['newton-cg', 'lbfgs', 'liblinear']
penalty = ['l2']
c_values = [100, 10, 1.0, 0.1, 0.01]

# define grid search
grid = dict(solver=solvers,penalty=penalty,C=c_values)
#cv = RepeatedStratifiedKFold(n_splits=8, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X_train_smote, y_train_smote)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


#%% LOGISTIC REGRESSION

# import model and matrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix,roc_curve, roc_auc_score, precision_score, recall_score, precision_recall_curve
from sklearn.metrics import f1_score

seed = 13

# split dataset into x,y
x = scaled_df.drop('Revenue',axis=1)
y = scaled_df['Revenue']
# train-test split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=seed)
pd.Series(y_train).value_counts().plot.bar()

#%%

# define model
lg1 = LogisticRegression(random_state=seed, class_weight=None)
# fit it
lg1.fit(X_train,y_train)
# test
y_pred = lg1.predict(X_test)
# performance
print(f'Accuracy Score: {accuracy_score(y_test,y_pred)}')
print(f'Confusion Matrix: \n{confusion_matrix(y_test, y_pred)}')
print(f'Area Under Curve: {roc_auc_score(y_test, y_pred)}')
print(f'Recall score: {recall_score(y_test,y_pred)}')

#%%
nbm = GaussianNB()
nbm.fit(X_train,y_train)
nbm_pred = nbm.predict(X_test)

print('Gaussian Naive Bayes Performance:')
print('---------------------------------')
print('Accuracy        : ', accuracy_score(y_test, nbm_pred))
print('F1 Score        : ', f1_score(y_test, nbm_pred))
print('Precision       : ', precision_score(y_test, nbm_pred))
print('Recall          : ', recall_score(y_test, nbm_pred))
print('Confusion Matrix:\n ', confusion_matrix(y_test, nbm_pred))
#%%
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
# 5 folds selected
kfold = KFold(n_splits=10, random_state=0, shuffle=True)
model = LogisticRegression(solver='liblinear')
results = cross_val_score(model, X_train, y_train, cv=kfold)
# Output the accuracy. Calculate the mean and std across all folds. 
print(f"Accuracy - mean : {results.mean()*100.0:.3f}, std: {results.std()*100:.3f}")



#%%

models = {}
#models['LR'] 



lrm_param_grid = {'C': [0.01, 0.1, 1, 10, 100],  
              'solver': ['newton-cg', 'lbfgs', 'sag', 'saga']} 
lrm_grid = GridSearchCV(LogisticRegression(),
                        lrm_param_grid,
                        refit=True,
                        verbose=3)




































