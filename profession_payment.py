import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Read the data sets
train = pd.read_csv("adult.data.txt", header=None, decimal=".", sep=",", na_values="?")
test = pd.read_csv("adult.test.txt", header=None, decimal=".", sep=",", na_values="?", skiprows=1)

# Set proper column names
col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
             'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
             'target']
train.columns = col_names
test.columns = col_names

# Drop the ordinal column 'education', we will use the continuous version of the column.
train = train.drop(['education'], axis=1)
test = test.drop(['education'], axis=1)

# Drop all rows that have NA's.

train_data = train.dropna(axis=0, how='any')
test_data = test.dropna(axis=0, how='any')

# Replace the '>50K' and '<=50K' labels with 1 and 0 integers
train = train.replace('>50k', 1)
train = train.replace('<=50k', 0)

test = test.replace('>50k.', 1)
test = test.replace('<=50k.', 0)

# split target

train_target = train.iloc[:, -1]
test_target = test.iloc[:, -1]

# Remove the last column (the target) of the data set
train_data = train.iloc[:, 0:13]
test_data = test.iloc[:, 0:13]

# Turn categorical values into dummies.
train_data = pd.get_dummies(train_data)
test_data = pd.get_dummies(test_data)

train_data = train_data.drop(
    ['workclass_Without-pay', 'marital-status_Never-married', 'occupation_Farming-fishing', 'relationship_Husband',
     'race_White', 'sex_Male', 'native-country_United-States'], axis=1)

test_data = test_data.drop(
    ['workclass_Without-pay', 'marital-status_Never-married', 'occupation_Farming-fishing', 'relationship_Husband',
     'race_White', 'sex_Male', 'native-country_United-States'], axis=1)

#  Are there columns which are not in one or the other data set?
drop_columns = list(set(list(train_data.columns.values)) - set(list(test_data.columns.values)))

train_data = train_data.drop(drop_columns, axis=1, errors='ignore')
test_data = test_data.drop(drop_columns, axis=1, errors='ignore')

# On logistic regression

fig = sns.lmplot(x='age', y='target', data=train, y_jitter=0.01, x_jitter=0.15, logistic=True, scatter_kws={
    'alpha': 0.05
}, line_kws={'color': 'red'})

fig.fig.set_size_inches(10, 6)
fig.savefig('fig1.png')

logistic_reg = sm.Logit(train_target, sm.add_constant(train_target.age))
result = logistic_reg.fit()
print(result.summary())

# Interpretation of the coefficient

print((np.exp(-2.744) + np.exp(0.0395 * 41)) / (np.exp(-2.744) + np.exp(0.0395 * 40)))

# Convert odds to probability
1 / (1 + np.exp(-(-2.7440 + 0.0395 * 40)))

# For scikit, we need matrices

full_col_names = list(train_data.columns.values)  # Store column names in a variable
test_data = test_data.as_matrix()
train_data = train_data.as_matrix()
test_target = test_target.as_matrix()
train_target = train_target.as_matrix()

# Set values of the grid search
C_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
C_grid = {
    'C': C_values
}
# Set the amount of folds for the cross-validation

n_folds = 5
# Do a model fit over a grid of C hyperparameters

logReg = LogisticRegression(penalty='l1', random_state=7)
grid_logReg = GridSearchCV(logReg, C_grid, cv=n_folds, refit=True)
grid_logReg.fit(train_data, train_target)

# Visualize maximum accuracy

plt.figure().set_size_inches(10, 6)
fig2 = plt.semilogx(C_values, grid_logReg.cv_results_['mean_test_score'])
plt.savefig('figure2.png')

# Run the model on the test data

best_logReg = grid_logReg.best_estimator_

print(best_logReg.score(test_data, test_target))

# Get the models coefficients (and top 5 and bottom 5)

logReg_coeff = pd.DataFrame({
    'feature_name': full_col_names,
    'model_coefficient': best_logReg.coef_.transpose().flatten()
})

logReg_coeff = logReg_coeff.sort_values('model_coefficient', ascending=False)
logReg_coeff_top = logReg_coeff.head(5)
logReg_coeff_bottom = logReg_coeff.tail(5)

# Plot top 5 coefficients

plt.figure().set_size_inches(10, 6)
fig3 = sns.barplot(x='feature_name', y='model_coefficient', data=logReg_coeff_top, palette='Blues_d')

fig3.set_xticklabels(rotation=35, labels=logReg_coeff_top.feature_name)

# Plot bottom 5 coefficients

plt.figure().set_size_inches(10, 6)
fig4 = sns.barplot(x='feature_name', y='model_coefficient', data=logReg_coeff_bottom, palette="GnBu_d")
fig4.set_xticklabels(rotation=35, labels=logReg_coeff_bottom.feature_name)

plt.xlabel('Feature')
plt.ylabel('Coefficient')

plt.subplots_adjust(bottom=0.4)
plt.savefig('fig4.png')
