# import sys
# print(sys.version)
import pandas as pd
import numpy as np
import xgboost
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.cross_validation import StratifiedKFold, cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn import naive_bayes as nb
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

import os
os.chdir('/Users/oneinchman/Documents/Git-repositories/Kaggle_scripts/Titanic challenge')

# train data loading
data_train = pd.read_csv('train.csv')
data_train.drop(['PassengerId', 'Ticket', 'Cabin'], axis=1, inplace=True)

# train preprocessing
data_train['Sex_mapped'] = data_train['Sex'].map({'male': 0, 'female': 1})
AgeMedians = data_train.pivot_table('Age', index=["Sex", "Pclass"], aggfunc='median')
data_train['Age_filled'] = data_train[['Sex',
                                       'Age',
                                       'Pclass']].apply(lambda x:
                                                        AgeMedians[x.Sex, x.Pclass]
                                                        if pd.isnull(x.Age) else

                                                        x.Age, axis=1)
# train encoding
enc = OneHotEncoder(sparse=False)
sex_train_encoded = enc.fit_transform(data_train['Sex_mapped'].values.reshape((-1, 1)))
train = np.hstack((data_train[['Pclass', 'Fare', 'Age_filled']].values, sex_train_encoded))
target = data_train['Survived'].values
sc = StandardScaler()
train_sc = sc.fit_transform(train)

# test loading, preprocessing and encoding
data_test = pd.read_csv('test.csv')
data_test.drop(['PassengerId', 'Ticket', 'Cabin'], axis=1, inplace=True)
data_test['Sex_mapped'] = data_test['Sex'].map({'male': 0, 'female': 1})
data_test['Fare'].fillna(value=data_train['Fare'].mean(), inplace=True)
data_test['Age_filled'] = data_test[['Sex', 'Pclass', 'Age']].apply(lambda x: AgeMedians[x.Sex, x.Pclass]
                                                                    if pd.isnull(x.Age) else x.Age, axis=1)
sex_test_encoded = enc.transform(data_test['Sex_mapped'].values.reshape((-1, 1)))

# modeling
cross_validator = StratifiedKFold(y=target, n_folds=10, shuffle=True, random_state=0)
param_grids = [{'C': [10**(x) for x in range(-5, 2)], 'class_weight': [None, 'balanced']},
               {'n_estimators': [100, 500, 800, 1000], 'max_depth': [2, 5, 8, None],
                'class_weight': [None, 'balanced']},
               {'C': [1 / x for x in range(1, 11)], 'class_weight': [None, 'balanced']},
               {'max_depth': [3, 4, 5, 6], 'learning_rate': [0.01, 0.1],
                'n_estimators': [100, 500, 800, 1000],
                'reg_lambda': [10**(x) for x in range(-5, 2)]}]
models = [LogisticRegression(), RandomForestClassifier(), SVC(), xgboost.XGBClassifier()]
models_names = ['Logistic Regression', 'Random Forest', 'SVC', 'Xgbost']
best_score = 0
for clf, params, name in zip(models, param_grids, models_names):
    gs = GridSearchCV(estimator=clf, param_grid=params, scoring='accuracy', cv=cross_validator)
    gs.fit(train_sc, target)
    print('For {} the score is {} with parameters {}'.format(name, gs.best_score_, gs.best_params_))
    if gs.best_score_ > best_score:
        best_score = gs.best_score_
        best_estimator = gs.best_estimator_

# clf_test = np.hstack((data_test[['Pclass', 'Fare', 'Age_filled']].values, sex_test_encoded))
# predictions = best_estimator.predict(clf_test)
# output_df = pd.DataFrame(data={'PassengerId': np.arange(892, 892 + data_test.shape[0]), 'Survived': predictions})
# output_df.to_csv('model1.csv', index=False)
