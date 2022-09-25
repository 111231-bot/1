import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from matplotlib.pyplot import MultipleLocator
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split, StratifiedShuffleSplit
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV, LassoLarsCV, ElasticNet, ElasticNetCV
from sklearn.model_selection import train_test_split,cross_val_score,KFold,RepeatedKFold
from scipy.stats import pearsonr, ttest_ind, levene
from sklearn.metrics import recall_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier,VotingClassifier,BaggingClassifier,ExtraTreesClassifier,GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import mutual_info_classif,SelectKBest
from sklearn import svm
import pingouin as pg
from sklearn.svm import LinearSVC
import os
import warnings
warnings.filterwarnings('ignore')
def lr(X_train, y_train):
    lr = LR()
    C= np.logspace(-3,2,8)
    param_grid = dict(C = C)
    grid_search = GridSearchCV(lr, param_grid, n_jobs=-1, scoring='roc_auc',cv=5)
    grid_search.fit(X_train, y_train)
    lr = LR(penalty = 'l1', C = grid_search.best_params_["C"], solver='saga', class_weight='balanced').fit(X_train, y_train)
    return lr

def svm(X_train, y_train):
    param_grid = {
            'C': [0.0001,0.001, 0.01, 0.1, 1, 10, 100],
            'gamma' : [0.0001,0.001, 0.01, 0.1, 1, 2, 5, 10, 20],
        }
    grid = GridSearchCV(svm.SVC(kernel='linear'), param_grid=param_grid, scoring='roc_auc', cv=5).fit(X_train, y_train)
    C = grid.best_params_['C']
    gamma = grid.best_params_['gamma']
    model_svm = svm.SVC(kernel='linear', C=C, gamma=gamma, probability=True).fit(X_train, y_train)
    return model_svm

def rf(X_train, y_train):
    model_forest = RandomForestClassifier()
    param_grid = {"n_estimators": [10,100,150,250, 300,350,400,450],
          "criterion": ["gini", "entropy"],
          "max_features": [1,3,4,5,6],
          "max_depth": [1,3,5,7,9, 10],
          "min_samples_split": [2, 4, 6]
              }
    grid_search = GridSearchCV(model_forest, param_grid, n_jobs=-1, scoring='accuracy', cv=5)
    grid_search.fit(X_train, y_train)
    model_forest = ExtraTreesClassifier(n_estimators=grid_search.best_params_["n_estimators"],
                                          criterion=grid_search.best_params_["criterion"],
                                          max_features=grid_search.best_params_["max_features"],
                                          max_depth=grid_search.best_params_["max_depth"],
                                          min_samples_split=grid_search.best_params_["min_samples_split"],
                                          ).fit(X_train, y_train)
    return model_forest

file_path_0 = "path0.csv"
file_path_1 = "path1.csv"
file_path_2 = "path2.csv"
file_path_3 = "path3.csv"
file_path_4 = "path4.csv"


data_icc_t2 = "t2.csv"
data_icc_ep = "ep.csv"
data_icc_t1 = "t1.csv"
data_icc_dwi = "dwi.csv"
data_icc_adc = "adc.csv"
data_icc_np = "np.csv"
data_icc_cp = "cp.csv"
columns_icc_t2 = data_icc_t2.columns[2:]
columns_icc_ep = data_icc_ep.columns[2:]
columns_icc_t1 = data_icc_t1.columns[2:]
columns_icc_dwi = data_icc_dwi.columns[2:]
columns_icc_adc = data_icc_adc.columns[2:]
columns_icc_np = data_icc_np.columns[2:]
columns_icc_cp = data_icc_cp.columns[2:]

index_icc = []
iii = []
#
for i in range(0,len(columns_icc_t2)):
    icc = pg.intraclass_corr(data = data_icc_t2, targets = "target", raters = "reader",ratings = columns_icc_t2[i])
    # print(icc.loc[2][0],icc.loc[2][2])
    if (icc.loc[1][2] > 0.85):
        index_icc.append(columns_icc_t2[i])
        iii.append(icc.loc[1][2])


for i in range(0,len(columns_icc_ep)):
    icc = pg.intraclass_corr(data = data_icc_ep, targets = "target", raters = "reader",ratings = columns_icc_ep[i])
    # print(icc.loc[2][0],icc.loc[2][2])
    if (icc.loc[1][2] > 0.85):
        index_icc.append(columns_icc_ep[i])
        iii.append(icc.loc[1][2])

for i in range(0,len(columns_icc_t1)):
    icc = pg.intraclass_corr(data = data_icc_t1, targets = "target", raters = "reader",ratings = columns_icc_t1[i])
    # print(icc.loc[2][0],icc.loc[2][2])
    if (icc.loc[1][2] > 0.85):
        index_icc.append(columns_icc_t1[i])
        iii.append(icc.loc[1][2])

for i in range(0,len(columns_icc_dwi)):
    icc = pg.intraclass_corr(data = data_icc_dwi, targets = "target", raters = "reader",ratings = columns_icc_dwi[i])
    # print(icc.loc[2][0],icc.loc[2][2])
    if (icc.loc[1][2] > 0.85):
        index_icc.append(columns_icc_dwi[i])
        iii.append(icc.loc[1][2])
# #
for i in range(0,len(columns_icc_adc)):
    icc = pg.intraclass_corr(data = data_icc_adc, targets = "target", raters = "reader",ratings = columns_icc_adc[i])
    # print(icc.loc[2][0],icc.loc[2][2])
    if (icc.loc[1][2] > 0.85):
        index_icc.append(columns_icc_adc[i])
        iii.append(icc.loc[1][2])

for i in range(0,len(columns_icc_np)):
    icc = pg.intraclass_corr(data = data_icc_np, targets = "target", raters = "reader",ratings = columns_icc_np[i])
    # print(icc.loc[2][0],icc.loc[2][2])
    if (icc.loc[1][2] > 0.85):
        index_icc.append(columns_icc_np[i])
        iii.append(icc.loc[1][2])
#
for i in range(0,len(columns_icc_cp)):
    icc = pg.intraclass_corr(data = data_icc_cp, targets = "target", raters = "reader",ratings = columns_icc_cp[i])
    # print(icc.loc[2][0],icc.loc[2][2])
    if (icc.loc[1][2] > 0.85):
        index_icc.append(columns_icc_cp[i])
        iii.append(icc.loc[1][2])
iii.sort()

data_0 = pd.read_csv(file_path_0)
data_1 = pd.read_csv(file_path_1)
data_2 = pd.read_csv(file_path_2)
data_train = pd.concat([data_0,data_1,data_2])


data_3 = pd.read_csv(file_path_3)
data_4 = pd.read_csv(file_path_4)
data_outertest = pd.concat([data_3,data_4])


data_all = pd.concat([data_train,data_outertest])
x_train = data_all.iloc[0:data_train.shape[0],2:]
y_train = data_all.iloc[0:data_train.shape[0],1]
x_train = x_train.values
y_train = y_train.values

x_testouter = data_all.iloc[data_train.shape[0]:data_train.shape[0]+data_outertest.shape[0],2:]
y_testouter = data_all.iloc[data_train.shape[0]:data_train.shape[0]+data_outertest.shape[0],1]

colnames = data_all.columns[2:]
x_train = x_train.astype(np.float64)
x_testouter = x_testouter.astype(np.float64)


ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_testouter = ss.transform(x_testouter)

X_train = pd.DataFrame(x_train)
X_train.columns = colnames



X_testouter = pd.DataFrame(x_testouter)
X_testouter.columns = colnames
X_train = X_train[index_icc]
X_testouter = X_testouter[index_icc]


alphas = np.logspace(-3,3,50)#(-3,1,50)
model_lasso = LassoCV(alphas = alphas, cv = 5, max_iter = 100000).fit(X_train,y_train)

#
coef = pd.Series(model_lasso.coef_, index = X_train.columns)
coef = coef.sort_values(ascending=False)
#
index = coef[coef!=0].index
X_train = X_train[index]
X_testouter = X_testouter[index]


lr=lr(X_train, y_train)

y_pred = (lr.predict_proba(X_testouter)[:,1] >= 0.5).astype(bool)

test_data_confusion_matrix = confusion_matrix(y_testouter, y_pred)
M = test_data_confusion_matrix
score = (M[0,0]+M[1,1])/len(y_testouter)
sensitity1 = M[1, 1] / (M[1, 1] + M[1, 0])
specificity1 = M[0, 0] / (M[0, 1] + M[0, 0])
rf_probe1 = lr.predict_proba(X_testouter)[:, 1]
auc1 = roc_auc_score(y_testouter, rf_probe1)



