import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
#读入数据
housing1 = pd.read_csv('train.csv')
housing2 = pd.read_csv('test.csv')
housing = pd.concat([housing1,housing2],axis=0,ignore_index=True)
##数据处理
##1. 填充数据，文本型数据用NONE填充，数字型数据利用中位数冲入

for i in range(1,housing.shape[1]):
    if(np.any(housing.iloc[:, i].isnull())):                                         ##判断一列中是否有空
        if(type(np.any(housing.iloc[:,i]))== str):                                   ##判断有空的这一列是字符型还是数值型
            housing.iloc[:, i] = housing.iloc[:, i].fillna(value='None')             ##将字符空值写入字符串None
        else:
            housing.iloc[:, i]=housing.iloc[:, i].fillna(value=0)                    ##将数组空值写入0



##2.1.2 独热编码
housing = pd.get_dummies(housing, columns=['MSZoning','Street','Alley','Neighborhood',
                                           'RoofStyle','RoofMatl','Exterior1st','Exterior2nd',
                                           'Heating','MiscFeature','SaleCondition'],prefix='_',dummy_na=False,drop_first=False)
##2.2 程度编码
housing['LotShape'] = housing['LotShape'].map({'Reg': 0, 'IR1': 1, 'IR2': 2, 'IR3': 3})
housing['LandContour'] = housing['LandContour'].map({'Lvl': 0, 'Bnk': 1, 'HLS': 2, 'Low': 3})
housing['Utilities'] = housing['Utilities'].map({'AllPub': 0, 'NoSewr': 1, 'NoSeWa': 2, 'ELO': 3})
housing['LotConfig'] = housing['LotConfig'].map({'Inside': 0, 'Corner': 1, 'CulDSac': 2, 'FR2': 3, 'FR3': 4})
housing['LandSlope'] = housing['LandSlope'].map({'Gtl': 0, 'Mod': 1, 'Sev': 2})
housing['Condition1'] = housing['Condition1'].map({'Artery': 0, 'Feedr': 1, 'Norm': 2, 'RRNn': 3, 'RRAn': 4, 'PosN': 5, 'PosA': 6, 'RRNe': 7, 'RRAe': 8 })
housing['Condition2'] = housing['Condition2'].map({'Artery': 0, 'Feedr': 1, 'Norm': 2, 'RRNn': 3,'RRAn':4,'PosN': 5,'PosA': 6, 'RRNe': 7,'RRAe':8 })
housing['BldgType'] = housing['BldgType'].map({'1Fam': 0, '2fmCon': 1, 'Duplex': 2, 'TwnhsE': 3, 'Twnhs':4})
housing['MasVnrType'] = housing['MasVnrType'].map({'BrkCmn': 0, 'BrkFace': 1, 'CBlock': 2, 'None': 3,'Stone':4})
housing['ExterQual'] = housing['ExterQual'].map({'Ex': 0, 'Gd': 1, 'TA': 2, 'Fa': 3,'Po': 4})
housing['ExterCond'] = housing['ExterCond'].map({'Ex': 0, 'Gd': 1, 'TA': 2, 'Fa': 3,'Po': 4})
housing['Foundation'] = housing['Foundation'].map({'BrkTil': 0, 'CBlock': 1, 'PConc': 2, 'Slab': 3, 'Stone': 4, 'Wood': 5})
housing['BsmtQual'] = housing['BsmtQual'].map({'Ex': 0, 'Gd': 1, 'TA': 2, 'Fa': 3, 'Po': 4, 'None': 5})
housing['BsmtCond'] = housing['BsmtCond'].map({'Ex': 0, 'Gd': 1, 'TA': 2, 'Fa': 3, 'Po': 4, 'None': 5})
housing['BsmtExposure'] = housing['BsmtExposure'].map({'Gd': 0, 'Av': 1, 'Mn': 2, 'No': 3, 'None': 4})
housing['BsmtFinType1'] = housing['BsmtFinType1'].map({'GLQ': 0, 'ALQ': 1, 'BLQ': 2, 'Rec': 3, 'LwQ': 4, 'Unf': 5, 'None': 6})
housing['BsmtFinType2'] = housing['BsmtFinType2'].map({'GLQ': 0, 'ALQ': 1, 'BLQ': 2, 'Rec': 3, 'LwQ': 4, 'Unf': 5, 'None': 6})
housing['HeatingQC'] = housing['HeatingQC'].map({'Ex': 0, 'Gd': 1, 'TA': 2, 'Fa': 3, 'Po': 4})
housing['CentralAir'] = housing['CentralAir'].map({'N': 0, 'Y': 1})
housing['Electrical'] = housing['Electrical'].map({'SBrkr': 0, 'FuseA': 1, 'FuseF': 2, 'FuseP': 3, 'Mix': 4, 'None': 5})
housing['KitchenQual'] = housing['KitchenQual'].map({'Ex': 0, 'Gd': 1, 'TA': 2, 'Fa': 3, 'Po': 4})
housing['Functional'] = housing['Functional'].map({'Typ': 0, 'Min1': 1, 'Min2': 2, 'Mod': 3, 'Maj1': 4,'Maj2': 5,'Sev': 6, 'Sal': 7})
housing['FireplaceQu'] = housing['FireplaceQu'].map({'Ex': 0, 'Gd': 1, 'TA': 2, 'Fa': 3, 'Po': 4, 'None': 5})
housing['GarageType'] = housing['GarageType'].map({'2Types': 0, 'Attchd': 1, 'Basment': 2, 'BuiltIn': 3, 'CarPort': 4, 'Detchd': 5, 'None': 6})
housing['GarageFinish'] = housing['GarageFinish'].map({'Fin': 0, 'RFn': 1, 'Unf': 2, 'None': 3})
housing['GarageQual'] = housing['GarageQual'].map({'Ex': 0, 'Gd': 1, 'TA': 2, 'Fa': 3, 'Po': 4,'None': 5})
housing['GarageCond'] = housing['GarageCond'].map({'Ex': 0, 'Gd': 1, 'TA': 2, 'Fa': 3, 'Po': 4,'None': 5})
housing['PavedDrive'] = housing['PavedDrive'].map({'Y': 0, 'P': 1, 'N': 2})
housing['PoolQC'] = housing['PoolQC'].map({'Ex': 0, 'Gd': 1, 'TA': 2, 'Fa': 3, 'None': 4})
housing['Fence'] = housing['Fence'].map({'GdPrv': 0, 'MnPrv': 1, 'GdWo': 2, 'MnWw': 3, 'None': 4})
housing['HouseStyle'] = housing['HouseStyle'].map({'1Story': 0, '1.5Fin': 1, '1.5Unf': 2, '2Story': 3, '2.5Fin': 4, '2.5Unf': 5,
                                                   'SFoyer': 6, 'SLvl': 7})
housing['SaleType'] = housing['SaleType'].map({'WD': 0, 'CWD': 1, 'VWD': 2, 'New': 3, 'COD': 4,'Con': 5,'ConLw': 6, 'ConLI': 7, 'ConLD': 8,
                         'Oth': 9})

#丢弃价格为0的行
housing = housing[~housing['SalePrice'].isin([0])]
##准备数据集
temp_X_col = np.array(housing.columns)
temp_X_col = temp_X_col.tolist()
temp_X_col.remove('SalePrice')
temp_X_col.remove('Id')

X = housing[temp_X_col]
#X.dropna(axis=1, how='any')
y = housing['SalePrice']
##进行测试集分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
print(X_train.isnull().sum())
#进行数据标准化
pipeline = Pipeline([('std_scalar', StandardScaler())])
X_train = pipeline.fit_transform(X_train)
X_test = pipeline.transform(X_test)
#定义交叉验证
def cross_val(model):
    pred = cross_val_score(model, X, y, cv=10)
    return pred.mean()

##打印预测评估
def print_evaluate(true, predicted):
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    print('MAE:', mae)
    print('MSE:', mse)
    print('RMSE:', rmse)
    print('R2 Square', r2_square)

##对模型进行评估
def evaluate(true, predicted):
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    return mae, mse, rmse, r2_square

##选择线性回归作为训练模型
lin_reg = LinearRegression(normalize=True)
lin_reg.fit(X_train, y_train)
#打印模型特征
print(lin_reg.intercept_)
##打印数据相关性
coeff_df = pd.DataFrame(lin_reg.coef_, X.columns, columns=['Coefficient'])
print(coeff_df)
##利用测试集的预测进行画图
pred = lin_reg.predict(X_test)
plt.scatter(y_test, pred)
sns.distplot((y_test - pred), bins=50)
#对测试集的准确率进行评估
test_pred = lin_reg.predict(X_test)
train_pred = lin_reg.predict(X_train)
print('Test set evaluation:\n_____________________________________')
print_evaluate(y_test, test_pred)
print('====================================')
print('Train set evaluation:\n_____________________________________')
print_evaluate(y_train, train_pred)
#以表格形式输出结果
results_df = pd.DataFrame(data=[["Linear Regression", *evaluate(y_test, test_pred) , cross_val(LinearRegression())]],
                          columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', "Cross Validation"])
print(results_df)
plt.show()
