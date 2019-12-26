# import modules
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

def generate_data():
    '''
    A function to read the stock data and output standardized x and y
    args: None
    return: standardized x, y and column names
    '''
    # import feature and returns
    df = pd.read_csv('./data/project_machine.csv')
    column_names = list(df)
    
    x = pd.DataFrame()
    for stock_id in [10025, 10182, 10259, 10501]:
        df_temp = df[df['permno'] == stock_id]
        x = x.append(df_temp)
    x = x.dropna(axis=1)
     
    y = pd.DataFrame()
    
    for stock_id in [10025, 10182, 10259, 10501]:
        df = pd.read_csv('./data/{}.csv'.format(stock_id))
        df = df['RET']
        df = df.to_frame()
        y = y.append(df)
    
    # standardize the data
    
    min_max_scaler = MinMaxScaler()
    x = min_max_scaler.fit_transform(x)
    y = y.values
    return x, y, column_names

def plot_pred(y_test, y_pred, title):
    '''
    A function to plot the actual and predicted stock return
    args: actual return, predicted return, type of regression
    return: save an image
    '''
    plt.figure()
    plt.scatter(y_test, y_pred)
    plt.xlabel('Actual return')
    plt.ylabel('Predicted return')
    plt.grid()
    plt.savefig('./plots/{}.svg'.format(title), dpi=400)
    plt.close()

def lasso(x_train, x_test, y_train, y_test, alpha=0.001):
    '''
    A lasso regression function:
        
    '''
    clf = Lasso(alpha=alpha, max_iter=10000)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    coef = clf.coef_.squeeze()
    coef_max = np.abs(coef).argsort()[-10:][::-1]
    return y_pred, [column_names[i] for i in coef_max]

def ridge(x_train, x_test, y_train, y_test):
    clf = Ridge(alpha=1, max_iter=10000)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    coef = clf.coef_.squeeze()
    coef_max = np.abs(coef).argsort()[-10:][::-1]
    return y_pred, [column_names[i] for i in coef_max]

def ols(x_train, x_test, y_train, y_test):
    clf = LinearRegression()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    coef = clf.coef_.squeeze()
    coef_max = np.abs(coef).argsort()[-10:][::-1]
    return y_pred, [column_names[i] for i in coef_max]

x, y, column_names = generate_data()

# 4 fold validation
kf = KFold(n_splits=4, shuffle=True, random_state=42)
kf.get_n_splits(x)
for train_index, test_index in kf.split(x):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    ridge(x_train, x_test, y_train, y_test)
    

