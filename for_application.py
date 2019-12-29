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
    df = pd.read_csv(r'../yujia_data/data/project_machine.csv')
    column_names_old = list(df)
    df = df[df['sic2']==54]
    df = df.fillna(0)
    #df = df.dropna(axis=1)
    not_include = ['RET','DATE', 'permno', 'sic2', 'gvkey', 'fyear', 'datadate',
                   'rdq', 'ewret', 'DLRET', 'DLSTCD']
    x = df.drop(not_include, axis=1)
    column_names = [e for e in column_names_old if e not in not_include] 
    y = df['RET']
    
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

def lasso(x_train, x_test, y_train, y_test, alpha=0.0001):
    '''
    A lasso regression function:
        
    '''
    clf = Lasso(alpha=alpha, max_iter=10000)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    coef = clf.coef_.squeeze()
    return y_pred, coef#[column_names[i] for i in coef_max]

def ridge(x_train, x_test, y_train, y_test):
    clf = Ridge(alpha=1, max_iter=10000)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    coef = clf.coef_.squeeze()
    coef_max = np.abs(coef).argsort()[-25:][::-1]
    return y_pred, [column_names[i] for i in coef_max]

def ols(x_train, x_test, y_train, y_test):
    clf = LinearRegression()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    coef = clf.coef_.squeeze()
    coef_max = np.abs(coef).argsort()[-25:][::-1]
    return y_pred, [column_names[i] for i in coef_max]

x, y, column_names = generate_data()

# 4 fold validation
kf = KFold(n_splits=4, shuffle=True, random_state=42)
kf.get_n_splits(x)
y_tests, y_preds, coefs = [], [], []
for train_index, test_index in kf.split(x):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    y_pred, coef = lasso(x_train, x_test, y_train, y_test)
    y_tests.append(y_test)
    y_preds.append(y_pred)
    coefs.append(coef)
y_test = np.concatenate(y_tests)
y_pred = np.concatenate(y_preds)
coef = np.abs(np.array(coefs).sum(axis=0))
coef_max = np.abs(coef).argsort()[-50:][::-1]


#%%
x_train = x_train[:, coef_max]
x_test = x_test[:, coef_max]
#%%
import keras
from keras import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
np.random.seed(42)
def regression_cnn():
    model = Sequential()
    model.add(Dense(256, activation='relu', input_shape=(50,)))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation = 'linear'))
    return model

weight_name = r'./yujia.h5'
model = regression_cnn()  
adam = keras.optimizers.Adam(lr=0.01)
model.compile(loss='mean_squared_error', optimizer= adam, metrics=['mean_squared_error'])
elstp = EarlyStopping(monitor='val_loss', min_delta=0, patience=200, verbose=0, mode='min')
checkpoint = ModelCheckpoint(weight_name, monitor='val_loss', verbose=0, save_best_only=True, mode='min')
callbacks_list = [checkpoint,elstp]
history = model.fit(x_train, y_train, epochs=2000, batch_size=len(y_train), validation_split=0.25, callbacks=callbacks_list, verbose=2)
history_history = history.history
model.load_weights(weight_name)
model.compile(loss='mean_squared_error', optimizer= adam, metrics=['mean_squared_error'])
y_pred = model.predict(x_test)
print('Lasso R2: {}'.format(r2_score(y_test, y_pred)))
print('Lasso RMSE: {}'.format(mean_squared_error(y_test, y_pred)**0.5))


    

