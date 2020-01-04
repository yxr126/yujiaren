# import modules
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# import my own mudules
from utils import generate_data, lasso, ridge, ols, k_fold, bayesianridge, svr, sgdregressor, decisiontree
from plot_tools import plot_pred

x, y, column_names = generate_data(sic2_number=1)



# Ordinary Least Square
def evaluation(x, y, metric, alpha):
    x_trains, x_tests, y_trains, y_tests = k_fold(x, y, n_splits=4)
    y_true = []
    y_pred = []
    for i in range(len(x_trains)):
        x_train = x_trains[i]
        x_test = x_tests[i]
        y_train = y_trains[i]
        y_test = y_tests[i]
        y_pred.append(metric(x_train, x_test, y_train, y_test, alpha)[0])
        y_true.append(y_tests[i])
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    # evaluation
    plot_pred(y_true, y_pred, metric.__name__)
    r2 = r2_score(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred)**0.5
    print('R2: {:.4f}'.format(r2))
    print('RMSE: {:.4f}'.format(rmse))
    return r2, rmse

evaluation(x, y, metric=ols, alpha=0)
evaluation(x, y, metric=ridge, alpha=0.2)
evaluation(x, y, metric=lasso, alpha=0.0001)
evaluation(x, y, metric=bayesianridge, alpha=0.0001)
evaluation(x, y, metric=svr, alpha=0)
evaluation(x, y, metric=sgdregressor, alpha=0)
evaluation(x, y, metric=decisiontree, alpha=0)


# select descriptors
# Removing features with low variance
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(.99 * (1 - .99))
x_vt = sel.fit_transform(x)
evaluation(x_vt, y, metric=ols, alpha=0)
evaluation(x_vt, y, metric=lasso, alpha=0.0001)

# Univariate feature selection
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression, Lasso, Ridge, BayesianRidge
lr = LinearRegression()
rfe = RFE(estimator=lr, n_features_to_select=90, step=1)
rfe.fit(x, y)
ranking = rfe.ranking_
select = rfe.support_
x_rfe = x[:, select]
evaluation(x_rfe, y, metric=ols, alpha=0)

# PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=10)
pca.fit(x)
x_pca = pca.fit_transform(x)
evaluation(x_pca, y, metric=ols, alpha=0)

# CCA
from sklearn.cross_decomposition import CCA
cca = CCA(n_components=1)
cca.fit(x, y)
x_cca, y_cca = cca.transform(x, y)
evaluation(x_cca, y, metric=ols, alpha=0)

#%%


#
#coef = np.abs(np.array(coefs).sum(axis=0))
#coef_max = np.abs(coef).argsort()[-50:][::-1]


#%%
#x_train = x_train[:, coef_max]
#x_test = x_test[:, coef_max]
#%%
#import keras
#from keras import Sequential
#from keras.layers import Dense, BatchNormalization, Dropout
#from keras.callbacks import ModelCheckpoint, EarlyStopping
#np.random.seed(42)
def regression_cnn():
    model = Sequential()
    model.add(Dense(256, activation='relu', input_shape=(139,)))
    model.add(BatchNormalization())
    model.add(Dense(256, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dense(1, activation = 'linear'))
    return model

weight_name = r'./yujia.h5'
model = regression_cnn()  
adam = keras.optimizers.Adam(lr=0.001)
model.compile(loss='mean_squared_error', optimizer= adam, metrics=['mean_squared_error'])
elstp = EarlyStopping(monitor='val_loss', min_delta=0, patience=1000, verbose=0, mode='min')
checkpoint = ModelCheckpoint(weight_name, monitor='val_loss', verbose=0, save_best_only=True, mode='min')
callbacks_list = [checkpoint,elstp]
history = model.fit(x_train, y_train, epochs=20000, batch_size=len(y_train), validation_split=0.25, callbacks=callbacks_list, verbose=2)
history_history = history.history
model.load_weights(weight_name)
model.compile(loss='mean_squared_error', optimizer= adam, metrics=['mean_squared_error'])
y_pred = model.predict(x_test)
print('Lasso R2: {}'.format(r2_score(y_test, y_pred)))
print('Lasso RMSE: {}'.format(mean_squared_error(y_test, y_pred)**0.5))
#%%
plt.scatter(y_test, y_pred)
#%%
for key in history_history.keys():
    plt.plot(history_history[key], label=key)
    plt.legend()

    

