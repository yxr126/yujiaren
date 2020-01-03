#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 14:24:49 2019

@author: Yujia Ren
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, BayesianRidge
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold



def generate_data(sic2_number=54):
    '''
    A function to read the stock data and output standardized x and y
    args: None
    return: standardized x, y and column names
    '''
    # import feature and returns
    df = pd.read_csv(r'../yujia_data/data/project_machine.csv')
    column_names_old = list(df)
    df = df[df['sic2']==sic2_number]
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

def lasso(x_train, x_test, y_train, y_test, alpha=0.0001):
    clf = Lasso(alpha=alpha, max_iter=10000)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    coef = clf.coef_.squeeze()
    return y_pred, coef

def ridge(x_train, x_test, y_train, y_test, alpha=0.2):
    clf = Ridge(alpha=alpha, max_iter=10000)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    coef = clf.coef_.squeeze()
    return y_pred, coef

def ols(x_train, x_test, y_train, y_test, alpha=0):
    clf = LinearRegression()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    coef = clf.coef_.squeeze()
    return y_pred, coef

def bayesianridge(x_train, x_test, y_train, y_test, alpha=0):
    clf = BayesianRidge()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    coef = clf.coef_.squeeze()
    return y_pred, coef

def svr(x_train, x_test, y_train, y_test, alpha=0):
    clf = SVR(kernel='linear', degree=alpha, gamma='scale')
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    return y_pred, 0

def sgdregressor(x_train, x_test, y_train, y_test, alpha=0):
    clf = SGDRegressor(penalty='l1')
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    coef = clf.coef_.squeeze()
    return y_pred, coef

def decisiontree(x_train, x_test, y_train, y_test, alpha=0):
    clf = DecisionTreeRegressor()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    return y_pred, 0


def k_fold(x, y, n_splits=4):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    kf.get_n_splits(x)
    x_trains, x_tests, y_trains, y_tests = [], [], [], []
    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        x_trains.append(x_train)
        x_tests.append(x_test)
        y_trains.append(y_train)
        y_tests.append(y_test)
    return x_trains, x_tests, y_trains, y_tests