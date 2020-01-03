#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 14:28:42 2019

@author: Yujia Ren
"""
import matplotlib.pyplot as plt

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
    plt.show();
    