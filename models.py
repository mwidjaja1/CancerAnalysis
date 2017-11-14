#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 11:13:10 2017

@author: matthew
"""

from sklearn import discriminant_analysis, linear_model, model_selection, naive_bayes, neighbors, svm, tree
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import featureselect


def scikit_validator(in_train, out_train):
    """ Tests Scikit Learn models against training data

        Conclusions was that Logistic Regression (Mean 0.949 & SD 0.033) and
        LinearDiscriminantAnalysis (Mean 0.956 & SD 0.033) were the best. LDA
        is consistently high while LR is most likely to perform dead on.

    """
    # Spot Check Algorithms
    models = []
    models.append(('LR', linear_model.LogisticRegression()))
    models.append(('LDA', discriminant_analysis.LinearDiscriminantAnalysis()))
    models.append(('QDA', discriminant_analysis.QuadraticDiscriminantAnalysis()))
    models.append(('KNN', neighbors.KNeighborsClassifier()))
    models.append(('TREE', tree.DecisionTreeClassifier()))
    models.append(('NB', naive_bayes.GaussianNB()))
    models.append(('SVM', svm.SVC()))

    # Evaluate Model
    sk_summary = {}
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10)
        results = model_selection.cross_val_score(model, in_train, out_train,
                                                  cv=kfold, scoring='accuracy')
        sk_summary[name] = results
        print('{}: {} ({})'.format(name, results.mean(), results.std()))

    # Plots Results
    fig, ax = plt.subplots(1, 1)
    ax.set_title('Algorithm Comparison')
    plt.boxplot([sk_summary[x] for x in sk_summary])
    ax.set_xticklabels([x for x in sk_summary])
    plt.show()
    
    return sk_summary


def lda(in_train, out_train, in_test, out_test):
    lda = discriminant_analysis.LinearDiscriminantAnalysis()
    lda.fit(in_train, out_train)
    predict = pd.Series(lda.predict(in_test))

    matrix = confusion_matrix(out_test, predict)
    featureselect.plot_confusion_matrix(matrix, out_train.unique(),
                                        title='LDA Confusion Matrix')

    print('\nLDA Model Results')
    print('Score: {}'.format(lda.score(predict, out_test)))
    print('Accuracy: {}'.format(metrics.accuracy_score(predict, actual)))
    print('Precision: {}'.format(metrics.precision_score(predict, actual)))
    print('Recall/F1: {}'.format(metrics.f1_score(predict, actual)))
    return predict


def qda(in_train, out_train, in_test, out_test):
    qda = discriminant_analysis.QuadraticDiscriminantAnalysis()
    qda.fit(in_train, out_train)
    predict = pd.Series(qda.predict(in_test))

    matrix = confusion_matrix(out_test, predict)
    featureselect.plot_confusion_matrix(matrix, out_train.unique(),
                                        title='QDA Confusion Matrix')

    print('\nQDA Model Results')
    print('Score: {}'.format(lda.score(predict, out_test)))
    print('Accuracy: {}'.format(metrics.accuracy_score(predict, actual)))
    print('Precision: {}'.format(metrics.precision_score(predict, actual)))
    print('Recall/F1: {}'.format(metrics.f1_score(predict, actual)))
    return predict