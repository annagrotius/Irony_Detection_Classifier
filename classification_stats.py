# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 16:37:10 2019

@author: Lisa and Lauren
"""

import pandas as pd

def get_classification_ironic(masterdf, newdf, mastercolumnindex_number, newcolumnindexnumber, weight):
    """Compares two columns of two dataframes with row index of 0,
    based on indices inputted; calculates the difference between the two values
    and multiples by the weight assigned. Returns list with new feature values"""
    ironic_average = masterdf.iloc[0][mastercolumnindex_number]
    #access column ONLY and all rows
    x = list(newdf.iloc[:,newcolumnindexnumber])
    new_list = []
    for item in x:
        new_list.append(abs(ironic_average - item)*weight)
    return new_list

def get_classification_non_ironic(masterdf, newdf, mastercolumnindex_number, newcolumnindexnumber, weight):
    """Compares two columns of two dataframes with row index of 1,
    based on indices inputted; calculates the difference between the two values
    and multiples by the weight assigned. Returns list with new feature values"""
    non_ironic_avergae = masterdf.iloc[1][mastercolumnindex_number]
    #access column ONLY and all rows
    x = list(newdf.iloc[:,newcolumnindexnumber])
    new_list = []
    for item in x:
        new_list.append(abs(non_ironic_avergae - item)*weight)
    return new_list

def final_predicition_results(feature_resultdf):
    """Takes dataframe with feature weights, calculates whicb feature weight
    is the smallest and returns list with final relevant classification label"""
    list_of_tuple_results = [tuple(x) for x in feature_resultdf.to_records(index=False)]
    prediciton_list = []
    for tup in list_of_tuple_results:
        non_ironic, ironic = tup
        if non_ironic > ironic:
            prediciton_list.append(1) #ironic
        elif non_ironic < ironic:
            prediciton_list.append(-1) #non-ironic
    return prediciton_list

def accuracy(testdf):
    """Compares labelled data with prediction and calculates accuracy of
    the classification"""
    label = list(testdf.iloc[:,1])
    prediciton = list(testdf.iloc[:,2])
    list_of_tuple_evaluations = list(zip(label, prediciton))
    gold_match = []
    no_match = []
    for tup in list_of_tuple_evaluations:
        label, prediction = tup
        if label == prediction:
            gold_match.append("True") #gold label match
        elif label != prediction:
            no_match.append("False")
    total_leng = len(testdf)
    gold_leng = len(gold_match)
    accuracy = gold_leng / total_leng
    return accuracy
