# -*- coding: utf-8 -*-
"""
Created on Fri May  7 12:48:56 2021

@author: Tejas

Custom KNN imputation.
"""
### Things to be changed in the code for other phases

import pandas as pd
import numpy as np
import ast
from datetime import datetime
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')


#### Custom KNN imputation:
#X = fitData, X1 = transform data 
def myKNN(X_fit, X_transform, n_neighbours, dist_metric, value_metric):
    X_fit = pd.DataFrame(X_fit)
    X_fit.reset_index(drop=True, inplace = True)
    X_transform.reset_index(drop = True, inplace = True)
    df_dist = pd.DataFrame()
    for i in range(0, X_transform.shape[0]):
        current_row = X_transform.iloc[i,:]
        
        if dist_metric == 'manhattan':
            #Manhattan Distance
            current_dist = []
            for j in range(0, X_fit.shape[0]):
                current_dist.append(sum(abs(current_row - X_fit.iloc[j,:]).fillna(0)))
                #print(j)
            df_dist = pd.concat([df_dist, pd.DataFrame(current_dist)], axis = 1)          
            
        if dist_metric == 'euclidean':
            current_dist = []
            for j in range(0, X_fit.shape[0]):
                current_dist.append(np.sqrt(sum((current_row - X_fit.iloc[j,:]).fillna(0)**2)))
            df_dist = pd.concat([df_dist, pd.DataFrame(current_dist)], axis = 1)
        
        if dist_metric == 'Jaccard':
            current_dist = []
            for j in range(0, X_fit.shape[0]):
                current_dist.append(1 - (list(current_row - X_fit.iloc[j,:]).count(0)/len(list(current_row - X_fit.iloc[j,:]))))
            df_dist = pd.concat([df_dist, pd.DataFrame(current_dist)], axis = 1)
        
        '''
        if dist_metric == 'Gower':
            # Manhattan for contineous + Jaccard for one hot columns.
            current_dist = []
            for j in range(0, X_fit.shape[0]):
                #man distance
                man_dist = sum(abs(current_row[non_onehotcols] - X_fit.iloc[j, :][non_onehotcols]).fillna(0))
                jac_dist = 1 - (list((current_row[onehotcols] - X_fit.iloc[j,:][onehotcols]).fillna(0)).count(0)/ len(onehotcols))
                current_dist.append(man_dist + jac_dist)
            df_dist = pd.concat([df_dist, pd.DataFrame(current_dist)], axis = 1)
        print(i)
        '''    
        
    ### Replace the 0 distance (row neighbour to itself) with a very large num, so it is last neigh
    df_dist.replace([0], 999999, inplace = True)
    
    ### from distance matrix, extract k-nearest neighbours  
    NN_all = pd.DataFrame(np.argsort(df_dist))  
        
    
    ### Fill the values by median of K neighbours
    X_filled = X_transform.copy()  # Fill the missing values in X_filled.
    X_null = X_transform.isnull()  # Has 'True' in all places with NaN in X
    for i in range(0, X_transform.shape[0]): #current row = i
        for j in range(0, X_transform.shape[1]): #current num of cols = j
            if X_null.iloc[i,j] == True:

                #take all index numbers where jth column has no missing
                idx_all_rows_with_value = list(X_fit.iloc[:,j].dropna().index)
                
                #From ith row, take all nearest neighbours
                NN = list(NN_all.iloc[i, :])
                
                # Remove nearest neighbours that have NaN in same column
                NN_good = [x for x in NN if x in idx_all_rows_with_value]
    
                # Keep only the n required neighbours
                NN_good = NN_good[:n_neighbours]
                
                # Extract rows in NN_good from actual X
                X_neigh = list(X_fit.iloc[NN_good, j])
                
                if value_metric == 'median':
                    # take the mean/mode/median of these elements and substitute in X_filled
                    X_filled.iloc[i,j] = np.median(X_neigh)
    
                if value_metric == 'mean':
                    # take the mean/mode/median of these elements and substitute in X_filled
                    X_filled.iloc[i,j] = np.mean(X_neigh)
                    
                if value_metric == 'mode':
                    import statistics
                    # take the mean/mode/median of these elements and substitute in X_filled
                    X_filled.iloc[i,j] = statistics.mode(X_neigh)
        print(i)
    return X_filled



X = [[1, 2, np.nan], [3, 4, 3], [np.nan, 6, 5], [8, 8, 7], [1, np.nan, 9]]
X = pd.DataFrame(X)



myKNN(X_fit=X, X_transform=X, n_neighbours=2, dist_metric='manhattan', value_metric = 'median')

'''
Out[10]: 
     0    1    2
0  1.0  2.0  4.0
1  3.0  4.0  3.0
2  2.0  6.0  5.0
3  8.0  8.0  7.0
4  1.0  5.0  9.0

'''




