# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 15:32:57 2022

@author: ellaschellekens
"""
from functions import *


""" Load the data and extract the titles and clean the data
"""
k=4 #shingle size
original_data=load_data()
title=data_title(original_data)
df=pd.DataFrame(title, columns=['modelID', 'title', 'shop', 'brand'])
cleaned_data=clean_data(df)
Nbootstraps=5

for boot in range(Nbootstraps):
    
    train_data, test_data= bootstrap(cleaned_data)
    
    
    shingle_matrix=boolean_matrix(k, train_data)
    nHashes=192 #approx 50% of number of words
    minhash=minhashing(shingle_matrix, nHashes)
    possible_b, possible_r = value_b_r(minhash)
    
    result=pd.DataFrame(columns=['PQ', 'PC', 'F1StarLSH', 'precision', 'recall', 'F1_cluster', 'fraction_comparisonLSH', 'fraction_cluster','b' ,'r', 'thres'])
    for i in range(len(possible_b)):
    
        b=possible_b[i]
        r=possible_r[i]
        
        LSH_pairs=lsh(minhash, b, r)
        
        distance_matrix=dissimilarity_matrix(LSH_pairs, train_data,k)
        threshold=optimal_threshold(train_data, distance_matrix, LSH_pairs)
        cluster_pairs=clustering(distance_matrix, threshold)
        
        
        evals=evaluate_cluster_LSH(LSH_pairs, cluster_pairs, train_data)
        evals.append(b)
        evals.append(r)
        evals.append(threshold)
       
       # PQ, PC, F1StarLSH, precision, recall, F1_cluster, fraction_comparisonLSH, fraction_cluster = evaluate_cluster_LSH(LSH_pairs, cluster_pairs, train_data)
       
        result.loc[len(result.index)]=evals
        print(f"bootstrap: {boot}, and number: {i} out of 13")
        
    print(f"bootstrap done: {boot}")
    result.to_excel("result_ave_k4_boot_thres"+ str(boot) + ".xlsx")
        
        
