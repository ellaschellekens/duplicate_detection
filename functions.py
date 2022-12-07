# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 14:57:15 2022

@author: ellaschellekens
"""

import json
import pandas as pd
import numpy as np
import re
from nltk import flatten
import math
import random
import collections
import itertools
import scipy
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.utils import resample
from random import shuffle

def isPrime(n): #to find prime number for hashing
     
    # Corner cases
    if(n <= 1):
        return False
    if(n <= 3):
        return True
     
    # This is checked so that we can skip
    # middle five numbers in below loop
    if(n % 2 == 0 or n % 3 == 0):
        return False
     
    for i in range(5,int(math.sqrt(n) + 1), 6):
        if(n % i == 0 or n % (i + 2) == 0):
            return False
     
    return True

def nextPrime(N): #to find prime number for hashing
 
    # Base case
    if (N <= 1):
        return 2
 
    prime = N
    found = False
 
    # Loop continuously until isPrime returns
    # True for a number greater than n
    while(not found):
        prime = prime + 1
 
        if(isPrime(prime) == True):
            found = True
 
    return prime

def load_data():
    f = open( '/Users/ellas/OneDrive/Documents/Master/computer science/assignment/TVs-all-merged.json') #load product information dataset
    Jsondata =json.loads(f.read())
    #data= pd.DataFrame.from_dict(Jsondata, orient= "index")
    return(Jsondata)

def data_title(data):
    new_data=[]
    for key, value in data.items():
        for item in value:
            new_data.append((key, item.get('title'),item.get('shop'), item.get('featuresMap').get('Brand')))
    return(new_data)


def clean_data(data): 
    # need to remove capital  letters, interpunction etc from title, herz=hz, inch, remove store name, remove weird notation
    data['title']=data['title'].str.lower() # remove capital letters
    data['title']=data['title'].str.replace("\"", "inch")
    data['title']=data['title'].str.replace("inches", "inch")
    data['title']=data['title'].str.replace("'", "inch")
    data['title']=data['title'].str.replace("hertz", "hz")
    data['title']=data['title'].str.replace("best buy", "")
    data['title']=data['title'].str.replace("newegg.com", "")
    data['title']=data['title'].str.replace("thenerds.net", "")
    data['title']=data['title'].str.replace("amazon", "")
    data['title']=data['title'].str.replace(" - ", " ")
    data['title']=data['title'].str.replace(" / ", " ")
    data['title']=data['title'].str.replace("diagonal", "diag")
    data['title']=data['title'].str.replace('[^\w\s]','')
    data['title']=data['title'].str.strip()
    #data['title']=data['title'].str.replace(" ","")
    data['brand']=data['brand'].str.lower() # remove capital letters
    data['shop']=data['shop'].str.lower() # remove capital letters
    return(data)

def bootstrap(data):
    train=resample(range(0, len(data)), replace=True, n_samples=len(data))
    train=set(train)    
    traindata=pd.DataFrame(columns=['modelID', 'title', 'shop', 'brand'])
    testdata=pd.DataFrame(columns=['modelID', 'title', 'shop', 'brand'])
    for i in range(len(data)):
        if i in train:
            traindata.loc[len(traindata.index)]=data.iloc[i]
        else:
            testdata.loc[len(testdata.index)]=data.iloc[i]
    return traindata, testdata


def model_words(title):
    patternmwTitle = re.compile(r'(\b(?:[0-9]*)[a-z]+\b)|(\b[a-z]+\b)')
    mwTitle = [x for sublist in patternmwTitle.findall(title)for x in sublist if x != ""]
    return mwTitle
  
#r'(\b(?:[0-9]*)[a-z]+\b)|(\b[a-z]+\b)'
#r'([a-zA-Z0-9]*(([0-9]+[^0-9, ]+)|([^0-9, ]+[0-9]+))[a-zA-Z0-9]*)'
def get_shingle(k,text):
    lijst=[]
    for i in range (0,len(text)-k+1):
        #yield tuple(text[i:i+k])
        lijst.append(text[i:i+k])
    return lijst
        

def get_word(text): #similar to get shingke but k=1, used to find all words 
    lijst=[]
    for i in range (0,len(text)-1+1):
        #yield tuple(text[i:i+k])
        lijst.append(text[i:i+1])
    return lijst
        
#shingles1 = { i for i in get_shingle(2,test1234)}
#print(shingles1)


def boolean_matrix(k,data): #0,1 matrix which denotes if a word is in the title of the tv
    nRows=len(data)
    allwords=[]
    allShingles=[]
    
    for i in range(nRows):
        allwords.append(model_words(data['title'][i])) # create nested list where first list contains words of product 1
    for i in range(len(allwords)):    
        allShingles.extend(get_word(allwords[i])) #create list with all words that occur in the tv titles
        
   
    unique_words=[] # list that will contain all the unique words
    for x in allShingles:
        if x not in unique_words:
            unique_words.append(x)
  
    number_words=int(len(unique_words))
    matrix=np.zeros((number_words, nRows))
    for i in range(number_words):
        for j in range(nRows):
            matrix[i,j]=set(unique_words[i]).issubset(allwords[j])
   
   
    return matrix

def minhashing(shingle_matrix, nHashes): 
    nWords, nProducts =shingle_matrix.shape
    
    prime=nextPrime(nHashes)
    #nHashes=math.ceil(nWords/2) #number of hashes is equal to half of the number of words
    
    result=np.full((nHashes,nProducts),np.inf) # final result matrix initialised with large value
    allhashes=np.full((nHashes), np.inf) #array to save hashes per row
   
    
    a= []
    b= []
    for i in range(nHashes):
        a.append(random.randint(1,100))
        b.append(random.randint(1,100)) 
    for row in range (nWords):
        for h in range(nHashes):
            allhashes[h]=(a[h]+b[h]*row) % prime
        for col in range(nProducts):
            if shingle_matrix.item((row, col))==1:
                minvalue=np.minimum(result[:,col],allhashes)
                result[:,col]=minvalue
             
    
    return result


def value_b_r(hash_matrix):#possible values for band and rows
    nHashes, nProducts=hash_matrix.shape
    row=[]
    band=[]
    for r in range(nHashes+1):
        for b in range (nHashes+1):
            if r*b==nHashes:
                row.append(r)
                band.append(b)
    
    
    return row, band

def lsh(hash_matrix, b, r): #do lsh
    nHashes, nProducts=hash_matrix.shape
    #assert(b*r==nHashes)
    bands=np.split(hash_matrix,b,axis=0)#werkt alleen als size deelbaar is door b
    hashedbuckets = collections.defaultdict(set)
    
    LSH_pairs = set()
    for item, band in enumerate(bands):
       
        for j in range(nProducts):
             band_id = tuple(list(band[:,j])+[str(item)])
             hashedbuckets[band_id].add(j)
    for bucket_pairs in hashedbuckets.values():
        if len(bucket_pairs) > 1:
            for pair in itertools.combinations(bucket_pairs, 2): #makes combination of two of all products in the bucket
                LSH_pairs.add(pair)
                
    lsh_list=list(LSH_pairs)
    lsh_list_new=[el for el in lsh_list if el[0] < el[1]] #delete (2,1) if (1,2) is in list
    LSH_pairs_new=set(lsh_list_new)
    
    return LSH_pairs_new




def duplicates(data): #find total number of duplicates in data
    dups=0
    for i in range(len(data)):
        for j in range(i+1,len(data),1):
           if data['modelID'][i]==data['modelID'][j]:
                    dups=dups+1
    return dups


def evaluateLSH(LSH_pairs, data):#no longer needed is included in other evaluation
    actualPairs = 0
    for pair in LSH_pairs:
        if data['modelID'][pair[0]] == data['modelID'][pair[1]]:
            actualPairs = actualPairs + 1
    numberDuplicates = duplicates(data)
    numberCandidates = len(LSH_pairs)+0.0000001
    PQ = actualPairs / numberCandidates
    PC = actualPairs / numberDuplicates
    F1StarLSH = 2*(PQ*PC)/(PQ+PC+0.0000001)
    print(f" PQ_LSH: {PQ}, PC_LSH: {PC}, F1_LSH: {F1StarLSH}")
    N=len(data)
    totalComparisons=(N*(N-1))/2
    fraction_comparison=numberCandidates/totalComparisons
    print(f"total comparisons: {totalComparisons}, and fraction: {fraction_comparison}")
    return [PQ, PC, F1StarLSH, fraction_comparison]

def jaccard_distance(word1, word2):
    
   return len(word1.intersection(word2)) / len(word1.union(word2))



def dissimilarity_matrix(LSH_pairs, data,k): #needed for clustering method
    nProducts=len(data)
    resultmatrix=np.full((nProducts, nProducts),100, float) #initialise products with large distance
    np.fill_diagonal(resultmatrix,0) #distance between the product itself is 0
    
    for pair in LSH_pairs:
        shingle0=get_shingle(k,data['title'][pair[0]])
        shingle1=get_shingle(k,data['title'][pair[1]])
        resultmatrix[pair[0], pair[1]]=1-(jaccard_distance(set(shingle0),set(shingle1))) #1- because jaccard is similarity and i need distance
        if(data['shop'][pair[0]] == data['shop'][pair[1]] ): #cannot be from same shop
            resultmatrix[pair[0], pair[1]]=100
        if(data['brand'][pair[0]] is not None and data['brand'][pair[1]] is not None and data['brand'][pair[0]] != data['brand'][pair[1]]):
            resultmatrix[pair[0], pair[1]]=100 #cannot have a different brand
        resultmatrix[pair[1],pair[0]]==resultmatrix[pair[0], pair[1]] #symmetric distance
    
            
    return resultmatrix

def similar_pairs(LSH_pairs, data,k): #not used, short method that see products as duplicates is jaccard similarity is larger than threshold
    
    cluster_pairs = set()
    for pair in LSH_pairs:
        shingle0=get_shingle(k,data['title'][pair[0]])
        shingle1=get_shingle(k,data['title'][pair[1]])
        if(jaccard_distance(set(shingle0),set(shingle1))>0.6):
            cluster_pairs.add(pair)
            
    
    return cluster_pairs



def clustering(dissimilarity_matrix, threshold):
    
     linkage2 = AgglomerativeClustering(n_clusters=None, affinity="precomputed",linkage='average', distance_threshold=threshold)
     clusters = linkage2.fit_predict(dissimilarity_matrix)
     #products with the same value in clusters are potential pairs
     nProducts=len(clusters)
     count=0
   
     dictcluster = collections.defaultdict(set)
     cluster_pairs = set()
     
     for j in range(nProducts):
              dictcluster[clusters[j]].add(j)
     for pair in dictcluster.values():
         if len(pair) > 1:
             count=count+1
             for pair in itertools.combinations(pair, 2): #makes combination of two of all products in the bucket
                #pair_sort=(sorted(set(bucket_pairs)))
                 cluster_pairs.add(pair)
                 
     cluster_list=list(cluster_pairs)
     cluster_list_new=[el for el in cluster_list if el[0] < el[1]] #delete (2,1) if (1,2) is in list
     cluster_pairs_new=set(cluster_list_new)
     
     #test2=linkage(dissimilarity_matrix, method='average')
     
     
     return cluster_pairs_new
 

def evaluate_cluster_LSH(LSH_pairs, cluster_pairs, data):
    actualPairs_LSH = 0
    actualPairs_cluster = 0
    
    for pair in LSH_pairs:
        if data['modelID'][pair[0]] == data['modelID'][pair[1]]:
            actualPairs_LSH = actualPairs_LSH + 1
    for pair in cluster_pairs:
        if data['modelID'][pair[0]] == data['modelID'][pair[1]]:
            actualPairs_cluster=actualPairs_cluster+1
            
    numberDuplicates = duplicates(data)
    N=len(data)
    totalComparisons=(N*(N-1))/2
    
    #for LSH
    numberCandidatesLSH = len(LSH_pairs)+0.0000001
    PQ = actualPairs_LSH / numberCandidatesLSH
    PC = actualPairs_LSH / numberDuplicates
    F1StarLSH = 2*(PQ*PC)/(PQ+PC+0.0000001)
    fraction_comparisonLSH=numberCandidatesLSH/totalComparisons
    
   #for clustering
    precision= actualPairs_cluster/(len(cluster_pairs)+0.0000001)
    recall=actualPairs_cluster/numberDuplicates
    F1_cluster= (2*(precision*recall))/(precision+recall+0.0000001)
    fraction_cluster= (len(cluster_pairs))/totalComparisons
   
    
    return[PQ, PC, F1StarLSH, precision, recall, F1_cluster, fraction_comparisonLSH, fraction_cluster]

def optimal_threshold(data,distance_matrix, LSH_pairs): #for each combination of row and band find optimal threshold for cluster
    optimalF1=0
    threshold=1
    for t in range(21):
        cluster_pair=clustering(distance_matrix, t/20)
        [ _, _ , _, _, _, F1, _, _]=evaluate_cluster_LSH(LSH_pairs,cluster_pair,data)
        if(F1>optimalF1):
            threshold=t/20
            optimalF1=F1
    return threshold
        

    


