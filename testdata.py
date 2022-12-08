
from functions import *
import time

k=4 #shingle size
original_data=load_data()
title=data_title(original_data)
df=pd.DataFrame(title, columns=['modelID', 'title', 'shop', 'brand'])
cleaned_data=clean_data(df)
Nbootstraps=5
#times=np.zeros((Nbootstraps,1))
thresholds =np.array([1,1,1,1,0.55,0.55,0.55,0.65,0.65,0.65,0.65,0.65,0.65,0.65]) #fill in optimal thresholds

for boot in range(Nbootstraps):
    
    train_data, test_data= bootstrap(cleaned_data)
    shingle_matrix=boolean_matrix(k, test_data)
    
    nHashes=192 #approx 50% of number of words
    
    minhash=minhashing(shingle_matrix, nHashes)
    possible_b, possible_r = value_b_r(minhash)
    
    result=pd.DataFrame(columns=['PQ', 'PC', 'F1StarLSH', 'precision', 'recall', 'F1_cluster', 'fraction_comparisonLSH', 'fraction_cluster','b' ,'r','time'])
    for i in range(len(possible_b)):
       startTime = time.perf_counter() 
       b=possible_b[i]
       r=possible_r[i]
            
       LSH_pairs=lsh(minhash, b, r)
            
       distance_matrix=dissimilarity_matrix(LSH_pairs, test_data,k)
       threshold=thresholds[i]
       
       cluster_pairs=clustering(distance_matrix, threshold)
            
        
       evals=evaluate_cluster_LSH(LSH_pairs, cluster_pairs, test_data)
       evals.append(b)
       evals.append(r)
       endTime = time.perf_counter()
       timetaken=endTime-startTime
       evals.append(timetaken)
       
           
       result.loc[len(result.index)]=evals
       print(f"bootstrap: {boot}, and number: {i} out of 13")
       
    
    
    result.to_excel("result_boot_test" +str(boot)+ ".xlsx")
   
