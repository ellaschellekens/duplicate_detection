# duplicate_detection
Method that uses LSH and hierarchical clustering for product duplicate detection

## Data

- The data set is available from:  https://personal.eur.nl/frasincar/datasets/TVs-allmerged.zip 
- The dataset that contains information about TVs from four different websites, namely, Amazon, Newegg, Best-Buy and The Nerds[4]. 
- The data set consists of 1624 products of which 1262 products are unique. 
- For each product a list of features is available, however, these features differ across products and web shops

## Software explanation

- main.py: program to run the training of the methods.
- functions.py: file that contains all functions needed for LSH and Clustering.
- testdata.py: program to run the test evaluation of the method. Needs as input the average of the optimal thresholds of the training bootrstaps.
- results.py: file to create plots of the test results. Needs as input the excel files that are saved after running testdata.py.
