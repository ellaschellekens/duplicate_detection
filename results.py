import pandas as pd
import matplotlib.pyplot as plt

result0=pd.read_excel('/Users/ellas/OneDrive/Documents/Master/computer science/assignment/result_boot_test0.xlsx')
result1=pd.read_excel('/Users/ellas/OneDrive/Documents/Master/computer science/assignment/result_boot_test1.xlsx')
result2=pd.read_excel('/Users/ellas/OneDrive/Documents/Master/computer science/assignment/result_boot_test2.xlsx')
result3=pd.read_excel('/Users/ellas/OneDrive/Documents/Master/computer science/assignment/result_boot_test3.xlsx')
result4=pd.read_excel('/Users/ellas/OneDrive/Documents/Master/computer science/assignment/result_boot_test4.xlsx')




combined=pd.concat((result0, result1, result2, result3, result4))
group=combined.groupby(combined.index)
average=group.mean()



plt.plot(average['fraction_comparisonLSH'],average['PC'])
plt.ylabel("Pair completeness")
plt.xlabel("Fraction of comparisons")
plt.show()

plt.plot(average['fraction_comparisonLSH'],average['PQ'])
plt.ylabel("Pair quality")
plt.xlabel("Fraction of comparisons")
plt.show()

plt.plot(average['fraction_comparisonLSH'],average['F1StarLSH'])
plt.ylabel("F1 LSH")
plt.xlabel("Fraction of comparisons")
plt.show()

plt.plot(average['fraction_comparisonLSH'],average['F1_cluster'])
plt.ylabel("F1 cluster")
plt.xlabel("Fraction of comparisons")
plt.show()


plt.plot(average['fraction_comparisonLSH'],average['time'])
plt.ylabel("time (seconds)")
plt.xlabel("Fraction of comparisons")
plt.show()

plt.plot(average['F1_cluster'],average['time'])
plt.ylabel("time (seconds)")
plt.xlabel("F1")
plt.show()


