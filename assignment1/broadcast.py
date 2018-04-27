import numpy as np 

A = np.array([[56.0, 0.0, 4.4, 68.0],
              [1.2, 104.0, 52.0, 8.0],
              [1.8, 135.0, 99.0, 0.9]])
#print (A)
 
cal = A.sum(axis=0)
#print (cal)
#print (cal.reshape(1,4))
percentage = 100*A/cal
print (percentage)

B = A + 100
print (B)