import numpy as np
import matplotlib.pyplot as plt
 
# create a 8x8 matrix of two numbers-0 and 1. 
# O represents dark color and 1 represents bright color 
arr=np.array([[1,0]*4,[0,1]*4]*4)
print(arr)
# use the imshow function to display the image made from the above array
plt.imshow(arr)