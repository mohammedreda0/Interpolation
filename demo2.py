import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 


# data = pd.read_csv("ECGNormal.csv")
# x = data['# t']
# y = data['x']
# xnew = np.linspace(x[0], x[len(x)-1], num=20000)
# a = np.polyfit(x, y, 5)

def LateX(degree):
    o=""
    y=range(degree)
    for i in range(len(a)):
        if i!=len(x)-1:
            o+= str("{:.3f}".format(a[i]))+ "$X^" +str(len(a)-1-i)+"$+" 
        else:
            o+=str("{:.3f}".format(a[i]))
    return o

fig = plt.figure()
ax = fig.add_subplot()
fig.subplots_adjust(top=0.85)

ax.text(0.5, 6, LateX(5), fontsize=15)

ax.set(xlim=(0, 10), ylim=(0, 10))
plt.axis('off')
plt.show()