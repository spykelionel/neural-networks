# -*- coding: utf-8 -*-
"""
Created on Mon May  9 18:19:03 2022

@author: protech
"""

import matplotlib.pyplot as plt
x = np.arange(len(error_list))
y = error_list
plt.figure(figsize=(10,8))
plt.plot(x,y)
plt.xlabel("iteration")
plt.ylabel("error")
plt.title("Artificial Neural Networks Training")
plt.show()