# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 14:05:56 2020

@author: acn980
"""

from matplotlib.pyplot import plt

#Here all_hz is a DataFrame with as index dates and 4 different columns, on for each hazard type.
#The index goes from 1980-01-01 until 2016-01-01, with a daily timestep
#In each columns I put a 1,2,3 or 4 on the days where there was a tropical cyclone, flood, earthquake or volcano outburt, respectively

f = plt.figure(figsize=(10,2))
ax = plt.axes()
#f.patch.set_visible(False)
plt.scatter(all_hz.index, all_hz['TC'], c='blue', marker="+", s=markersize) 
plt.scatter(all_hz.index, all_hz['FL'], c='lightblue', marker="+", s=markersize) #marker="|"
plt.scatter(all_hz.index, all_hz['EQ'], c='red' ,marker="+", s=markersize)
plt.scatter(all_hz.index, all_hz['VO'], c='saddlebrown', marker="+", s=markersize)
plt.ylim([0.5,3])
plt.yticks([])
#plt.xticks(pd.date_range(start='1980-01-01', end='2016-01-01', freq='AS'))
ax.xaxis.set_minor_locator(years)
#ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
plt.show()
