
# coding: utf-8

# In[25]:

import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[17]:

import pandas as pd


# In[18]:

Boston = load_boston()
Boston_df = pd.DataFrame(Boston.data, columns= Boston.feature_names)


# In[19]:

Boston_df['y_MEDV'] = Boston.target


# In[20]:

Boston_df.head()


# In[23]:

X_train, X_test, y_train, y_test = train_test_split(Boston_df[Boston.feature_names], Boston_df['y_MEDV'], test_size=0.30)


# In[26]:

#With training data
linreg = LinearRegression().fit(X_train,y_train)


# In[ ]:




# In[47]:

#take absolute value since we only care about magnitude
coef = {'Feature':Boston.feature_names,'Coef Value':np.abs(linreg.coef_)}
pd.DataFrame(coef).sort_values(by = 'Coef Value', ascending = False)


# In[ ]:




# In[53]:

from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets


# In[54]:

Iris = pd.datasets.load_iris()
Wine = datasets.load_wine()


# In[58]:

Iris_df = pd.DataFrame(Iris.data,columns = Iris.feature_names)
Wine_df = pd.DataFrame(Wine.data,columns = Wine.feature_names)


# In[ ]:




# In[68]:

#Using the elbow heuristic we can see that the "elbow" in this case would be 3 clusters.
distance = []
for k in range(1,10):
    kmeanModel = KMeans(n_clusters=k).fit(Iris_df)
    distance.append(kmeanModel.inertia_)
# Plot the elbow
plt.plot(range(1,10), distance)
plt.xlabel('k')
plt.ylabel('Distance')
plt.title('The Elbow Heuristic (Iris)')
plt.show()


# In[69]:

#Using the elbow heuristic we can see that the "elbow" in this case would be 3 clusters.
distance = []
for k in range(1,10):
    kmeanModel = KMeans(n_clusters=k).fit(Wine_df)
    distance.append(kmeanModel.inertia_)
# Plot the elbow
plt.plot(range(1,10), distance)
plt.xlabel('k')
plt.ylabel('Distance')
plt.title('The Elbow Heuristic (Wine)')
plt.show()

