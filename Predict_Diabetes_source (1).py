#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


dataset = pd.read_csv("diabetes.csv")
dataset


# In[4]:


dataset.info()


# In[5]:


dataset.isnull().sum()


# In[6]:


dataset.describe()


# In[7]:


plt.figure(figsize=(10,8))
sns.heatmap(dataset.corr(), annot=True , fmt=".3f", cmap="YlGnBu")
plt.title("correlation plt of variable")
plt.show()


# In[12]:


plt.figure(figsize=(10,8))

sns.kdeplot(
    x=dataset.loc[dataset["Outcome"] == 1, "Pregnancies"],
    color="red",
    fill=True,
    label="Positive"
)

sns.kdeplot(
    x=dataset.loc[dataset["Outcome"] == 0, "Pregnancies"],
    color="blue",
    fill=True,
    label="Negative"
)

plt.xlabel("Pregnancies")
plt.ylabel("Density")
plt.legend()

plt.show()


# In[46]:


plt.figure(figsize=(10,8))
sns.violinplot(data=dataset,x="Outcome",y="Glucose",split=True,inner="quart",linewidth=2)


# In[15]:


plt.figure(figsize=(10,8))

sns.kdeplot(
    x=dataset.loc[dataset["Outcome"] == 1, "Glucose"],
    color="red",
    fill=True,
    label="Positive"
)

sns.kdeplot(
    x=dataset.loc[dataset["Outcome"] == 0, "Glucose"],
    color="blue",
    fill=True,
    label="Negative"
)

plt.xlabel("Glucose")
plt.ylabel("Density")
plt.title("Glucose vs Outcome")
plt.legend()

plt.show()


# In[16]:


#replace 0 values with the mean\median
#glucose
dataset["Glucose"]=dataset["Glucose"].replace(0,dataset["Glucose"].median())
#blood pressure
dataset["BloodPressure"]=dataset["BloodPressure"].replace(0,dataset["BloodPressure"].median())
#skin thickness
dataset["SkinThickness"]=dataset["SkinThickness"].replace(0,dataset["SkinThickness"].mean())
#Insulin
dataset["Insulin"]=dataset["Insulin"].replace(0,dataset["Insulin"].mean())
#BMI
dataset["BMI"]=dataset["BMI"].replace(0,dataset["BMI"].mean())


# In[17]:


dataset


# In[18]:


x=dataset.drop(["Outcome"],axis=1)
y=dataset["Outcome"]


# In[19]:


x


# In[20]:


y


# In[21]:


#splitting the dataset into training and testing dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=42)


# In[22]:


x_train


# In[23]:


#knn
from sklearn.neighbors import KNeighborsClassifier
training_accuracy=[]
test_accuracy=[]
for n_neighbors in range(1,11):
  knn=KNeighborsClassifier(n_neighbors=n_neighbors)
  knn.fit(x_train,y_train)
  #check accuracy
  training_accuracy.append(knn.score(x_train,y_train))
  test_accuracy.append(knn.score(x_test,y_test))


# In[24]:


plt.plot(range(1,11),training_accuracy,label="training accuracy")
plt.plot(range(1,11),test_accuracy,label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()


# In[25]:


knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,y_train)
print(knn.score(x_train,y_train),":Training accuracy")
print(knn.score(x_test,y_test),":Testing accuracy")


# In[26]:


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(random_state=0)
dt.fit(x_train,y_train)
print(dt.score(x_train,y_train),":Training accuracy")
print(dt.score(x_test,y_test),":Testing accuracy")


# In[28]:


dt1=DecisionTreeClassifier(max_depth=3,random_state=0)
dt1.fit(x_train,y_train)
print(dt1.score(x_train,y_train),":Training accuracy")
print(dt1.score(x_test,y_test),":Testing accuracy")


# In[36]:


from sklearn.neural_network import MLPClassifier
mlp=MLPClassifier(random_state=42)
mlp.fit(x_train,y_train)
print(mlp.score(x_train,y_train),":Training accuracy")
print(mlp.score(x_test,y_test),":Testing accuracy")


# In[42]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit(x_train)
x_train_scaled=sc.fit_transform(x_train)
x_test_scaled=sc.fit_transform(x_test)


# In[45]:


mlp1=MLPClassifier(random_state=0)
mlp1.fit(x_train_scaled,y_train)
print(mlp1.score(x_train_scaled,y_train),":Training accuracy")
print(mlp1.score(x_test_scaled,y_test),":Testing accuracy")


# In[ ]:




