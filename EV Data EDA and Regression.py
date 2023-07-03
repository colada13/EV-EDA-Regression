#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import statsmodels.api as sm


# In[2]:


df = pd.read_csv(r"C:\Users\kshiti.sinha\Downloads\ElectricCarData_Clean.csv")


# In[3]:


df.head(5)


# In[4]:


#Finding the number of null values


# In[5]:


df.isnull().sum()


# In[6]:


#there exists no null values


# In[7]:


df.describe()


# In[8]:


df.info()


# In[10]:


a=np.arange(1,104)


# pairplot of all columns based on rapid charger -
# it is used to plot pairwise relationships between variables in a dataset

# In[15]:


sb.pairplot(df,hue = 'RapidCharge')


# In[ ]:


heatmap to show the correlation in the data


# In[16]:


ax= plt.figure(figsize=(15,8))
sb.heatmap(df.corr(),linewidths=1,linecolor='blue',annot=True)


# In[ ]:


frequency of brands in the dataset


# In[17]:


ax= plt.figure(figsize=(20,5))
sb.barplot(x='Brand',y=a,data=df)
plt.grid(axis='y')
plt.title('Brands in the datset')
plt.xlabel('Brand')
plt.ylabel('Frequency')
plt.xticks(rotation=45)


# Popular : Byton,Fiat,Smart
# Not Popular : Polestar

# Top speeds achieved by the cars 

# In[19]:


ax= plt.figure(figsize=(20,5))
sb.barplot(x='Brand',y='TopSpeed_KmH',data=df,palette='Paired')
plt.grid(axis='y')
plt.title('Top Speed achieved by a brand')
plt.xlabel('Brand')
plt.ylabel('Top Speed')
plt.xticks(rotation=45)


# Fastest : Porsche, Lucid and Tesla
# Slowest : Smart

# Range of car
# 

# In[21]:


ax= plt.figure(figsize=(20,5))
sb.barplot(x='Brand',y='Range_Km',data=df,palette='tab10')
plt.grid(axis='y')
plt.title('Maximum Range achieved by a brand')
plt.xlabel('Brand')
plt.ylabel('Range')
plt.xticks(rotation=45)


# Highest : Lucid, Lightyear and Tesla
# Lowest : Smart

# In[ ]:


#Car efficiency


# In[22]:


ax= plt.figure(figsize=(20,5))
sb.barplot(x='Brand',y='Efficiency_WhKm',data=df,palette='hls')
plt.grid(axis='y')
plt.title('Efficiency achieved by a brand')
plt.xlabel('Brand')
plt.ylabel('Efficiency')
plt.xticks(rotation=45)


# Most Efficient : Byton,Jaguar and Audi
# least efficient : Lightyear

# Number of seats in each car

# In[23]:


ax= plt.figure(figsize=(20,5))
sb.barplot(x='Brand',y='Seats',data=df,palette='husl')
plt.grid(axis='y')
plt.title('Seats in a car')
plt.xlabel('Brand')
plt.ylabel('Seats')
plt.xticks(rotation=45)


# Highest : Mercedes,Tesla and Nissan
# Lowest : Smart

# In[25]:


#Price in Euros


# In[26]:


ax= plt.figure(figsize=(20,5))
sb.barplot(x='Brand',y='PriceEuro',data=df,palette='Set2')
plt.title('Price of a Car')
plt.xlabel('Price in Euro')
plt.grid(axis='y')
plt.ylabel('Frequency')
plt.xticks(rotation=45)


# Highest : Lightyear, Porsche and Lucid
# Least : Smart

# In[ ]:


#Type of Plug in for charging


# In[27]:


df['PlugType'].value_counts().plot.pie(figsize=(8,15),autopct='%.0f%%',explode=(.1,.1,.1,.1))
plt.title('Plug Type')


# Most companies use Type 2 CCS and Type 1 CHAdeMo the least

# In[28]:


#Cars and their body style


# In[29]:


df['BodyStyle'].value_counts().plot.pie(figsize=(8,15),autopct='%.0f%%',explode=(0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1))
plt.title('Body Style')


# Most card are either SUV or Hatchback

# In[30]:


#Segment in which the cars fall under


# In[31]:


df['Segment'].value_counts().plot.pie(figsize=(8,15),autopct='%.0f%%',explode=(0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1))
plt.title('Segment')


# Most cars are either C or B type

# In[32]:


#Number of seats


# In[33]:


df['Seats'].value_counts().plot.pie(figsize=(8,15),autopct='%.0f%%',explode=(0.1,0.1,0.1,0.1,0.1))
plt.title('Seats')


# Majority of cars have 5 seats

# In[34]:


x=df[['AccelSec','Range_Km','TopSpeed_KmH','Efficiency_WhKm']]
y=df['PriceEuro']


# Finding linear regression using OLS method

# In[41]:


from sklearn.model_selection import train_test_split


# In[42]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=365)


# In[43]:


#Importing Linear Regression


# In[44]:


from sklearn.linear_model import LinearRegression
lr= LinearRegression()


# In[45]:


lr.fit(X_train, y_train)
pred = lr.predict(X_test)


# In[46]:


#Finding out the R squared value


# In[47]:


from sklearn.metrics import r2_score
r2=(r2_score(y_test,pred))
print(r2*100)


# Around 78% of the dependant variable has been explained by the independant variables

# Putting Yes value as 1 and No value as 0 for Logistic Regression

# In[48]:


df['RapidCharge'].replace(to_replace=['No','Yes'],value=[0, 1],inplace=True)


# In[49]:


y1=df[['RapidCharge']]
x1=df[['PriceEuro']]


# In[50]:


from sklearn.model_selection import train_test_split
X1_train, X1_test, y1_train, y1_test = train_test_split(x1, y1, test_size=0.2,random_state=365)


# In[51]:


#Importing Logistic Regression


# In[52]:


from sklearn.linear_model import LogisticRegression


# In[53]:


log= LogisticRegression()


# In[54]:


log.fit(X1_train, y1_train)
pred1 = log.predict(X1_test)
pred1


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




