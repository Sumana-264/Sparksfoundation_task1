#!/usr/bin/env python
# coding: utf-8

# # SUMANA BUSHIREDDY
# 
# # PREDICTION USING SUPERVISED ML 
# 

# In this task we will predict the percentage of marks that a student is expected to score based upon the number of hours they studied. This is a simple linear regression task as it involves just two variables.
# 
# Data is given in : http://bit.ly/w-data

# In[1]:


#importing all the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


#reading the data
url = "http://bit.ly/w-data"
data = pd.read_csv(url)
data


# In[3]:


#copying the original dataset into other new variable to make any changes
data_copy = pd.DataFrame(data)
data_copy


# Let's plot our data points on 2-D graph to our dataset and see if we can manually find any relationship between the data. We can create the plot with the following code:

# In[4]:


data_copy.plot(x='Hours', y='Scores', style='o').grid()  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# 
# 
# From the graph above, we can clearly see that there is a positive linear relation between the number of hours studied and percentage of score.
# 

# # Preparing data

# dividing data into inputs and targets

# In[5]:


x = data_copy.iloc[:, :-1].values  
y = data_copy.iloc[:, 1].values


# Splitting dataset into training and testing dataset using Scikit-Learn's built-in train_test_split() method

# In[7]:


from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                            test_size=0.2, random_state=0)


# # Training our algorithm
# In the previous step we splited the dataset into training and testing dataset, now it's time to train our algorithm

# In[9]:


from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(x_train, y_train) 


# Now we should plot the regression line for the above graph

# In[10]:


# Plotting the regression line
line = regressor.coef_*x+regressor.intercept_

# Plotting for the test data
po = plt.scatter(x, y)
plt.plot(x, line)
plt.show()


# Now we should make predictions
# 
# In the previous step we trained our algorithm, now it's time to make some predictions

# In[11]:


#according to the given task
#predicting with 9.25 hrs per day.
hours = np.array([9.25]).reshape(-1,1)
own_pred = regressor.predict(hours)
print("No of Hours = {}".format(hours[0][0]))
print("Predicted Score = {}".format(own_pred[0]))


# In[12]:


#predicting with 5 hrs per day.
hours = np.array([5]).reshape(-1,1)
own_pred = regressor.predict(hours)
print("No of Hours = {}".format(hours[0][0]))
print("Predicted Score = {}".format(own_pred[0]))


# In[15]:


predictions = regressor.predict(x_test)
print(predictions)


# # Comparing actual vs predicted
# 

# In[19]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})  
df


# # Evaluating the model
# The last step to do is evaluating the model
# Though there are many methods to do, i'm following the easy and efficient method called mean square error

# In[20]:


from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, predictions))


# In[ ]:




