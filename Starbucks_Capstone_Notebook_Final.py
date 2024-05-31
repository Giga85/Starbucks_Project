#!/usr/bin/env python
# coding: utf-8

# # Starbucks Capstone Challenge
# 
# ### Introduction
# 
# This data set contains simulated data that mimics customer behavior on the Starbucks rewards mobile app. Once every few days, Starbucks sends out an offer to users of the mobile app. An offer can be merely an advertisement for a drink or an actual offer such as a discount or BOGO (buy one get one free). Some users might not receive any offer during certain weeks. 
# 
# Not all users receive the same offer, and that is the challenge to solve with this data set.
# 
# Your task is to combine transaction, demographic and offer data to determine which demographic groups respond best to which offer type. This data set is a simplified version of the real Starbucks app because the underlying simulator only has one product whereas Starbucks actually sells dozens of products.
# 
# Every offer has a validity period before the offer expires. As an example, a BOGO offer might be valid for only 5 days. You'll see in the data set that informational offers have a validity period even though these ads are merely providing information about a product; for example, if an informational offer has 7 days of validity, you can assume the customer is feeling the influence of the offer for 7 days after receiving the advertisement.
# 
# You'll be given transactional data showing user purchases made on the app including the timestamp of purchase and the amount of money spent on a purchase. This transactional data also has a record for each offer that a user receives as well as a record for when a user actually views the offer. There are also records for when a user completes an offer. 
# 
# Keep in mind as well that someone using the app might make a purchase through the app without having received an offer or seen an offer.
# 
# ### Example
# 
# To give an example, a user could receive a discount offer buy 10 dollars get 2 off on Monday. The offer is valid for 10 days from receipt. If the customer accumulates at least 10 dollars in purchases during the validity period, the customer completes the offer.
# 
# However, there are a few things to watch out for in this data set. Customers do not opt into the offers that they receive; in other words, a user can receive an offer, never actually view the offer, and still complete the offer. For example, a user might receive the "buy 10 dollars get 2 dollars off offer", but the user never opens the offer during the 10 day validity period. The customer spends 15 dollars during those ten days. There will be an offer completion record in the data set; however, the customer was not influenced by the offer because the customer never viewed the offer.
# 
# ### Cleaning
# 
# This makes data cleaning especially important and tricky.
# 
# You'll also want to take into account that some demographic groups will make purchases even if they don't receive an offer. From a business perspective, if a customer is going to make a 10 dollar purchase without an offer anyway, you wouldn't want to send a buy 10 dollars get 2 dollars off offer. You'll want to try to assess what a certain demographic group will buy when not receiving any offers.
# 
# ### Final Advice
# 
# Because this is a capstone project, you are free to analyze the data any way you see fit. For example, you could build a machine learning model that predicts how much someone will spend based on demographics and offer type. Or you could build a model that predicts whether or not someone will respond to an offer. Or, you don't need to build a machine learning model at all. You could develop a set of heuristics that determine what offer you should send to each customer (i.e., 75 percent of women customers who were 35 years old responded to offer A vs 40 percent from the same demographic to offer B, so send offer A).

# # Data Sets
# 
# The data is contained in three files:
# 
# * portfolio.json - containing offer ids and meta data about each offer (duration, type, etc.)
# * profile.json - demographic data for each customer
# * transcript.json - records for transactions, offers received, offers viewed, and offers completed
# 
# Here is the schema and explanation of each variable in the files:
# 
# **portfolio.json**
# * id (string) - offer id
# * offer_type (string) - type of offer ie BOGO, discount, informational
# * difficulty (int) - minimum required spend to complete an offer
# * reward (int) - reward given for completing an offer
# * duration (int) - time for offer to be open, in days
# * channels (list of strings)
# 
# **profile.json**
# * age (int) - age of the customer 
# * became_member_on (int) - date when customer created an app account
# * gender (str) - gender of the customer (note some entries contain 'O' for other rather than M or F)
# * id (str) - customer id
# * income (float) - customer's income
# 
# **transcript.json**
# * event (str) - record description (ie transaction, offer received, offer viewed, etc.)
# * person (str) - customer id
# * time (int) - time in hours since start of test. The data begins at time t=0
# * value - (dict of strings) - either an offer id or transaction amount depending on the record
# 
# **Note:** If you are using the workspace, you will need to go to the terminal and run the command `conda update pandas` before reading in the files. This is because the version of pandas in the workspace cannot read in the transcript.json file correctly, but the newest version of pandas can. You can access the termnal from the orange icon in the top left of this notebook.  
# 
# You can see how to access the terminal and how the install works using the two images below.  First you need to access the terminal:
# 
# <img src="pic1.png"/>
# 
# Then you will want to run the above command:
# 
# <img src="pic2.png"/>
# 
# Finally, when you enter back into the notebook (use the jupyter icon again), you should be able to run the below cell without any errors.

# In[1]:


import pandas as pd
import numpy as np
import math
import json

import matplotlib.pyplot as plt
import re
import random
import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats

get_ipython().run_line_magic('matplotlib', 'inline')
# read in the json files
portfolio = pd.read_json('data/portfolio.json', orient='records', lines=True)
profile = pd.read_json('data/profile.json', orient='records', lines=True)
transcript = pd.read_json('data/transcript.json', orient='records', lines=True)


# #### Gathering data

# #### portfolio data frame

# In[2]:


portfolio.head()


# In[3]:


portfolio.shape


# In[4]:


# Check if there are nulls
portfolio.isnull().sum()


# #### profile data frame

# In[5]:


profile.head()


# In[6]:


# Check if there are nulls
profile.isnull().sum()


# In[7]:


profile.shape


# Missing data for gender, income is present for 2175 out of 17000. It is quite high

# In[8]:


profile.describe()


# We can see that, age is very high

# In[9]:


# Check age distribution
plt.hist(profile.age, bins = 20)


# The rate of missing data for gender, income and age is high, so we don't want to remove them. We will impute data so they are a distinct group.

# #### transcript data frame

# In[10]:


transcript.head()


# In[11]:


# Check if there are nulls
transcript.isnull().sum()


# In[12]:


transcript.shape


# #### Data cleaning

# #### 1. Portfolio
# Columns channels and offer_type should be broken up into columns with binary values.

# In[13]:


# Create columns of 1's and 0's for channel type
channel_items = ['web','email','mobile','social']

for channel in portfolio['channels']:
    for item in channel_items:
        if item in channel:
            portfolio[item] = 1
        else:
            portfolio[item] = 0 


# In[14]:


portfolio_dummies = pd.get_dummies(portfolio['offer_type'])


# In[15]:


# break offer_type into columns
portfolio_merged = pd.concat([portfolio, portfolio_dummies],axis=1)


# In[16]:


# Drop columns channels and offer_type
portfolio_clean = portfolio_merged.drop(['channels','offer_type'],axis=1)


# In[17]:


portfolio_clean.head()


# #### 2. Profile data frame

# Null values in column gender will be changed to unknown

# In[18]:


profile['gender'][profile['gender'].isnull()] = 'unknown'


# break offer_type into columns

# In[19]:


profile_dummies = pd.get_dummies(profile['gender'])
profile_merged = pd.concat([profile, profile_dummies],axis=1)


# In[20]:


profile_merged.head()


# Check income distribution

# In[21]:


plt.hist(profile_merged['income'][profile_merged['income'].notnull()], bins = 20)


# Impute random incomes into income

# In[22]:


num_null = profile_merged['income'].isnull().sum()
income_samples = random.choices(list(profile_merged['income'][profile_merged['income'].notnull()]),k=num_null)
n = 0

for idx in range(0,profile_merged.shape[0]):
    if math.isnan(profile_merged['income'][idx]):
        profile_merged['income'][idx] = income_samples[n]
        n+=1


# In[23]:


# Check new income distribution
plt.hist(profile_merged['income'], bins = 20)


# Replace special values in age column to a random sample

# In[24]:


#Change age == 118 to a random sample
num_118 = sum(profile_merged['age']==118)

age_samples = random.choices(list(profile_merged['age'][profile_merged['age']!=118]),k=num_118)
n = 0

for idx in range(0,profile_merged.shape[0]):
    if profile_merged['age'][idx] == 118:
        profile_merged['age'][idx] = age_samples[n]
        n+=1


# In[25]:


#Check new distribution
plt.hist(profile_merged['age'], bins = 20)


# Convert became_member_on to days_as_member

# In[26]:


# Change became_member_on to days_as_member
days_as_member = []
today = datetime.datetime.today()
for idx in range(0,profile_merged.shape[0]):
    start_date = datetime.datetime.strptime(str(profile_merged['became_member_on'][idx]),"%Y%m%d")
    days_as_member.append((today-start_date).days)
profile_merged['days_as_member'] = days_as_member


# Remove columns 'gender','became_member_on'

# In[27]:


profile_clean = profile_merged.drop(['gender','became_member_on'],axis=1)


# In[28]:


#Check data 
profile_clean.head(10)


# #### 2. Transcript 

# Pull out values of offer_id and break into columns

# In[29]:


# Pull offer id, reward, and transaction amount out of value
# Break event into multiple columns
people = transcript['person'].drop_duplicates()
transaction_df = pd.DataFrame({'person':[],'offer_id':[],'offer_recieved':[],
                              'offer_viewed':[],'offer_completed':[],
                              'reward':[],'transaction_amount':[]})

for person in people:
# for each person

    #create df of each person
    person_df = transcript[transcript['person']==person]

    #create offer recieved df
    person_event_df = person_df[person_df['event']=='offer received']
    
    #Create blank lists to append
    offer_id_list = []
    offer_recieved_list = []
    offer_viewed_list = []
    offer_completed_list = []
    reward_list = []
    transaction_amount_list = []

    # Go through each person
    for idx in range(0,person_event_df.shape[0]):
        #Get offer id
        offer_id = person_event_df.iloc[idx,:]['value']['offer id']
        offer_id_list.append(offer_id)

        #Subset by offer duration
        offer_duration = (portfolio_clean[portfolio_clean['id'] == offer_id]['duration'])*24
        initial_time = person_event_df.iloc[idx,:]['time']
        end_time = int(initial_time + offer_duration)
        start_window_df = person_df[person_df['time'] >= initial_time]
        time_window_df = start_window_df[start_window_df['time']<=end_time]

        #set offer recieved status
        or_time_window_df = time_window_df[time_window_df['event']=='offer received']
        n=0
        for event in or_time_window_df['value']:
            if event['offer id'] == offer_id:
                n+=1
        n = 1 if n>=1 else 0
        offer_recieved_list.append(n)

        #set offer viewed status
        ov_time_window_df = time_window_df[time_window_df['event']=='offer viewed']
        n=0
        for event in ov_time_window_df['value']:
            if event['offer id'] == offer_id:
                n+=1
        n = 1 if n>=1 else 0
        offer_viewed_list.append(n)

        #set offer completed status
        oc_time_window_df = time_window_df[time_window_df['event']=='offer completed']
        n=0
        for event in oc_time_window_df['value']:
            if event['offer_id'] == offer_id:
                n+=1
                reward = event['reward']
        n = 1 if n>=1 else 0
        offer_completed_list.append(n)
        reward_list.append(reward)

        #calculate transaction amount
        t_time_window_df = time_window_df[time_window_df['event']=='transaction']
        n=0
        for event in t_time_window_df['value']:
            n += event['amount']
        transaction_amount_list.append(n)    

    num_rows = len(offer_id_list)    
    person_list = [person] * num_rows
    
    #Put it all the data to gether for each person
    interaction_df = pd.DataFrame({'person':person_list,'offer_id':offer_id_list,'offer_recieved':offer_recieved_list,
                                  'offer_viewed':offer_viewed_list,'offer_completed':offer_completed_list,
                                  'reward':reward_list,'transaction_amount':transaction_amount_list})
    
    # add person info to transaction_df
    
    transaction_df = pd.concat([transaction_df,interaction_df],axis=0, ignore_index=True)


# Create a new column showing transaction

# In[30]:


# Create column showing transaction minus reward
transaction_df['net_transaction'] = transaction_df['transaction_amount'] - transaction_df['reward']


# In[31]:


transaction_df.head()


# Merge 3 data sets and then drop rows where an individual did not view the offer.

# Rename columns

# In[32]:


#Change column name id to person
profile_clean2 = profile_clean.rename({'id':'person'}, axis=1)


# In[33]:


#Chnage column name id to offer_id
portfolio_clean2 = portfolio_clean.rename({'id':'offer_id'},axis=1)


# Merge datasets

# In[34]:


#Combine profile data
prelim_df = transaction_df.merge(profile_clean2, on ='person')
prelim_df.head(8)


# In[35]:


# Combine portfolio data
final_df = prelim_df.merge(portfolio_clean2, on = 'offer_id')


# In[36]:


final_df.head()


# Drop rows

# In[37]:


#Drop rows where individual did not view offer
final_df = final_df.drop(final_df[final_df['offer_viewed']==0].index,axis=0)


# In[38]:


def clean_data(portfolio=portfolio, profile=profile, transcript=transcript):
    '''
    Input:
    - portfolio = portfolio data
    - profile = profile data
    - transcript = transcript data
    
    Output:
    - final_df = cleaned dataframe that combines portfolio, profile, and transcript data
    
    
    '''
    ########################
    #Portfolio Cleaning
    # Create columns of 1's and 0's for channel type
    channel_items = ['web','email','mobile','social']

    for channel in portfolio['channels']:
        for item in channel_items:
            if item in channel:
                portfolio[item] = 1
            else:
                portfolio[item] = 0 

    # break offer_type into columns
    portfolio_merged = pd.concat([portfolio, pd.get_dummies(portfolio['offer_type'])],axis=1)  

    # Drop columns channels and offer_type
    portfolio_clean = portfolio_merged.drop(['channels','offer_type'],axis=1)

    ##########################
    #Profile Cleaning
    # Input unknown into null values
    profile['gender'][profile['gender'].isnull()] = 'unknown'
    
    # break offer_type into columns
    profile = pd.concat([profile, pd.get_dummies(profile['gender'])],axis=1)    

    # Impute random incomes into income==null
    num_null = profile['income'].isnull().sum()

    income_samples = random.choices(list(profile['income'][profile['income'].notnull()]),k=num_null)
    n = 0

    for idx in range(0,profile.shape[0]):
        if math.isnan(profile['income'][idx]):
            profile['income'][idx] = income_samples[n]
            n+=1

    #Change age == 118 to a random sample
    num_118 = sum(profile['age']==118)

    age_samples = random.choices(list(profile['age'][profile['age']!=118]),k=num_118)
    n = 0

    for idx in range(0,profile.shape[0]):
        if profile['age'][idx] == 118:
            profile['age'][idx] = age_samples[n]
            n+=1      

    # Change became_member_on to days_as_member
    days_as_member = []
    today = datetime.datetime.today()
    for idx in range(0,profile.shape[0]):
        start_date = datetime.datetime.strptime(str(profile['became_member_on'][idx]),"%Y%m%d")
        days_as_member.append((today-start_date).days)
    profile['days_as_member'] = days_as_member

    profile_clean = profile.drop(['gender','became_member_on'],axis=1)

    ###########################
    #Transcript cleaing
    # Pull offer id, reward, and transaction amount out of value
    # Break event into multiple columns
    people = transcript['person'].drop_duplicates()
    transaction_df = pd.DataFrame({'person':[],'offer_id':[],'offer_recieved':[],
                                  'offer_viewed':[],'offer_completed':[],
                                  'reward':[],'transaction_amount':[]})

    for person in people:
    # for each person

        #create df of each person
        person_df = transcript[transcript['person']==person]

        #create offer recieved df
        person_event_df = person_df[person_df['event']=='offer received']

        #Create blank lists to append
        offer_id_list = []
        offer_recieved_list = []
        offer_viewed_list = []
        offer_completed_list = []
        reward_list = []
        transaction_amount_list = []

        # Go through each person
        for idx in range(0,person_event_df.shape[0]):
            #Get offer id
            offer_id = person_event_df.iloc[idx,:]['value']['offer id']
            offer_id_list.append(offer_id)

            #Subset by offer duration
            offer_duration = (portfolio_clean[portfolio_clean['id'] == offer_id]['duration'])*24
            initial_time = person_event_df.iloc[idx,:]['time']
            end_time = int(initial_time + offer_duration)
            start_window_df = person_df[person_df['time'] >= initial_time]
            time_window_df = start_window_df[start_window_df['time']<=end_time]

            #set offer recieved status
            or_time_window_df = time_window_df[time_window_df['event']=='offer received']
            n=0
            for event in or_time_window_df['value']:
                if event['offer id'] == offer_id:
                    n+=1
            #n = 1 if n>=1 else 0
            offer_recieved_list.append(n)

            #set offer viewed status
            ov_time_window_df = time_window_df[time_window_df['event']=='offer viewed']
            n=0
            for event in ov_time_window_df['value']:
                if event['offer id'] == offer_id:
                    n+=1
            #n = 1 if n>=1 else 0
            offer_viewed_list.append(n)

            #set offer completed status
            oc_time_window_df = time_window_df[time_window_df['event']=='offer completed']
            n=0
            for event in oc_time_window_df['value']:
                if event['offer_id'] == offer_id:
                    n+=1
                    reward = event['reward']
            #n = 1 if n>=1 else 0
            offer_completed_list.append(n)
            reward_list.append(reward)

            #set offer recieved status
            t_time_window_df = time_window_df[time_window_df['event']=='transaction']
            n=0
            for event in t_time_window_df['value']:
                n += event['amount']
            transaction_amount_list.append(n)    

        num_rows = len(offer_id_list)    
        person_list = [person] * num_rows

        #Put it all the data to gether for each person
        interaction_df = pd.DataFrame({'person':person_list,'offer_id':offer_id_list,'offer_recieved':offer_recieved_list,
                                      'offer_viewed':offer_viewed_list,'offer_completed':offer_completed_list,
                                      'reward':reward_list,'transaction_amount':transaction_amount_list})

        # add person info to transaction_df

        transaction_df = pd.concat([transaction_df,interaction_df],axis=0, ignore_index=True)
        transaction_df['net_transaction'] = transaction_df['transaction_amount'] - transaction_df['reward']


    #Change column name id to person
    profile_clean2 = profile_clean.rename({'id':'person'}, axis=1)

    #Chnage column name id to offer_id
    portfolio_clean2 = portfolio_clean.rename({'id':'offer_id'},axis=1)

    #Combine profile data
    prelim_df = transaction_df.merge(profile_clean2, on ='person')

    # Combine portfolio data
    final_df = prelim_df.merge(portfolio_clean2, on = 'offer_id')

    #Drop rows where individual did not view offer
    final_df = final_df.drop(final_df[final_df['offer_viewed']==0].index,axis=0)
    
    return final_df


# In[39]:


df=clean_data()


# #### Check values to see if everything matches

# In[40]:


transcript['event'].value_counts()


# In[41]:


transcript[transcript['event']=='offer completed'].shape


# In[42]:


transcript[transcript['event']=='offer viewed'].shape


# In[43]:


transcript[transcript['event']=='offer received'].shape


# #### Modeling

# #### Check number of bogo, Discount, and informationals offers

# In[44]:


print('Bogo: ', df['bogo'].sum())
print('Discount: ', df['discount'].sum())
print('Informational: ', df['informational'].sum())


# #### Create function to split data, train, and test a model

# In[45]:


def train_test_model(X,y):
    '''
    Input:
    X: Demographic and offer details
    y: Prediction net_transaction
    
    Output:
    Prints a number of metrics related to the model
    
    Summary:  This function will split data into test and train sets, train and test the model,
    and print metrics related to the ability of the model   
    
    '''
    #Split into test and train
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33, random_state=42)

    # Fit and predict model
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_preds = model.predict(X_test)

    #Check metrics
    print('R Squared: ',r2_score(y_test,y_preds))
    print('Percent Difference: ',abs(y_test.sum()-y_preds.sum())/y_test.sum())

    # Check for significance
    #REF https://stackoverflow.com/questions/27928275/find-p-value-significance-in-scikit-learn-linearregression
    X2 = sm.add_constant(X)
    est = sm.OLS(y, X2)
    est2 = est.fit()
    print(est2.summary())
    
    
# Split data into X and y
X = df[['age','M','F','O','unknown','income','days_as_member','difficulty','duration','reward_y','web','email','mobile','social',
       'bogo','discount','informational']]
y = df['net_transaction']



# General model
train_test_model(X,y)


# #### Test Bogo offers only

# In[46]:


# Split data into X and y
X = df[['age','M','F','O','unknown','income','days_as_member','difficulty','duration','reward_y','web','email','mobile','social',
       'bogo','discount','informational']][df['bogo']==1]
y = df['net_transaction'][df['bogo']==1]

# General model
train_test_model(X,y)


# #### Test discount offers only

# In[47]:


# Split data into X and y
X = df[['age','M','F','O','unknown','income','days_as_member','difficulty','duration','reward_y','web','email','mobile','social',
       'bogo','discount','informational']][df['discount']==1]
y = df['net_transaction'][df['discount']==1]

# General model
train_test_model(X,y)


# #### Test informational offers only

# In[48]:


# Split data into X and y
X = df[['age','M','F','O','unknown','income','days_as_member','difficulty','duration','reward_y','web','email','mobile','social',
       'bogo','discount','informational']][df['informational']==1]
y = df['net_transaction'][df['informational']==1]

# General model
train_test_model(X,y)


# #### Modeling Offers 

# In[49]:


# Split data into X and y
bogo_df = df[df['discount']==1]
X = bogo_df[['age','M','F','O','unknown','income','days_as_member','difficulty','duration','reward_y','web','email','mobile','social',
       'bogo','discount','informational']]
y = bogo_df['offer_completed']

# Split into test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Fit and predict model
model = LinearRegression()
model.fit(X_train, y_train)
y_preds = model.predict(X_test)

# optimize f1_score
for i in [0.62,0.63,0.64,.65,0.66,0.67]:
    y_test_predictions_high_recall = [1 if (x >= i) else 0 for x in y_preds]
    print('Threshold for promotion: ',i)
    print('Accuracy: ', accuracy_score(y_test,y_test_predictions_high_recall))
    print('Precision: ', precision_score(y_test,y_test_predictions_high_recall,average='micro'))
    print('Recall: ', recall_score(y_test,y_test_predictions_high_recall,average='micro'))
    print('F1 Score: ', f1_score(y_test,y_test_predictions_high_recall,average='micro'))
    print(' ')


# #### Data Visualization and Analysis

# Net Transactions

# Firstly, I will visualize the relationship between net transactions and days, gender, income as member, and offer type.

# In[50]:


#Average Net Transaction by Gender
# Calculate values for bar graph
avg_male_net = df[df['M']==1]['net_transaction'].sum()/df['M'].sum()
avg_female_net = df[df['F']==1]['net_transaction'].sum()/df['F'].sum()
avg_other_net = df[df['O']==1]['net_transaction'].sum()/df['O'].sum()
avg_unknown_net = df[df['unknown']==1]['net_transaction'].sum()/df['unknown'].sum()

# Make Bar graph
objects = ('Male', 'Female', 'Other', 'Unknown')
y_pos = np.arange(len(objects))
performance = [avg_male_net,avg_female_net,avg_other_net,avg_unknown_net]

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Average Net Transaction ($)')
plt.title('Average Net Transaction by Gender')

plt.show()


# From the graph we can see that the average net transaction of those who's gender is uknown is close to zero. 
# Females on average spend most while males spend the least

# 
# Average Net Transaction by Offer Type

# In[51]:


# Calculate values for bar graph
avg_bogo_net = df[df['bogo']==1]['net_transaction'].sum()/df['bogo'].sum()
avg_discount_net = df[df['discount']==1]['net_transaction'].sum()/df['discount'].sum()
avg_informational_net = df[df['informational']==1]['net_transaction'].sum()/df['informational'].sum()

# Make Bar graph
objects = ('Bogo', 'Discount', 'informational')
y_pos = np.arange(len(objects))
performance = [avg_bogo_net,avg_discount_net,avg_informational_net]

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Average Net Transaction ($)')
plt.title('Average Net Transaction by Offer Type')

plt.show()


# Net transactions are highest on average for discounts and lowest for informationals.

# Avg Net Transaction vs Income

# In[52]:


N=df.shape[0]

x_ax = df['income']
y_ax = df['net_transaction']
colors = np.random.rand(N)
area = (30 * np.random.rand(N))**2  # 0 to 15 point radii

plt.scatter(x_ax, y_ax, s=area, c=colors, alpha=0.1)
plt.ylabel('Average Net Transaction ($)')
plt.xlabel('Income')
plt.title('Avg Net Transaction vs Income')
plt.show()


# Higher net transactions have a higher probability given a higher income.

# Avg Net Transaction vs Days as Member

# In[53]:


x_ax = df['days_as_member']
y_ax = df['net_transaction']
colors = np.random.rand(N)
area = (30 * np.random.rand(N))**2  # 0 to 15 point radii

plt.scatter(x_ax, y_ax, s=area, c=colors, alpha=0.1)
plt.ylabel('Average Net Transaction ($)')
plt.xlabel('Days as Member')
plt.title('Avg Net Transaction vs Days as Member')
plt.show()


# Avg Net Transaction vs Age

# In[54]:


x_ax = df['age']
y_ax = df['net_transaction']
colors = np.random.rand(N)
area = (30 * np.random.rand(N))**2  # 0 to 15 point radii

plt.scatter(x_ax, y_ax, s=area, c=colors, alpha=0.1)
plt.ylabel('Average Net Transaction ($)')
plt.xlabel('Age')
plt.title('Avg Net Transaction vs Age')
plt.show()


# Offers Completed

# The below visualizations show the relationships between offers completed and income, gender, age, offer type, and days as member.

# Create data frame of the offers completed

# In[55]:


offer_df = df[df['offer_completed']==1]


# Offers Completed by Offer

# In[56]:


objects = ('Bogo', 'Discount', 'Informational')
y_pos = np.arange(len(objects))
performance = [offer_df['bogo'].sum(),offer_df['discount'].sum(),offer_df['informational'].sum()]

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Number of Offers Completed')
plt.title('Offers Completed by Offer')

plt.show()


# It can be seen that the numbers of Bogo and dicounts are similar.

# Offers Completed by Gender

# In[57]:


objects = ('Male', 'Female', 'Other', 'Unknown')
y_pos = np.arange(len(objects))
performance = [offer_df['M'].sum(),offer_df['F'].sum(),offer_df['O'].sum(),offer_df['unknown'].sum()]

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Number of Offers Completed')
plt.title('Offers Completed by Gender')

plt.show()


# The graph shows that the number of offers done by Males and females are similar. 

# Age distribution

# In[58]:


plt.hist(offer_df['age'], bins = 20)


# In[ ]:





# Days as member distribution

# In[59]:


plt.hist(offer_df['days_as_member'], bins = 20)


# Income distribution

# In[60]:


plt.hist(offer_df['income'], bins = 20)


# #### Offers Completed Summary
# 
# Offers completed for age, income, and days as member mirror the population histograms for each.

# #### The relationships between net_transaction and oder_completed.

# In[61]:


# Check for correlations
df_treat = df[['age','M','F','O','unknown','income','days_as_member','difficulty','duration','reward_y',
       'bogo','discount','informational','transaction_amount','net_transaction','offer_completed']]

corr = df_treat.corr()
corr.style.background_gradient(cmap='coolwarm').set_precision(2)


# There are a small number of correlations, The maximum is 0.37, and the minimum is 0.17.
# 
# #### Offer Completed:
# 
# Positive: Female, income, days_as_member, difficulty, duration, reward, discount.
# Negative: Unknown gender.
# Noteworthy: There is a positive correlation around age, female, and income with eachother.
#     
# #### Net Transaction:
# 
# Positive: Female, income, days_as_member, difficulty, duration, discount.
# Negative: Unknown gender, informational.

# In[62]:


print('done')


# In[ ]:




