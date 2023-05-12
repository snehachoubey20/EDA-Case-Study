#!/usr/bin/env python
# coding: utf-8

# In[1]:


####The following case study will give the idea and understanding of risk analytics in Banking and Finance sectors. The study will provide with the understanding of the risks involved in lending the amount to customers as loan and the methods to minimize the risks involved so that the loans are credited to the customers who are eligible and to minimize the risks of lending the amount to defaulters or the prospective defaulters. 


# In[ ]:


### Problem Statement1:
##when any institutions receives a loan application they assess the profile of the customer and then decide if the customer is eligible for the loan or not. 
##Two majors risks that are involved with the decision is :-
    #1) If the applicant is likely to repay the loan, then not approving the loan results in a loss of business to the company.
 #2)If the applicant is not likely to repay the loan, i.e. he/she is likely to default, then approving the loan may lead to a financial loss for the company.


# In[ ]:


# In this study we will study with the help of EDA how the tendency to default is influenced by the customer attribute and loan attribute. 


# In[1]:


##import warning and libraries.

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_rows', None)


# In[2]:


## read the dataset "application data" in app dataframe. Displaying top 5 records.

app = pd.read_csv(r'C:\Users\sneha\Downloads\application_data (1).csv')
app.head()


# In[3]:


##checking shape of the dataframe.

app.shape


# In[4]:


##checking the info of the dataframe.

app.info()


# In[ ]:


## Data cleaning:- Treating null values.


# In[5]:


##checking the columns having null values higher than 50%. 

emptycol=app.isnull().sum()/len(app)*100
emptycol=emptycol[emptycol.values>50.0]
print(emptycol)
len(emptycol)


# In[6]:


##dropping the columns having null values higher than 50%.

emptycol = list(emptycol[emptycol.values>=50.0].index)
app.drop(labels=emptycol,axis=1,inplace=True)
print(len(emptycol))


# In[7]:


##checking the shape of the dataframe acfter removing the columns having null values higher than 50%.

app.shape


# In[8]:


##Checking the percentage of null values in the remaining columns of the datframe.

app.isnull().sum()/len(app)*100


# In[ ]:


##By removing the columns having high null values we get a dataframe having lesser null values and is more easy to analyse. 
##The present dataframe has columns with lesser than 50% null values.


# In[ ]:


##Analysing the dataframe for the columns having null values lesser than 19% and to check if we need to replace the data in those columns.


# In[10]:


##checking the columns having null values lesser than 19%.

emptycol=app.isnull().sum()/len(app)*100
emptycol=emptycol[emptycol.values<19.0]
print(emptycol)
len(emptycol)


# In[11]:


##Analysing the column for "AMT_ANNUITY" creating a box plot to detect the outliers.
## detecting outliers will help us to clean the data further for the analysis.


plt.figure(figsize=(7,8))
sns.boxplot(y=app['AMT_ANNUITY'])
plt.yscale('log') 
plt.title("Analysis of AMT_ANNUTY",fontsize=16)
plt.show()


# In[12]:


##description of the column "AMT_ANNUITY"

print(app['AMT_ANNUITY'].mean())
print(app['AMT_ANNUITY'].median())
print(app['AMT_ANNUITY'].describe())


# In[ ]:


##by checking the box plot and the description of the "AMT_ANNUITY" we can see that there are many outliers and the difference in maximum and minimum values is very high. We need to replace the null values here with the median value of the column.


# In[13]:


##checking for the value of null values in "AMT_ANNUITY"

app.AMT_ANNUITY.isnull().sum()


# In[15]:


##filling the missing values with Median of the column "AMT_ANNUITY"

fillMissingVal=app['AMT_ANNUITY'].median()
app['AMT_ANNUITY'].fillna(value = fillMissingVal, inplace =True)


# In[16]:


##checking the null values in "AMT_ANNUITY" after replacing them with median value.

app.AMT_ANNUITY.isnull().sum()


# In[17]:


##Checking the percentage of null values in the columns. 

app.isnull().sum()/len(app)*100


# In[ ]:


##Analysis of column "CNT_FAM_MEMBERS"


# In[18]:


##checking the count of family members. 

app['CNT_FAM_MEMBERS'].value_counts(dropna=False)


# In[19]:


##ploting a box plot to check the outliers in the "CNT_FAM_MEMBERS" column.

sns.boxplot(y=app['CNT_FAM_MEMBERS'])
plt.yscale('log')
plt.show()


# In[20]:


##description of the column "CNT_FAM_MEMBERS"

print(app['CNT_FAM_MEMBERS'].mean())
print(app['CNT_FAM_MEMBERS'].median())
print(app['CNT_FAM_MEMBERS'].describe())


# In[ ]:


##refering to the box plot and description it is clearb that the column has outliers and thus replacing them with the median value to clean the data further.


# In[22]:


##filling the missing values in CNT_FAM_MEMBERS with the median value.

fillMissingVal=app['CNT_FAM_MEMBERS'].median()
app['CNT_FAM_MEMBERS'].fillna(value = fillMissingVal, inplace =True)


# In[23]:


## checking the null values in the CNT_FAM_MEMBERS after the replacement.

app.CNT_FAM_MEMBERS.isnull().sum()


# In[24]:


##checking the columns with null values.

app.isnull().sum()/len(app)*100


# In[ ]:


##Analysis of "CODE_GENDER"


# In[25]:


##checking the count of gender in the CODE_GENDER

app['CODE_GENDER'].value_counts(dropna=False)


# In[ ]:


##checking the details of this column we can clearly see that the number of females is higher and only 4 values is XNA we can replace these values with F as it will not impact the analysis.


# In[26]:


##replacing XNA by F in the CODE_GENDER column. 
##checking the details of the column after the replacement. 

app.loc[app['CODE_GENDER']=='XNA','CODE_GENDER']='F'
app['CODE_GENDER'].value_counts()


# In[27]:


##The plot for CODE_GENDER.

app['CODE_GENDER'].value_counts(normalize=True).plot.bar(title='Analysis of Code Gender')
plt.show()


# In[ ]:


##Analysis of ORGANIZATION_TYPE.


# In[28]:


##checking the details of the various organizations where the customers work.

app['ORGANIZATION_TYPE'].value_counts(dropna=False)


# In[29]:


##description of the column "ORGANIZATION_TYPE"

print(app['ORGANIZATION_TYPE'].mode())
print(app['ORGANIZATION_TYPE'].describe())


# In[ ]:


##There are 55374 XNA entries which is below 18% we can discard them it will not affect the analysis.


# In[ ]:


##Analysis of AMT_GOODS_PRICE.


# In[30]:


##plotting a boxplot for "AMT_GOODS_PRICE"

plt.figure(figsize=(7,8))
sns.boxplot(y=app['AMT_GOODS_PRICE'])
plt.yscale('log') 
plt.title("Analysis of AMT_GOODS_PRICE",fontsize=16)
plt.show()


# In[31]:


##description of the column "AMT_GOODS_PRICE"

print(app['AMT_GOODS_PRICE'].describe())
print(app['AMT_GOODS_PRICE'].median())
print(app['AMT_GOODS_PRICE'].mean())
print(app['AMT_GOODS_PRICE'].max())
print(app['AMT_GOODS_PRICE'].min())


# In[ ]:


##By plotting the box plotb and looking at the description of the column we can see that there is no specific inference that we can draw so we will keep the null values. 


# In[ ]:


## Analysis of AMT_REQ_CREDIT_BUREAU_DAY


# In[32]:


##plotting a box plot for "AMT_REQ_CREDIT_BUREAU_DAY"

sns.boxplot(y=app['AMT_REQ_CREDIT_BUREAU_DAY'])
plt.title("Analysis of AMT_REQ_CREDIT_BUREAU_DAY",fontsize=14)
plt.show()


# In[33]:


##description of the AMT_REQ_CREDIT_BUREAU_DAY

print(app['AMT_REQ_CREDIT_BUREAU_DAY'].describe())


# In[ ]:


## As we can see that the column has outlier which needs to handled. They need to be removed or capped. The null values needs to be treated to for that we will fill them with median value.


# In[35]:


col_of_outliers=['AMT_REQ_CREDIT_BUREAU_DAY']
for col in col_of_outliers:
    percentiles = app[col].quantile([0.01,0.99]).values
    app[col][app[col] <= percentiles[0]] = percentiles[0]
    app[col][app[col] >= percentiles[1]] = percentiles[1]


# In[37]:


##plotting a box plot for AMT_REQ_BUREAU_DAY

sns.boxplot(y=app['AMT_REQ_CREDIT_BUREAU_DAY'])
plt.show()


# In[ ]:


##checking the datatype of the dataframe. Changing the datatype for the required columns.


# In[38]:


app.dtypes


# In[39]:


##changing the datatype to numeric values. checking the top 5 values after the change. 

numeric_cols=['TARGET','CNT_CHILDREN','AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','REGION_POPULATION_RELATIVE','DAYS_BIRTH',
                'DAYS_EMPLOYED','DAYS_REGISTRATION','DAYS_ID_PUBLISH','HOUR_APPR_PROCESS_START','LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY',
       'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY']

app[numeric_cols]=app[numeric_cols].apply(pd.to_numeric)
app.head(5)


# In[ ]:


## binning the vlaues.


# In[40]:


##creating bins for income amount.

bins = [0,25000,50000,75000,100000,125000,150000,175000,200000,225000,250000,275000,300000,325000,350000,375000,400000,425000,450000,475000,500000,10000000000]
slot = ['0-25000', '25000-50000','50000-75000','75000,100000','100000-125000', '125000-150000', '150000-175000','175000-200000',
       '200000-225000','225000-250000','250000-275000','275000-300000','300000-325000','325000-350000','350000-375000',
       '375000-400000','400000-425000','425000-450000','450000-475000','475000-500000','500000 and above']

app['AMT_INCOME_RANGE']=pd.cut(app['AMT_INCOME_TOTAL'],bins,labels=slot)


# In[41]:


##creating bins for credit amount.

bins = [0,150000,200000,250000,300000,350000,400000,450000,500000,550000,600000,650000,700000,750000,800000,850000,900000,1000000000]
slots = ['0-150000', '150000-200000','200000-250000', '250000-300000', '300000-350000', '350000-400000','400000-450000',
        '450000-500000','500000-550000','550000-600000','600000-650000','650000-700000','700000-750000','750000-800000',
        '800000-850000','850000-900000','900000 and above']

app['AMT_CREDIT_RANGE']=pd.cut(app['AMT_CREDIT'],bins=bins,labels=slots)


# In[42]:


##Dividing the dataframe in two datasets 1) Target1 :- The customers having difficulties in payments. 
##2) Target 0

target0 = app.loc[app["TARGET"]==0]
target1 = app.loc[app["TARGET"]==1]


# In[43]:


##checking the imbalance percentage.

imbalance=round(len(target0)/len(target1),2)
imbalance


# In[ ]:


##Univariate Analysis:-


# In[ ]:


##creating a reusable plotting function. 


# In[44]:


def plotfunc(df,col,title,hue = None):
    
    sns.set_style('darkgrid')
    sns.set_context('poster')
    plt.rcParams["axes.labelsize"] = 15
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['axes.titlepad'] = 30
    
    temp = pd.Series(data = hue)
    fig, ax = plt.subplots()
    width = len(df[col].unique()) + 7 + 4*len(temp.unique())
    fig.set_size_inches(width , 8)
    plt.yscale('log')
    plt.title(title,fontsize=14)
    ax = sns.countplot(data=df, x=col, hue=hue, palette='bright') 
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    plt.show()


# In[45]:


##Plotting income range for Target 0 
##The plot will have the information for the income groups of customers of different genders that falls under Target0

plotfunc(target0,col='AMT_INCOME_RANGE',title='Target 0 Income Range',hue='CODE_GENDER')


# In[ ]:


##inferences:-
##The graph clarifies that females have low rate of being the defaulters. 
##Females have more credits in this range. 


# In[46]:


##Ploting income range for Target1


plotfunc(target1,col='AMT_INCOME_RANGE',title='Target 1 Income Range',hue='CODE_GENDER')


# In[ ]:


##Inferences:-
##this graph clearly indicates that Males have higher tendency to be the defaulters. 


# In[47]:


##ploting income type for Target0.

plotfunc(target0,col='NAME_INCOME_TYPE',title='Target 0 Income Type',hue='CODE_GENDER')


# In[ ]:


##Inferences:-
##1. Female have more credit than males
##2. Income type working , commercial associate , pensioner and state servant have high credits.
##3. Income type student ,unemployed, businessman and maternity leave have low credits.


# In[48]:


##plotting income type for Target1

plotfunc(target1,col='NAME_INCOME_TYPE',title='Target 1 Income Type',hue='CODE_GENDER')


# In[ ]:


##Inferences:- By looking at the graph we can infer that
##1. income type working , commercial associate , pensioner and state servant is high. Following same trend as target0
##2. income type unemployed and maternity leave have low credits similar to Traget0


# In[49]:


### Plotting for CNT_CHILDREN for target0 and target1


fig, ax =plt.subplots(1,2,figsize=(12,6))
sns.countplot(target0['CNT_CHILDREN'], ax=ax[0]).set_title('Target 0( Not A Defaulter)')
sns.countplot(target1['CNT_CHILDREN'], ax=ax[1]).set_title('Target 1 (Defaulter)')
fig.show()


# In[ ]:


##Inferences:- There is no specific patter found between the child count and the prospect of being a defaulter.


# In[50]:


# Plotting for NAME_EDUCATION_TYPE for target0 and target1


fig, ax=plt.subplots(1,2,figsize=(50,13))
sns.countplot(target0['NAME_EDUCATION_TYPE'], ax=ax[0]).set_title('Target 0( Not A Defaulter)')
sns.countplot(target1['NAME_EDUCATION_TYPE'], ax=ax[1]).set_title('Target 1 (Defaulter)')
fig.show()


# In[ ]:


##By looking at the graphs we can infer that the people with secondary education have defaulted the most in both the Target groups.


# In[51]:


###Plotting for NAME_CONTRACT_TYPE for target0


plotfunc(target0,col='NAME_CONTRACT_TYPE',title='Target 0 Of Contract Type',hue='CODE_GENDER')


# In[ ]:


##inferences:- 
##1) cash loans have higher number than revolving loans.
##2)Females have higher credits.


# In[52]:


###Plotting for NAME_CONTRACT_TYPE for target1


plotfunc(target1,col='NAME_CONTRACT_TYPE',title='Target 1 Contract Type',hue='CODE_GENDER')


# In[ ]:


##Inferences:- 
##1) Cash loans are hifgher in numbers than revolving loans
##2) only females have opted for the revolving loans.


# In[53]:


##Function for Box plot:-

def cusBoxPlot(data,col,title):
    sns.set_style('darkgrid')
    sns.set_context('poster')
    plt.rcParams["axes.labelsize"] = 17
    plt.rcParams['axes.titlesize'] = 19
    plt.rcParams['axes.titlepad'] = 34
    
    plt.title(title)
    plt.yscale('log')
    sns.boxplot(data =data, x=col,orient='v',color="Green")
    plt.show()


# In[54]:


###Distribution of income amount for Target0


cusBoxPlot(data=target0,col='AMT_INCOME_TOTAL',title='Target 0 Income Amount')


# In[ ]:





# In[55]:


# Distribution of income amount for Target1


cusBoxPlot(data=target1,col='AMT_INCOME_TOTAL',title='Target 1 Income Amount')


# In[ ]:


##Inferences:-

##1. Outliners are present in both
##2. 3rd quartile is narrow for both target 1 and target 0
##3. Most of the clients have income in the 1st quartile


# In[56]:


# Disrtibution of credit amount for Target 0

cusBoxPlot(data=target0,col='AMT_CREDIT',title='Target 0 Credit Amount')


# In[57]:


# Disrtibution of credit amount for Target 1


cusBoxPlot(data=target1,col='AMT_CREDIT',title='Target 1 Credit Amount')


# In[ ]:


##Inference:

##1. Outliners are present in both
##2. 3rd quartile is narrow for both target 1 and target 0
##3. Most of the clients have credit amount in the 1st quartile


# In[ ]:


##Bivariate Analysis:-


# In[58]:


#Plotting Correlation matrix for Target 0 application data


d=target0[['SK_ID_CURR','CNT_CHILDREN','AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY',
                               'AMT_GOODS_PRICE','DAYS_BIRTH','DAYS_EMPLOYED','CNT_FAM_MEMBERS','REGION_RATING_CLIENT',
                              'REGION_POPULATION_RELATIVE','DAYS_ID_PUBLISH']]

plt.figure(figsize=(30,30))
sns.heatmap(d.corr(), fmt='.1f', cmap="RdYlGn", annot=True)
plt.title("Correlation Matrix for Non-Defaulters",fontsize=30, pad=20 )
plt.show()


# In[ ]:


##Higher correlation values for Target0


# In[59]:


#Plotting Correlation matrix for Target 1 application data


d=target1[['SK_ID_CURR','CNT_CHILDREN','AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY',
                               'AMT_GOODS_PRICE','DAYS_BIRTH','DAYS_EMPLOYED','CNT_FAM_MEMBERS','REGION_RATING_CLIENT',
                              'REGION_POPULATION_RELATIVE','DAYS_ID_PUBLISH']]
f, ax = plt.subplots(figsize=(30, 30))
sns.heatmap(d.corr(), annot=True, fmt='.1f',cmap="RdYlGn", linewidths=.5, ax=ax)
plt.title("Correlation matrix for Clients with payment difficulties",fontsize=30, pad=20 )
plt.show()


# In[ ]:


##These columns have higher correlation to both Target 0 and Target1.


# In[61]:


#ploting income vs credit for Target 0

sns.jointplot('AMT_INCOME_TOTAL', 'AMT_CREDIT', target0)
plt.show()


# In[ ]:





# In[62]:


#ploting income vs credit for Target 1

sns.jointplot('AMT_INCOME_TOTAL', 'AMT_CREDIT', target1)
plt.show()


# In[63]:


#ploting AMT_INCOME_TOTAL vs CNT_CHILDREN for Target 0


sns.jointplot('CNT_CHILDREN', 'AMT_INCOME_TOTAL', target0)
plt.show()


# In[64]:


#ploting AMT_INCOME_TOTAL vs CNT_CHILDREN for Target 1


sns.jointplot('CNT_CHILDREN', 'AMT_INCOME_TOTAL', target1)
plt.show()


# In[ ]:


## Analysis of Credit amount with respect to Education status



# In[65]:


#ploting NAME_EDUCATION_TYPE vs AMT_CREDIT for each family status for Target 0


sns.catplot(data =target0, x='NAME_EDUCATION_TYPE',y='AMT_CREDIT', hue ='NAME_FAMILY_STATUS',height=6,aspect=4, kind="bar", palette="muted")
plt.title('Credit Amount vs Education Status For Traget 0')


# In[ ]:


###Inference:

##1. Customers holding academic degree have greater credit amount, Civil marriage segment being the highest among them.
##2. Lower educated customers tends to have lower credit amount, Widows being the lowest among them
##3. Married customers in almost all education segment except lower secondary and academic degrees have a higher credit amount.


# In[66]:


###ploting NAME_EDUCATION_TYPE vs AMT_CREDIT for each family status for target 1


sns.catplot(data =target1, x='NAME_EDUCATION_TYPE',y='AMT_CREDIT', hue ='NAME_FAMILY_STATUS',height=6,aspect=4, kind="bar", palette="muted")
plt.title('Credit Amount vs Education Status for Traget 1')


# In[ ]:


##Inference:-

##1. Married Academic degree holding customers generally have a higher credit amount and so their defaulting rate is also high
##2. Accross all education segment married customer tends to have higher credit amount
##3. Customers holding lower eductation tends to have a lower credit amount
##4. Single and Married are the only 2 family types present in academic degree .


# In[ ]:


## Analysis of Income amount with respect to Education Status



# In[67]:


# Box plotting for Income amount vs Education Status for Target 0 in logarithmic scale


import textwrap
plt.figure(figsize=(10,10))
plt.xticks(rotation=45,ha="right",rotation_mode="anchor",fontsize=12)
plt.yscale('log')
g1=sns.boxplot(data =target0, x='NAME_EDUCATION_TYPE',y='AMT_INCOME_TOTAL', hue ='NAME_FAMILY_STATUS',orient='v')
plt.title('Income Amount vs Education Status For Non-Defaulters',fontsize=16)
plt.show()


# In[ ]:


##Inference:
##1.For Education type 'Higher education' the income amount mean is mostly equal with family status. It does contain many outliers.
##2. Less outlier are having for Academic degree but they are having the income amount is little higher that Higher education.
##3. Lower secondary of civil marriage family status are have less income amount than others.


# In[68]:


# Box plotting for Income amount vs Education Status for Target 1 in logarithmic scale


plt.figure(figsize=(10,10))
plt.xticks(rotation=45,ha="right",rotation_mode="anchor",fontsize=12)
plt.yscale('log')
sns.boxplot(data =target1, x='NAME_EDUCATION_TYPE',y='AMT_INCOME_TOTAL', hue ='NAME_FAMILY_STATUS',orient='v')
plt.title('Income Amount vs Education Status For Defaulters',fontsize=16)
plt.show()


# In[ ]:


##Inference:
##1. Have some similarity with Target0, From above boxplot for Education type 'Higher education' the income amount is mostly equal with family status.
##2. No outlier for Academic degree but there income amount is little higher than that Higher education.
##3. Lower secondary are having less income amount than others.


# In[ ]:


Segment 2:- Previous Application.


# In[69]:


#Read the dataset of "previous_application" in pr dataframe


pr = pd.read_csv(r'C:\Users\sneha\Downloads\previous_application (1).csv')


# In[70]:


###Display the first 5 records

pr.head()


# In[71]:


##checking shape of the Previous application
pr.shape


# In[72]:


##checking the info of the previous application. 

pr.info()


# In[73]:


# Cleaning the missing data
# listing the null values columns having more than 50%

emptycol1=pr.isnull().sum()
emptycol1=emptycol1[emptycol1.values>(0.5*len(emptycol1))]
len(emptycol1)


# In[74]:


#Removing those 15 columns


emptycol1 = list(emptycol1[emptycol1.values>=0.5].index)
pr.drop(labels=emptycol1,axis=1,inplace=True)
pr.shape


# In[75]:


#Removing the column values of 'XNA' and 'XAP'


pr=pr.drop(pr[pr['NAME_CASH_LOAN_PURPOSE']=='XNA'].index)
pr=pr.drop(pr[pr['NAME_CASH_LOAN_PURPOSE']=='XAP'].index)
pr.shape


# In[77]:


#Merging the Application dataset with previous appliaction dataset


Merged_data=pd.merge(left=app,right=pr,how='inner',on='SK_ID_CURR',suffixes='_x')


# In[78]:


Merged_data.head()


# In[79]:


# Renaming the column names after merging


Merged_data = Merged_data.rename({'NAME_CONTRACT_TYPE_' : 'NAME_CONTRACT_TYPE','AMT_CREDIT_':'AMT_CREDIT','AMT_ANNUITY_':'AMT_ANNUITY',
                         'WEEKDAY_APPR_PROCESS_START_' : 'WEEKDAY_APPR_PROCESS_START',
                         'HOUR_APPR_PROCESS_START_':'HOUR_APPR_PROCESS_START','NAME_CONTRACT_TYPEx':'NAME_CONTRACT_TYPE_PREV',
                         'AMT_CREDITx':'AMT_CREDIT_PREV','AMT_ANNUITYx':'AMT_ANNUITY_PREV',
                         'WEEKDAY_APPR_PROCESS_STARTx':'WEEKDAY_APPR_PROCESS_START_PREV',
                         'HOUR_APPR_PROCESS_STARTx':'HOUR_APPR_PROCESS_START_PREV'}, axis=1)


# In[80]:


# Removing unwanted columns for analysis


Merged_data.drop(['SK_ID_CURR','WEEKDAY_APPR_PROCESS_START', 'HOUR_APPR_PROCESS_START','REG_REGION_NOT_LIVE_REGION', 
              'REG_REGION_NOT_WORK_REGION','LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY',
              'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY','WEEKDAY_APPR_PROCESS_START_PREV',
              'HOUR_APPR_PROCESS_START_PREV', 'FLAG_LAST_APPL_PER_CONTRACT','NFLAG_LAST_APPL_IN_DAY'],axis=1,inplace=True)


# In[81]:


#check shape of the Merged Data


Merged_data.shape


# In[82]:


##Univariate Analysis
# Distribution of contract status in logarithmic scale


sns.set_style('whitegrid')
sns.set_context('talk')

plt.figure(figsize=(10,10))
plt.rcParams["axes.labelsize"] = 20
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['axes.titlepad'] = 30
plt.title('Distribution Of Contract Status With Purposes')
plt.xscale('log')

ax = sns.countplot(data = Merged_data, y= 'NAME_CASH_LOAN_PURPOSE',orient="h",
                   order=Merged_data['NAME_CASH_LOAN_PURPOSE'].value_counts().index,hue= 'NAME_CONTRACT_STATUS',palette='Set2') 
ax.xaxis.tick_top()


# In[ ]:


##Inference:
##1.Most rejection of loans came from purpose 'Repairs'.
##2.We have almost equal number of approves and rejection for Medicine,Every day expenses and education purposes.


# In[83]:


#Distribution of purposes with target


sns.set_style('whitegrid')
sns.set_context('talk')

plt.figure(figsize=(10,10))
plt.rcParams["axes.labelsize"] = 20
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['axes.titlepad'] = 30
plt.xscale('log')
plt.title('Distribution of purposes with target ')
ax = sns.countplot(data = Merged_data, y= 'NAME_CASH_LOAN_PURPOSE',orient="h",
                   order=Merged_data['NAME_CASH_LOAN_PURPOSE'].value_counts().index,hue= 'TARGET',palette='Set2') 
ax.xaxis.tick_top()


# In[ ]:


##Inference: we can conclude from above plot that Loan purposes with 'Repairs' are facing more difficulites in payment on time.



# In[84]:


##Bivariate Analysis.
### Box plotting for Credit amount prev vs Housing type in logarithmic scale


sns.catplot(x="NAME_HOUSING_TYPE", y="AMT_CREDIT_PREV", hue="TARGET", data=Merged_data, kind="violin",height=6,aspect=4,palette='husl')
plt.title('Prev Credit amount vs Housing type')
plt.show()


# In[ ]:


##inference:- We can conclude that bank should avoid giving loans to the housing type of co-op apartment as they are having difficulties in payment.



# In[ ]:




