
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)


# In[2]:


pathTrain ='./train.csv'
dfTrain = pd.read_csv(pathTrain,index_col=None)

pathTest ='./test.csv'
dfTest = pd.read_csv(pathTest,index_col=None)
idTrain = dfTrain['Id']
idTest = dfTest['Id']
dfTrain.drop("Id", axis = 1, inplace = True) 
dfTest.drop("Id", axis = 1, inplace = True) 

dfTrain_X= dfTrain
dfAll = pd.concat((dfTrain_X , dfTest)).reset_index(drop=True)

qualitativecol  = [ col for col in dfTrain.columns if  dfTrain[col].dtype == object]
quantitativecol = [ col for col in dfTrain.columns if  dfTrain[col].dtype != object]
quantitativecol.remove('SalePrice')

# dfAll.head()


# In[3]:


print('qualitativecol col : {}'.format(len(qualitativecol)))
print('quantitativecol col : {}'.format(len(quantitativecol)))


# In[6]:


s='MSSubClass: The building class,MSZoning: The general zoning classification,LotFrontage: Linear feet of street connected to property,LotArea: Lot size in square feet,Street: Type of road access,Alley: Type of alley access,LotShape: General shape of property,LandContour: Flatness of the property,Utilities: Type of utilities available,LotConfig: Lot configuration,LandSlope: Slope of property,Neighborhood: Physical locations within Ames city limits,Condition1: Proximity to main road or railroad,Condition2: Proximity to main road or railroad (if a second is present),BldgType: Type of dwelling,HouseStyle: Style of dwelling,OverallQual: Overall material and finish quality,OverallCond: Overall condition rating,YearBuilt: Original construction date,YearRemodAdd: Remodel date,RoofStyle: Type of roof,RoofMatl: Roof material,Exterior1st: Exterior covering on house,Exterior2nd: Exterior covering on house (if more than one material),MasVnrType: Masonry veneer type,MasVnrArea: Masonry veneer area in square feet,ExterQual: Exterior material quality,ExterCond: Present condition of the material on the exterior,Foundation: Type of foundation,BsmtQual: Height of the basement,BsmtCond: General condition of the basement,BsmtExposure: Walkout or garden level basement walls,BsmtFinType1: Quality of basement finished area,BsmtFinSF1: Type 1 finished square feet,BsmtFinType2: Quality of second finished area (if present),BsmtFinSF2: Type 2 finished square feet,BsmtUnfSF: Unfinished square feet of basement area,TotalBsmtSF: Total square feet of basement area,Heating: Type of heating,HeatingQC: Heating quality and condition,CentralAir: Central air conditioning,Electrical: Electrical system,1stFlrSF: First Floor square feet,2ndFlrSF: Second floor square feet,LowQualFinSF: Low quality finished square feet (all floors),GrLivArea: Above grade (ground) living area square feet,BsmtFullBath: Basement full bathrooms,BsmtHalfBath: Basement half bathrooms,FullBath: Full bathrooms above grade,HalfBath: Half baths above grade,BedroomAbvGr: Number of bedrooms above basement level,KitchenAbvGr: Number of kitchens,KitchenQual: Kitchen quality,TotRmsAbvGrd: Total rooms above grade (does not include bathrooms),Functional: Home functionality rating,Fireplaces: Number of fireplaces,FireplaceQu: Fireplace quality,GarageType: Garage location,GarageYrBlt: Year garage was built,GarageFinish: Interior finish of the garage,GarageCars: Size of garage in car capacity,GarageArea: Size of garage in square feet,GarageQual: Garage quality,GarageCond: Garage condition,PavedDrive: Paved driveway,WoodDeckSF: Wood deck area in square feet,OpenPorchSF: Open porch area in square feet,EnclosedPorch: Enclosed porch area in square feet,3SsnPorch: Three season porch area in square feet,ScreenPorch: Screen porch area in square feet,PoolArea: Pool area in square feet,PoolQC: Pool quality,Fence: Fence quality,MiscFeature: Miscellaneous feature not covered in other categories,MiscVal: Value of miscellaneous feature,MoSold: Month Sold,YrSold: Year Sold,SaleType: Type of sale,SaleCondition: Condition of sale'
data_desc = {ss.split(':')[0]: ss.split(':')[1] for ss in  s.split(',') }
# del data_desc['SalePrice']
for k,v in data_desc.items():
    print('{:>}\t{}\t{}\t{}'.format(k,dfAll[k].isna().sum(),dfTrain[k].dtype,v))


# In[7]:


print(len(dfTrain.columns))
dfTrain.info()


# In[8]:


pd.set_option('display.max_columns', None)
dfTrain.head()
dfTrain.describe()


# In[27]:


### 檢查columns na ratio
dfna = pd.DataFrame(data={'columnName' : dfAll.columns[dfAll.isna().sum()!=0],
                          'count' : dfAll.isna().sum()[dfAll.isna().sum()!=0]/len(dfAll)*100})
dfna.plot(kind = 'bar')



# In[28]:


### 類別feature NA 都填"No"看箱形圖  
plt.figure(figsize=(20,200))

colname_obj = qualitativecol
colname_obj.append('OverallQual') 

for ind , col in enumerate(colname_obj):
    plt.subplot( ((len(colname_obj)//2) +1 ), 2 , ind+1)
    plt.xticks(rotation=90);
    data = pd.concat([dfTrain['SalePrice'], dfTrain[col]], axis=1)
    
    if (col != 'SalePrice'):
        data = pd.concat([dfTrain['SalePrice'], dfTrain[col]], axis=1).fillna('No')
        fig = sns.boxplot(x=col, y="SalePrice", data=data)
        plt.tight_layout()
        
plt.show()


# In[11]:


col = ['OverallQual','']

var = 'OverallQual'
data = pd.concat([dfTrain['SalePrice'], dfTrain[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);

print(dfTrain['OverallQual'].unique())
print(data_desc['OverallQual'])


# In[ ]:


#把幾個品質等級相關的 以各等級的售價平均 encode


# In[29]:


### number feature 看分布 
plt.figure(figsize=(20,160),dpi=150)
f = pd.melt(dfTrain, value_vars=quantitativecol)#[ col for col in dfTrain.columns if  dfTrain[col].dtype != object])
g = sns.FacetGrid(f, col="variable",  col_wrap=3, sharex=False, sharey=False)
g = g.map(sns.distplot, "value")
# g = g.map(sns.boxplot, "value")


# In[ ]:


### number feature 看scatter  
# plt.figure(figsize=(20,160))
# # plt.tight_layout()
# # plt.xticks(rotation=90);

# sns.pairplot(dfTrain[quantitativecol].fillna(0))
# plt.show()


# In[30]:


data_desc['1stFlrSF']


# In[31]:


#Correlation map to see how features are correlated with SalePrice
corrmat = dfTrain.corr()

plt.figure(figsize=(12,9))
sns.heatmap(corrmat [-1:][corrmat [-1:]>0.5], vmax=1, square=True)
plt.show()

# print(corrmat [-1:][corrmat [-1:]>0.5].isna().sum())



col_corr_05=['OverallQual','YearBuilt','YearRemodAdd','TotalBsmtSF','1stFlrSF','GrLivArea','FullBath','TotRmsAbvGrd','GarageYrBlt','GarageCars','GarageArea']

plt.subplots(figsize=(20,20))

for ind,col in enumerate(col_corr_05):
    
    plt.subplot( ((len(col_corr_05)//3) +1 ), 3 , ind+1)

    if (col != 'SalePrice'):
        data = pd.concat([dfTrain['SalePrice'], dfTrain[col]], axis=1).fillna('No')
        plt.scatter(x=dfTrain[col] , y=dfTrain['SalePrice'])
        plt.xlabel(col)
        plt.legend(data_desc[col])
            
plt.show()



# In[33]:


fig, ax = plt.subplots()

ax.scatter(x = dfTrain['GrLivArea'], y = dfTrain['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)

#highlight the outlier
ax.scatter(x = dfTrain['GrLivArea'][(dfTrain['GrLivArea']>4000) & (dfTrain['SalePrice']<300000) ],
           y = dfTrain['SalePrice'][(dfTrain['GrLivArea']>4000) & (dfTrain['SalePrice']<300000) ],
           c = 'g') ##outlier
plt.show()


# In[34]:


#把一些評估離群的點刪除  TotalBsmtSF 1stFlrSF GrLivArea
# plt.subplots(figsize=(12,9))
plt.figure(figsize=(15,5))

plt.subplot(131)
plt.scatter(x = dfTrain['TotalBsmtSF'], y = dfTrain['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('TotalBsmtSF', fontsize=13)
#highlight the outlier
plt.scatter(x = dfTrain['TotalBsmtSF'][(dfTrain['TotalBsmtSF']>4000)  ],
            y = dfTrain['SalePrice'][(dfTrain['TotalBsmtSF']>4000) ],
            c = 'g') ##outlier

plt.subplot(132)
plt.scatter(x = dfTrain['1stFlrSF'], y = dfTrain['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('1stFlrSF', fontsize=13)
# highlight the outlier
plt.scatter(x = dfTrain['1stFlrSF'][(dfTrain['1stFlrSF']>4000) ],
            y = dfTrain['SalePrice'][(dfTrain['1stFlrSF']>4000)  ],
            c = 'g') ##outlier

plt.subplot(133)
plt.scatter(x = dfTrain['GrLivArea'], y = dfTrain['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
#highlight the outlier
plt.scatter(x = dfTrain['GrLivArea'][(dfTrain['GrLivArea']>4000) & ((dfTrain['SalePrice']<300000))],
            y = dfTrain['SalePrice'][(dfTrain['GrLivArea']>4000) & ((dfTrain['SalePrice']<300000)) ],
            c = 'g') ##outlier

plt.show()

# 把綠點都丟掉
dfTrain = dfTrain.drop(dfTrain[(dfTrain['GrLivArea']>4000) & (dfTrain['SalePrice']<300000)].index)

print('drop the outliers')

plt.figure(figsize=(15,5))

plt.subplot(131)
plt.scatter(x = dfTrain['TotalBsmtSF'], y = dfTrain['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('TotalBsmtSF', fontsize=13)
#highlight the outlier
plt.scatter(x = dfTrain['TotalBsmtSF'][(dfTrain['TotalBsmtSF']>4000)  ],
            y = dfTrain['SalePrice'][(dfTrain['TotalBsmtSF']>4000) ],
            c = 'g') ##outlier

plt.subplot(132)
plt.scatter(x = dfTrain['1stFlrSF'], y = dfTrain['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('1stFlrSF', fontsize=13)
# highlight the outlier
plt.scatter(x = dfTrain['1stFlrSF'][(dfTrain['1stFlrSF']>4000) ],
            y = dfTrain['SalePrice'][(dfTrain['1stFlrSF']>4000)  ],
            c = 'g') ##outlier

plt.subplot(133)
plt.scatter(x = dfTrain['GrLivArea'], y = dfTrain['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
#highlight the outlier
plt.scatter(x = dfTrain['GrLivArea'][(dfTrain['GrLivArea']>4000) & ((dfTrain['SalePrice']<300000))],
            y = dfTrain['SalePrice'][(dfTrain['GrLivArea']>4000) & ((dfTrain['SalePrice']<300000)) ],
            c = 'g') ##outlier

plt.show()


# In[ ]:


##看分布


# In[35]:


from scipy import stats
from scipy.stats import norm, skew ,johnsonsu #for some statistics


# In[37]:



plt.figure(figsize=(20,10))
fig ,[ax1,ax2,ax3]  =plt.subplots(nrows=1,ncols=3)

# plt.figure(1); plt.title('Johnson SU')
ax1.set_title('Johnson SU')
sns.distplot(dfTrain['SalePrice'], kde=True, fit=stats.johnsonsu ,ax = ax1)
# ax1.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],            loc='best')
ax1.set_title('Normal')
sns.distplot(dfTrain['SalePrice'], kde=True, fit=stats.norm ,ax = ax2)
ax1.set_title('Log Normal')
sns.distplot(dfTrain['SalePrice'], kde=False, fit=stats.lognorm ,ax = ax3)


# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(dfTrain['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
# plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
#             loc='best')
# plt.ylabel('Frequency')
# plt.title('SalePrice distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(dfTrain['SalePrice'], plot=plt)
plt.show()
# res


# In[38]:


# dfTrain['SalePrice'] 右偏嚴重，取對數進行分布改造
dfTrain['SalePrice_log'] = np.log1p(dfTrain['SalePrice'])

plt.figure(figsize=(20,10))
fig ,[ax1,ax2,ax3]  =plt.subplots(nrows=1,ncols=3)

# plt.figure(1); plt.title('Johnson SU')
ax1.set_title('Johnson SU')
sns.distplot(dfTrain['SalePrice_log'], kde=True, fit=stats.johnsonsu ,ax = ax1)
# ax1.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],            loc='best')
ax1.set_title('Normal')
sns.distplot(dfTrain['SalePrice_log'], kde=True, fit=stats.norm ,ax = ax2)
ax1.set_title('Log Normal')
sns.distplot(dfTrain['SalePrice_log'], kde=False, fit=stats.lognorm ,ax = ax3)


# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(dfTrain['SalePrice_log'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
# plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
#             loc='best')
# plt.ylabel('Frequency')
# plt.title('SalePrice distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(dfTrain['SalePrice_log'], plot=plt)
plt.show()


# Features engineering
# let's first concatenate the train and test data in the same dataframe

# In[39]:


col = ['GrLivArea', '1stFlrSF' ]#,'TotalBsmtSF']
col_new = [x + '_log' for x in col]

for i,j in zip(col,col_new) :
    dfTrain[j]= np.log(dfTrain[i])
    

plt.figure(figsize=(10,5))

a=[]
b=[]
c=[]
ax=[a,b,c]
fig , ax=plt.subplots(nrows=3,ncols=2)
fig.set_figwidth(20)
fig.set_figheight(20)
for i in range(0,2):
    for j  in range(0,2):
        ax[i][j].set_title(col[i])
        if j ==0:
            sns.distplot(dfTrain[col[i]], kde=True, fit=stats.norm ,ax = ax[i][j])
        else:
            sns.distplot(dfTrain[col_new[i]], kde=True, fit=stats.norm ,ax = ax[i][j])
            
            
## special for TotalBsmtSF
dfTrain['hasBsmt'] = 0
dfTrain['hasBsmt'][dfTrain['TotalBsmtSF'] > 0] =1 
dfTrain['TotalBsmtSF_log']= np.log(dfTrain['TotalBsmtSF'][dfTrain['hasBsmt']==1])

ax[2][0].set_title('TotalBsmtSF')
sns.distplot(dfTrain['TotalBsmtSF'], kde=True, fit=stats.norm ,ax = ax[2][0])
ax[2][1].set_title('TotalBsmtSF_log')
sns.distplot(dfTrain['TotalBsmtSF_log'][dfTrain['hasBsmt']==1], kde=True, fit=stats.norm ,ax = ax[2][1])

dfTrain['TotalBsmtSF_log']=dfTrain['TotalBsmtSF_log'].fillna(0)


# In[41]:


###same for test data
col = ['GrLivArea', '1stFlrSF' ]#,'TotalBsmtSF']
col_new = [x + '_log' for x in col]

for i,j in zip(col,col_new) :
    dfTest[j]= np.log(dfTest[i])
    

plt.figure(figsize=(10,5))

a=[]
b=[]
c=[]
ax=[a,b,c]
fig , ax=plt.subplots(nrows=3,ncols=2)
fig.set_figwidth(20)
fig.set_figheight(20)
for i in range(0,2):
    for j  in range(0,2):
        ax[i][j].set_title(col[i])
        if j ==0:
            sns.distplot(dfTest[col[i]], kde=True, fit=stats.norm ,ax = ax[i][j])
        else:
            sns.distplot(dfTest[col_new[i]], kde=True, fit=stats.norm ,ax = ax[i][j])
            
            
## special for TotalBsmtSF
dfTest['hasBsmt'] = 0
dfTest['hasBsmt'][dfTest['TotalBsmtSF'] > 0] =1 
dfTest['TotalBsmtSF_log']= np.log(dfTest['TotalBsmtSF'][dfTest['hasBsmt']==1])
dfTest['TotalBsmtSF_log']=dfTest['TotalBsmtSF_log'].fillna(0)
dfTest['TotalBsmtSF']=dfTest['TotalBsmtSF'].fillna(0)
ax[2][0].set_title('TotalBsmtSF')
sns.distplot(dfTest['TotalBsmtSF'], kde=True, fit=stats.norm ,ax = ax[2][0])
ax[2][1].set_title('TotalBsmtSF_log')
sns.distplot(dfTest['TotalBsmtSF_log'][dfTest['hasBsmt']==1], kde=True, fit=stats.norm ,ax = ax[2][1])
dfTest['TotalBsmtSF_log']=dfTest['TotalBsmtSF_log'].fillna(0)


# In[42]:


ntrain = dfTrain.shape[0]
ntest = dfTrain.shape[0]
y_train = dfTrain.SalePrice_log.values
all_data = pd.concat((dfTrain, dfTest)).reset_index(drop=True)
all_data.drop(['SalePrice','SalePrice_log'], axis=1, inplace=True)
all_data.drop(['GrLivArea','1stFlrSF','TotalBsmtSF'], axis=1, inplace=True)
print("all_data size is : {}".format(all_data.shape))


# In[43]:


all_data.columns


# In[44]:


[col for col in all_data.columns if all_data[col].isna().sum() !=0 ]


# In[45]:


#Correlation map to see how features are correlated with SalePrice
corrmat = dfTrain.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True)


# In[46]:


# 要注意 test 的na col 是否與 train 的 na col 重複
print("dfTrain na count : {}".format(len(dfTrain.columns[dfTrain.isna().sum()!=0])))
print("dfTest  na count : {}".format(len(  dfTest.columns[dfTest.isna().sum()!=0])))
print("all_data na count : {}".format(len(all_data.columns[all_data.isna().sum()!=0])))


# In[47]:


all_data.isna().sum()[all_data.isnull().sum()!=0].sort_values(ascending=False)


# In[48]:


all_data_na  = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na  = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)#[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data['Missing Count'] = all_data.isna().sum()[all_data.isnull().sum()!=0].sort_values(ascending=False)
missing_data.head()


# In[49]:


plt.xticks(rotation='90')
sns.barplot(x=all_data_na.index, y=all_data_na)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)


# In[50]:


Na_col = all_data.columns[all_data.isna().sum()!=0]  #看一下哪些col 有NA

print(len(Na_col))

for k,v in data_desc.items():
    if k in Na_col:
        print(k,'\t',len(all_data[k].unique()),'\t',v)
        
dfTrain[Na_col].head()


# In[ ]:


####針對各類別 col NA 決定補植方式


# In[ ]:





# In[51]:


#### 類別補植方式
col_na_qualitativecol = [ col for col in Na_col if all_data[col].dtype == object]
col_na_qualitativecol
for col in col_na_qualitativecol:
    print('{}\t{}\t{}\t{}'.format(col, dfTrain[col].mode()[0] , dfTest[col].mode()[0] , data_desc[col] ))
    
col_fill_no=['Alley','BsmtQual','BsmtCond','BsmtExposure',
             'BsmtFinType1','BsmtFinType2',
             'GarageType','GarageFinish','GarageQual','GarageCond',
             'PoolQC','Fence','MiscFeature']
col_fill_mode=['MSZoning','Exterior1st','Exterior2nd','Electrical','KitchenQual',
               'Functional','FireplaceQu','SaleType']
col_fill_None = ['MasVnrType']
col_drop=['Utilities']

for col in col_fill_no:
    all_data[col]=all_data[col].fillna('no')
for col in col_fill_None:
    all_data[col]=all_data[col].fillna('None')
    
for col in col_fill_mode:
    all_data[col]=all_data[col].fillna(all_data[col].mode()[0])
    

all_data = all_data.drop(col_drop, axis=1)
Na_col=Na_col.drop('Utilities')
col_na_qualitativecol.remove('Utilities')

print(len(col_na_qualitativecol))
print(len(col_fill_no)+len(col_fill_mode)+len(col_drop))


# In[52]:


# col = ['OverallQual','']

var = 'Exterior1st'
print(dfTrain[var].isna().sum())
print(dfTest[var].isna().sum())

fig, [ax1,ax2] = plt.subplots(figsize=(12, 6),nrows=1,ncols=2)
plt.xticks(rotation=90)
data = pd.concat([dfTrain['SalePrice'], dfTrain[var].fillna('no')], axis=1)
fig = sns.boxplot(x=var, y="SalePrice", data=data,ax=ax1)
data = pd.concat([dfTrain['SalePrice_log'], dfTrain[var].fillna('no')], axis=1)
fig = sns.boxplot(x=var, y="SalePrice_log", data=data,ax=ax2)

# fig.axis(ymin=0, ymax=800000);

# dfTrain['OverallQual'].unique()


# In[53]:


#### 數值補植方式
col_na_quantitativecol = [ col for col in Na_col if all_data[col].dtype != object]
col_na_quantitativecol
for col in col_na_quantitativecol:
    print('{}\t{}\t{}\t{}'.format(col, dfTrain[col].mean() , dfTest[col].mean() , data_desc[col] ))
    
col_fill_0=['MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','BsmtFullBath','BsmtHalfBath','GarageYrBlt','GarageCars','GarageArea']
col_fill_mean=['LotFrontage']
# col_drop=['Utilities']

for col in col_fill_0:
    all_data[col]=all_data[col].fillna(0)

for col in col_fill_mean:
    all_data[col]=all_data[col].fillna(all_data[col].mean())
    
    
print(len(col_na_quantitativecol))
print(len(col_fill_0)+len(col_fill_mean))


# In[54]:


#Check remaining missing values if any 
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head()


# In[ ]:





# In[55]:


dic_dummy = {}

for col in qualitativecol:
    data = pd.concat([dfTrain['SalePrice'], dfTrain[col]], axis=1)
    dic_tmp =data.groupby(col).mean().to_dict()
    dic_tmp[col] = dic_tmp.pop('SalePrice')
    dic_dummy.update(dic_tmp)


# In[56]:


dic_dummy


# In[57]:


all_data.replace(dic_dummy, inplace=True)
all_data.head()


# In[58]:


#MSSubClass=The building class
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)


#Changing OverallCond into a categorical variable
all_data['OverallCond'] = all_data['OverallCond'].astype(str)


#Year and month sold are transformed into categorical features.
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)


# In[59]:


from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(all_data[c].values)) 
    all_data[c] = lbl.transform(list(all_data[c].values))

# shape        
print('Shape all_data: {}'.format(all_data.shape))


# In[60]:


all_data['GarageType'].unique()


# In[61]:


# Adding total sqfootage feature 
all_data['TotalSF'] = all_data['TotalBsmtSF_log'] + all_data['1stFlrSF_log'] + all_data['2ndFlrSF']


# In[62]:


numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

# Check the skew of all numerical features
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(10)


# In[63]:


skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    #all_data[feat] += 1
    all_data[feat] = boxcox1p(all_data[feat], lam)
    
#all_data[skewed_features] = np.log1p(all_data[skewed_features])


# In[64]:


all_data = pd.get_dummies(all_data)
print(all_data.shape)


# In[65]:


all_data.columns


# In[ ]:


# all_data = pd.get_dummies(all_data)
# print(all_data.shape)


# In[66]:


train = all_data[:ntrain]
test = all_data[ntrain:]


# In[67]:


train.info()


# In[69]:


# !pip install lightgbm --user
# !pip install xgboost 


# In[76]:


from sklearn.linear_model import ElasticNet, Lasso,   BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
# import lightgbm as lgb


# In[71]:


#Validation function
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)


# In[72]:


def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))
    print(rmsle(y_train, xgb_train_pred))


# In[ ]:


svr_poly = SVR(kernel='poly')
score = rmsle_cv(svr_poly)
print("\n svr_poly score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[ ]:


lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
score = rmsle_cv(lasso)
print("\n lasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[ ]:


lasso = make_pipeline(RobustScaler(),Lasso(alpha=0.00053, max_iter=50000, random_state=1) )
score = rmsle_cv(lasso)
print("\n lasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[ ]:


ridge = make_pipeline(RobustScaler(), KernelRidge(alpha=0.55, coef0=3.7, degree=2, kernel='polynomial') )
score = rmsle_cv(ridge)
print("\n ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[ ]:


RFR = make_pipeline(RobustScaler(), RandomForestRegressor(n_estimators=500,min_samples_leaf=2,min_samples_split=3))
score = rmsle_cv(RFR)
print("\n RFR score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[ ]:


ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
score = rmsle_cv(ENet)
print("\n ENet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[ ]:


KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.7)
score = rmsle_cv(KRR)
print("\n KRR score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[ ]:


GBoost = GradientBoostingRegressor(n_estimators=3500, learning_rate=0.01,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =9)
score = rmsle_cv(GBoost)
print("\n GBoost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[77]:


model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
score = rmsle_cv(model_xgb)
print("\n model_xgb score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[ ]:


model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
score = rmsle_cv(model_lgb)
print("\n model_lgb score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[75]:


class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)   


# In[ ]:


## withe new GB
averaged_models = AveragingModels(models = (ENet,  ridge, lasso))

score = rmsle_cv(averaged_models)
print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[ ]:


## withe new GB
averaged_models = AveragingModels(models = (ENet, GBoost, ridge, lasso,GBoost))

score = rmsle_cv(averaged_models)
print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[83]:


model_xgb.fit(train, y_train)
xgb_train_pred = model_xgb.predict(train)
xgb_pred = np.expm1(model_xgb.predict(test))


# In[84]:


sub = pd.DataFrame()
sub['Id'] = idTest
sub['SalePrice'] = xgb_pred
sub.to_csv('submission_xgb.csv',index=False)


# In[ ]:



                

