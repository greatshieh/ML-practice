import pandas as pd
import random
#from scipy.
import matplotlib.pyplot as plt
import numpy as np

trainSet = pd.read_csv('Train.csv')  #, index_col='Item_Identifier')  # 导入数据
testSet = pd.read_csv('Test.csv')  #, index_col='Item_Identifier')  # 导入数据

trainSet['source'] = 'train'
testSet['source'] = 'test'

#print(trainSet.shape)
#missdata = trainSet.apply(lambda x: sum(x.isnull()), axis=0)  # 计算缺失值
#print(missdata)  # 显示缺失值
#print(trainSet.apply(lambda x:len(x.unique())))
#
#print(testSet.shape)
#missdata = testSet.apply(lambda x: sum(x.isnull()), axis=0)  # 计算缺失值
#print(missdata)  # 显示缺失值
#print(testSet.apply(lambda x:len(x.unique())))
#
dataSet = pd.concat([trainSet, testSet], ignore_index=True)  # 合并训练数据和测试数据
#print(dataSet.shape)
#missdata = dataSet.apply(lambda x: sum(x.isnull()), axis=0)  # 计算缺失值
#print(missdata)  # 显示缺失值
#print(dataSet.apply(lambda x:len(x.unique())))
#
#print(dataSet.describe())
#
##Filter categorical variables
#categorical_columns = [x for x in dataSet.columns if dataSet.dtypes[x]=='object']
##Exclude ID cols and source:
#categorical_columns = [x for x in categorical_columns if x not in ['Item_Identifier', 'Outlet_Identifier', 'source']]
#
#for col in categorical_columns:
#    print('\nFrequency of Categories for varible %s' % col)
#    print(dataSet[col].value_counts())
#
#STEP2, DATA CLEANING
#计算分组平均数，pivot-table默认聚合类型
#item_avg_weight = dataSet.pivot_table(values='Item_Weight', index='Item_Identifier')
##标记缺失值'Item_Weight'
#miss_bool = dataSet['Item_Weight'].isnull()
#print('#total miss data is #%d' % sum(miss_bool))
##填充缺失值
#dataSet.ix[miss_bool, 'Item_Weight'] = dataSet.ix[miss_bool, 'Item_Identifier'].apply(lambda x: item_avg_weight[x])
#miss_bool = dataSet['Item_Weight'].isnull()
#print('#total miss data is #%d' % sum(miss_bool))

#outlet_size_mode = dataSet.pivot_table(values='Outlet_Size', columns='Outlet_Type',aggfunc=(lambda x:mode(x).mode[0]) )
#print(outl#et_size_mode)

#miss_bool = dataSet['Outlet_Size'].isnull()
#print(dataSet.ix[miss_bool, ['Outlet_Size', 'Outlet_Type']])
'''
Outlet_Size的图形分析，
Outlet_Size和Outlet_Type，以及Outlet_Location_Type有关

outlet_size = dataSet[['Outlet_Size', 'Outlet_Type', 'Outlet_Location_Type']]
outlet_size = outlet_size.dropna()
outlet_size.ix[outlet_size['Outlet_Size']=='Small', 'Outlet_Size'] = 1.0
outlet_size.ix[outlet_size['Outlet_Size']=='Medium', 'Outlet_Size'] = 2.0
outlet_size.ix[outlet_size['Outlet_Size']=='High', 'Outlet_Size'] = 3.0

outlet_size.ix[outlet_size['Outlet_Type']=='Grocery Store', 'Outlet_Type'] = 1.0
outlet_size.ix[outlet_size['Outlet_Type']=='Supermarket Type1', 'Outlet_Type'] = 2.0
outlet_size.ix[outlet_size['Outlet_Type']=='Supermarket Type2', 'Outlet_Type'] = 3.0
outlet_size.ix[outlet_size['Outlet_Type']=='Supermarket Type3', 'Outlet_Type'] = 4.0

outlet_size.ix[outlet_size['Outlet_Location_Type']=='Tier 1', 'Outlet_Location_Type'] = 'r'
outlet_size.ix[outlet_size['Outlet_Location_Type']=='Tier 2', 'Outlet_Location_Type'] = 'g'
outlet_size.ix[outlet_size['Outlet_Location_Type']=='Tier 3', 'Outlet_Location_Type'] = 'b'

T = outlet_size['Outlet_Location_Type']


for x in outlet_size.index:
    outlet_size.ix[x, 'Outlet_Size'] = outlet_size.ix[x, 'Outlet_Size'] + random.uniform(-0.3,0.3)
    outlet_size.ix[x, 'Outlet_Type'] = outlet_size.ix[x, 'Outlet_Type'] + random.uniform(-0.3, 0.3)

plt.scatter(outlet_size['Outlet_Type'], outlet_size['Outlet_Size'], c=np.array(T), alpha=0.2)
plt.show()
'''
miss_bool = dataSet['Outlet_Size'].isnull()
print('#Total miss data is %d'% sum(miss_bool))

dataSet.ix[miss_bool, 'Outlet_Size'] = 'Small'
miss_bool = dataSet['Outlet_Size'].isnull()
print('#Total miss data is %d' % sum(miss_bool))