"""
Features, transformation that have not been used yet in the refactored code but come from Tu's
notebook, engineering.ipython

There features need to be adopted to the new code base
"""
import numpy as np 
import pandas as pd 
import pickle
import time

train["revenue"] = train["item_cnt_day"] * train["item_price"]
#-----------------------------------------------------------------------------------------------
ts = time.time()
group = train.groupby( ["item_id"] ).agg({"item_price": ["mean"]})
group.columns = ["item_avg_item_price"]
group.reset_index(inplace = True)

matrix = matrix.merge( group, on = ["item_id"], how = "left" )
matrix["item_avg_item_price"] = matrix.item_avg_item_price.astype(np.float16)


group = train.groupby( ["date_block_num","item_id"] ).agg( {"item_price": ["mean"]} )
group.columns = ["date_item_avg_item_price"]
group.reset_index(inplace = True)

matrix = matrix.merge(group, on = ["date_block_num","item_id"], how = "left")
matrix["date_item_avg_item_price"] = matrix.date_item_avg_item_price.astype(np.float16)
lags = [1, 2, 3]
matrix = lag_feature( matrix, lags, ["date_item_avg_item_price"] )
for i in lags:
    matrix["delta_price_lag_" + str(i) ] = (matrix["date_item_avg_item_price_lag_" + str(i)]- matrix["item_avg_item_price"] )/ matrix["item_avg_item_price"]

def select_trends(row) :
    for i in lags:
        if row["delta_price_lag_" + str(i)]:
            return row["delta_price_lag_" + str(i)]
    return 0

matrix["delta_price_lag"] = matrix.apply(select_trends, axis = 1)
matrix["delta_price_lag"] = matrix.delta_price_lag.astype( np.float16 )
matrix["delta_price_lag"].fillna( 0 ,inplace = True)

features_to_drop = ["item_avg_item_price", "date_item_avg_item_price"]
for i in lags:
    features_to_drop.append("date_item_avg_item_price_lag_" + str(i) )
    features_to_drop.append("delta_price_lag_" + str(i) )
matrix.drop(features_to_drop, axis = 1, inplace = True)
time.time() - ts

#-----------------------------------------------------------------------------------------------

s = time.time()
group = train.groupby( ["date_block_num","shop_id"] ).agg({"revenue": ["sum"] })
group.columns = ["date_shop_revenue"]
group.reset_index(inplace = True)

matrix = matrix.merge( group , on = ["date_block_num", "shop_id"], how = "left" )
matrix['date_shop_revenue'] = matrix['date_shop_revenue'].astype(np.float32)

group = group.groupby(["shop_id"]).agg({ "date_block_num":["mean"] })
group.columns = ["shop_avg_revenue"]
group.reset_index(inplace = True )

matrix = matrix.merge( group, on = ["shop_id"], how = "left" )
matrix["shop_avg_revenue"] = matrix.shop_avg_revenue.astype(np.float32)
matrix["delta_revenue"] = (matrix['date_shop_revenue'] - matrix['shop_avg_revenue']) / matrix['shop_avg_revenue']
matrix["delta_revenue"] = matrix["delta_revenue"]. astype(np.float32)
lags = [1]
matrix = lag_feature(matrix, [1], ["delta_revenue"])
# for i in lags:
#     matrix["delta_revenue_lag_" + str(i) ] = (matrix["date_shop_revenue_lag_" + str(i)]- matrix['shop_avg_revenue'] )/ matrix["shop_avg_revenue"]
matrix["delta_revenue_lag_1"] = matrix["delta_revenue_lag_1"].astype(np.float32)
matrix.drop( ["date_shop_revenue", "shop_avg_revenue", "delta_revenue"] ,axis = 1, inplace = True)
time.time() - ts


#-----------------------------------------------------------------------------------------------

ts = time.time()
group = train.groupby( ["shop_id", "item_id"] ).agg({"item_price": ["mean"]})
group.columns = ["shop_item_avg_item_price"]
group.reset_index(inplace = True)

matrix = matrix.merge( group, on = ["shop_id", "item_id"], how = "left" )
matrix["shop_item_avg_item_price"] = matrix.shop_item_avg_item_price.astype(np.float16)


group = train.groupby( ["date_block_num", "shop_id", "item_id"] ).agg( {"item_price": ["mean"]} )
group.columns = ["date_shop_item_avg_item_price"]
group.reset_index(inplace = True)

matrix = matrix.merge(group, on = ["date_block_num","shop_id", "item_id"], how = "left")
matrix["date_shop_item_avg_item_price"] = matrix.date_shop_item_avg_item_price.astype(np.float16)
lags = [1, 2, 3]
matrix = lag_feature( matrix, lags, ["date_shop_item_avg_item_price"] )
for i in lags:
    matrix["delta_shop_price_lag_" + str(i) ] = (matrix["date_shop_item_avg_item_price_lag_" + str(i)]- matrix["shop_item_avg_item_price"] )/ matrix["shop_item_avg_item_price"]

def select_trends(row) :
    for i in lags:
        if row["delta_shop_price_lag_" + str(i)]:
            return row["delta_shop_price_lag_" + str(i)]
    return 0

matrix["delta_shop_price_lag"] = matrix.apply(select_trends, axis = 1)
matrix["delta_shop_price_lag"] = matrix.delta_shop_price_lag.astype( np.float16 )
matrix["delta_shop_price_lag"].fillna( 0 ,inplace = True)

features_to_drop = ["shop_item_avg_item_price", "date_shop_item_avg_item_price"]
for i in lags:
    features_to_drop.append("date_shop_item_avg_item_price_lag_" + str(i) )
    features_to_drop.append("delta_shop_price_lag_" + str(i) )
matrix.drop(features_to_drop, axis = 1, inplace = True)
time.time() - ts

#------------------------------------------------------------------------------------------
matrix['shop_months_since_opening']=matrix['date_block_num']-matrix['shop_id'].map(matrix[['date_block_num','shop_id']].groupby('shop_id').min()['date_block_num'])
matrix['shop_opening']=(matrix['shop_months_since_opening']==0)
matrix['item_month_id_of_release']=matrix['item_id'].map(matrix[['date_block_num','item_id']].groupby('item_id').min()['date_block_num'])
matrix['item_month_of_release']=matrix['item_month_id_of_release']%12+1
matrix['item_months_since_release']=matrix['date_block_num']-matrix['item_month_id_of_release']
matrix['item_months_since_release'].clip(0,12,inplace=True)  
matrix['item_new']=(matrix['item_months_since_release']==0)
# month where the item has first been sold in shop
group=train[['date_block_num','shop_id','item_id']].groupby(['shop_id','item_id']).min().rename({'date_block_num':'item_month_id_of_first_sale_in_shop'},axis=1)
matrix = pd.merge(matrix, group, on=['shop_id', 'item_id'], how='left')
matrix['item_month_of_first_sale_in_shop']=matrix['item_month_id_of_first_sale_in_shop']%12+1

# number of months since the item has been released in this shop
matrix['item_months_since_first_sale_in_shop']=(matrix['date_block_num']-matrix['item_month_id_of_first_sale_in_shop'])
matrix['item_months_since_first_sale_in_shop'].clip(0,12,inplace=True)  # group together items sold for more than a year in the shop

# whether the item has already been sold in this shop before or not
matrix['item_never_sold_in_shop_before']=~(matrix['item_months_since_first_sale_in_shop']>0)

# set month of release and number of months since the item has been released in this shop to -1 for all items never sold in shop before (remove info from future)
matrix.loc[matrix['item_never_sold_in_shop_before'],'item_months_since_first_sale_in_shop']=-1
matrix.loc[matrix['item_never_sold_in_shop_before'],'item_month_id_of_first_sale_in_shop']=-1
matrix.loc[matrix['item_never_sold_in_shop_before'],'item_month_of_first_sale_in_shop']=-1

# downcast dtype
matrix['item_months_since_first_sale_in_shop']=matrix['item_months_since_first_sale_in_shop'].astype(np.int8)
matrix['item_month_id_of_first_sale_in_shop']=matrix['item_month_id_of_first_sale_in_shop'].astype(np.int8)
matrix['item_month_of_first_sale_in_shop']=matrix['item_month_of_first_sale_in_shop'].astype(np.int8)

matrix['item_never_sold_in_shop_before_but_not_new']=((1-matrix['item_new'])*matrix['item_never_sold_in_shop_before']).astype(bool)
matrix['item_seniority']=(2-matrix['item_new'].astype(int)-matrix['item_never_sold_in_shop_before'].astype(int)).astype(np.int8)


#------------------------------------------------------------------------------------------
tmp_list=[]
for mid in matrix['date_block_num'].unique():
    tmp=train.loc[train['date_block_num']<mid,['date_block_num','shop_id','item_id']].groupby(['shop_id','item_id']).last().rename({'date_block_num':'item_month_id_of_last_sale_in_shop'},axis=1).astype(np.int16)
    tmp['date_block_num']=mid
    tmp.reset_index(inplace=True)
    tmp_list.append(tmp) 
tmp=pd.concat(tmp_list)
del tmp_list
matrix=matrix.join(tmp.set_index(['date_block_num','shop_id','item_id']),on=['date_block_num','shop_id','item_id'])
del tmp

# downcast dtype (int not possible due to NaN values)
matrix['item_month_id_of_last_sale_in_shop']=matrix['item_month_id_of_last_sale_in_shop'].astype(np.float16)

# time since last sale in shop
matrix['item_months_since_last_sale_in_shop']=matrix['date_block_num']-matrix['item_month_id_of_last_sale_in_shop']


#------------------------------------------------------------------------------------------
tmp_list=[]
for mid in matrix['date_block_num'].unique():
    tmp=train.loc[train['date_block_num']<mid,['date_block_num','item_id']].groupby('item_id').last().rename({'date_block_num':'item_month_id_of_last_sale'},axis=1)
    tmp['date_block_num']=mid
    tmp.reset_index(inplace=True)
    tmp_list.append(tmp)
    
tmp=pd.concat(tmp_list)
del tmp_list
matrix=matrix.join(tmp.set_index(['date_block_num','item_id']),on=['date_block_num','item_id'])
del tmp

# downcast dtype (int not possible due to NaN values)
matrix['item_month_id_of_last_sale']=matrix['item_month_id_of_last_sale'].astype(np.float16)

# time since last sale over all shops
matrix['item_months_since_last_sale']=matrix['date_block_num']-matrix['item_month_id_of_last_sale']


#------------------------------------------------------------------------------------------


matrix['item_sold_in_shop']=(matrix['item_cnt_month']>0)
matrix=matrix.join(matrix[['date_block_num','item_category_id']].groupby('date_block_num')['item_category_id'].value_counts(normalize=True).rename('item_category_freq',axis=1).astype(np.float32),on=['date_block_num','item_category_id'])
matrix=matrix.join(matrix[['date_block_num','subtype_code']].groupby('date_block_num')['subtype_code'].value_counts(normalize=True).rename('subtype_freq',axis=1).astype(np.float32),on=['date_block_num','subtype_code'])
matrix=matrix.join(matrix[['date_block_num','type_code']].groupby('date_block_num')['type_code'].value_counts(normalize=True).rename('type_freq',axis=1).astype(np.float32),on=['date_block_num','type_code'])

matrix=matrix.join(matrix[['date_block_num','item_seniority','item_id']].groupby(['date_block_num','item_seniority'])['item_id'].value_counts(normalize=True).rename('item_freq_in_seniority',axis=1).astype(np.float32),on=['date_block_num','item_seniority','item_id'])
matrix=matrix.join(matrix[['date_block_num','item_seniority','shop_id']].groupby(['date_block_num','item_seniority'])['shop_id'].value_counts(normalize=True).rename('shop_freq_in_seniority',axis=1).astype(np.float32),on=['date_block_num','item_seniority','shop_id'])


matrix=matrix.join(matrix[['date_block_num','item_seniority','item_category_id']].groupby(['date_block_num','item_seniority'])['item_category_id'].value_counts(normalize=True).rename('item_category_freq_in_seniority',axis=1).astype(np.float32),on=['date_block_num','item_seniority','item_category_id'])
matrix=matrix.join(matrix[['date_block_num','item_seniority','subtype_code']].groupby(['date_block_num','item_seniority'])['subtype_code'].value_counts(normalize=True).rename('subtype_freq_in_seniority',axis=1).astype(np.float32),on=['date_block_num','item_seniority','subtype_code'])
matrix=matrix.join(matrix[['date_block_num','item_seniority','type_code']].groupby(['date_block_num','item_seniority'])['type_code'].value_counts(normalize=True).rename('type_freq_in_seniority',axis=1).astype(np.float32),on=['date_block_num','item_seniority','type_code'])

#-------------------------------------------------------------------------
matrix_0=matrix.loc[matrix['item_seniority']==0,:]
matrix_1=matrix.loc[matrix['item_seniority']==1,:]
matrix_2=matrix.loc[matrix['item_seniority']==2,:]
pickle.dump(matrix_0, open('train_0.pickle', 'wb'))
pickle.dump(matrix_1, open('train_1.pickle', 'wb'))
pickle.dump(matrix_2, open('train_2.pickle', 'wb'))