"""
Features, transformation that have not been used yet in the refactored code but come from Tu's
notebook, engineering.ipython

There features need to be adopted to the new code base
"""
import numpy as np 
import pandas as pd 
import pickle
import time




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