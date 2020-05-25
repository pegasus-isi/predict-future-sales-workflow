import pandas as pd
import numpy as np
import re
import pickle
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

"""
Feature Engineering Part 1 Basics

	FILES IN: 
		'items_preprocessed.pickle'
		'shops_preprocessed.pickle',
		'categories_preprocessed.pickle'

	FILES OUT: 
		'shops_preprocessed_1.pickle',
		'categories_preprocessed_1.pickle',
		'items_preprocessed_1.pickle'

 """

# Adds new basic features to the dataframe: 'shop_category_id', 'shop_location_id'
def feature_eng_shops(shops):
	#Create new categories based on shop' s names
	shops["city"]     = shops.shop_name.str.split(" ").map( lambda x: x[0] )
	shops["category"] = shops.shop_name.str.split(" ").map( lambda x: x[1].lower() )

	common_categories = []
	for cat in shops.category.unique():
		if len(shops[shops.category == cat]) > 4:
			common_categories.append(cat)

	shops.category            = shops.category.apply( lambda x: x if (x in common_categories) else "etc" )	
	shops["shop_category_id"] = LabelEncoder().fit_transform( shops.category )
	shops["shop_location_id"] = LabelEncoder().fit_transform( shops.city )
	shops                     = shops[["shop_id", "shop_category_id", "shop_location_id"]]

	return shops

#Index(['item_category_id', 'item_broader_category_name_id','item_broader_category_name'],dtype='object')
def feature_eng_categories(categories):
	categories["type_code"] = categories.item_category_name.apply( lambda x: x.split(" ")[0] ).astype(str)
	
	category = []
	for cat in categories.type_code.unique():
		if len(categories[categories.type_code == cat]) > 4: 
			category.append( cat )

	categories.type_code       = categories.type_code.apply(lambda x: x if (x in category) else "etc")
	categories.type_code       = LabelEncoder().fit_transform(categories.type_code)
	categories["split"]        = categories.item_category_name.apply(lambda x: x.split("-"))
	categories["item_broader_category_name"]     = categories.split.apply(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())
	categories["item_broader_category_name_id"]  = LabelEncoder().fit_transform( categories["item_broader_category_name"] )
	
	categories = categories[["item_category_id", "item_broader_category_name_id"]]
	
	return categories



def feature_eng_items(items):
    def name_correction(x):
        x = x.lower()
        x = x.partition('[')[0]
        x = x.partition('(')[0]
        x = re.sub('[^A-Za-z0-9А-Яа-я]+', ' ', x)
        x = x.replace('  ', ' ')
        x = x.strip()
        return x

    items["item_name_p1"], items["item_name_p2"] = items.item_name.str.split("[", 1).str
    items["item_name_p1"], items["item_name_p3"] = items.item_name.str.split("(", 1).str

    items["item_name_p2"] = items.item_name_p2.str.replace('[^A-Za-z0-9А-Яа-я]+', " ").str.lower()
    items["item_name_p3"] = items.item_name_p3.str.replace('[^A-Za-z0-9А-Яа-я]+', " ").str.lower()
    items                 = items.fillna('0')
    items["item_name"]    = items["item_name"].apply(lambda x: name_correction(x))
    items.item_name_p2    = items.item_name_p2.apply( lambda x: x[:-1] if x !="0" else "0")

    items["type"]          = items["item_name_p1"].apply(lambda x: x[0:8] if x.split(" ")[0] == "xbox" else x.split(" ")[0] )
    items.loc[(items.type  == "x360") | (items.type == "xbox360") | (items.type == "xbox 360") ,"type"] = "xbox 360"
    items.loc[ items.type  == "", "type"] = "mac"

    items.type             = items.type.apply( lambda x: x.replace(" ", "") )
    items.loc[ (items.type == 'pc' )| (items.type == 'pс') | (items.type == "pc"), "type" ] = "pc"
    items.loc[ items.type  == 'рs3' , "type"] = "ps3"

    group_sum = items.groupby(["type"]).agg({"item_id": "count"})
    group_sum = group_sum.reset_index()

    drop_cols = []
    for cat in group_sum.type.unique():
        if group_sum.loc[(group_sum.type == cat), "item_id"].values[0] <40:
            drop_cols.append(cat)

    items["item_name_p2"] = items["item_name_p2"].apply( lambda x: "etc" if (x in drop_cols) else x )
    items["item_name_p2_id"] = LabelEncoder().fit_transform(items["item_name_p2"])
    items["item_name_p3_id"] = LabelEncoder().fit_transform(items["item_name_p3"])

    items = items.drop(["type"], axis = 1)
    items.drop(["item_name", "item_name_p1","item_name_p2","item_name_p3"],axis = 1, inplace= True)

    return items


def main():

    # Read in the data for basic feature engineering
    items            = pd.read_pickle('items_preprocessed.pickle')
    categories       = pd.read_pickle('categories_preprocessed.pickle')
    shops            = pd.read_pickle('shops_preprocessed.pickle')


    shops            = feature_eng_shops(shops)
    categories       = feature_eng_categories(categories)
    items            = feature_eng_items(items)


    pickle.dump(categories, open('categories_preprocessed_1.pickle', 'wb'), protocol = 4)
    pickle.dump(shops, open('shops_preprocessed_1.pickle', 'wb'), protocol = 4)
    pickle.dump(items, open('items_preprocessed_1.pickle', 'wb'), protocol = 4)








if __name__ == "__main__":
    main()