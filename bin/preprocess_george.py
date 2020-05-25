#!/usr/bin/env python3
from IPython import embed
import re
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def remove_outliers(sales_train):
    sales_train = sales_train[(sales_train.item_price < 100000 )& (sales_train.item_cnt_day < 1000)]
    sales_train = sales_train[sales_train.item_price > 0].reset_index(drop = True)
    sales_train.loc[sales_train.item_cnt_day < 1, "item_cnt_day"] = 0
    return

def drop_shops(sales_train, test, shops):
    #### Drop duplicate shops ####
    sales_train.loc[sales_train.shop_id == 0, "shop_id"] = 57
    test.loc[test.shop_id == 0 , "shop_id"] = 57
    
    sales_train.loc[sales_train.shop_id == 1, "shop_id"] = 58
    test.loc[test.shop_id == 1 , "shop_id"] = 58
    
    sales_train.loc[sales_train.shop_id == 11, "shop_id"] = 10
    test.loc[test.shop_id == 11, "shop_id"] = 10
    
    sales_train.loc[sales_train.shop_id == 40, "shop_id"] = 39
    test.loc[test.shop_id == 40, "shop_id"] = 39
    
    shops.drop(0,axis=0,inplace=True)
    shops.drop(1,axis=0,inplace=True)
    shops.drop(11,axis=0,inplace=True)
    shops.drop(40,axis=0,inplace=True)

    #### Drop shops with irregular sales ####
    sales_train.drop(sales_train.loc[sales_train['shop_id']==9].index,axis=0,inplace=True)
    sales_train.drop(sales_train.loc[sales_train['shop_id']==20].index,axis=0,inplace=True)
    sales_train.drop(sales_train.loc[sales_train['shop_id']==33].index,axis=0,inplace=True)
    
    shops.drop(9,axis=0,inplace=True)
    shops.drop(20,axis=0,inplace=True)
    shops.drop(33,axis=0,inplace=True)
    return

def add_shop_city_category(shops):
    shops.loc[ shops.shop_name == 'Сергиев Посад ТЦ "7Я"',"shop_name" ] = 'СергиевПосад ТЦ "7Я"'
    shops["city"] = shops.shop_name.str.split(" ").map( lambda x: x[0] )
    shops["category"] = shops.shop_name.str.split(" ").map( lambda x: x[1] )
    shops.loc[shops.city == "!Якутск", "city"] = "Якутск"

    category = []
    for cat in shops.category.unique():
        if len(shops[shops.category == cat]) > 4:
            category.append(cat)

    shops.category = shops.category.apply( lambda x: x if (x in category) else "etc" )

    shops["shop_category"] = LabelEncoder().fit_transform(shops["category"])
    shops["shop_city"] = LabelEncoder().fit_transform(shops["city"])
    shops = shops[["shop_id", "shop_category", "shop_city"]]
    return

def drop_item_categories(sales_train, items, item_categories):
    sales_train.drop(sales_train.loc[(sales_train['item_id'].map(items['item_category_id'])==8)].index.values,axis=0,inplace=True)
    sales_train.drop(sales_train.loc[(sales_train['item_id'].map(items['item_category_id'])==80)].index.values,axis=0,inplace=True)
    sales_train.drop(sales_train.loc[(sales_train['item_id'].map(items['item_category_id'])==81)].index.values,axis=0,inplace=True)
    sales_train.drop(sales_train.loc[(sales_train['item_id'].map(items['item_category_id'])==82)].index.values,axis=0,inplace=True)
    
    items.drop(items.loc[items['item_category_id']==8].index.values,axis=0,inplace=True)
    items.drop(items.loc[items['item_category_id']==80].index.values,axis=0,inplace=True)
    items.drop(items.loc[items['item_category_id']==81].index.values,axis=0,inplace=True)
    items.drop(items.loc[items['item_category_id']==82].index.values,axis=0,inplace=True)

    item_categories.drop(8,axis=0,inplace=True)
    item_categories.drop(80,axis=0,inplace=True)
    item_categories.drop(81,axis=0,inplace=True)
    item_categories.drop(82,axis=0,inplace=True)
    return

def add_item_subcategories(item_categories):
    item_categories["type_code"] = item_categories.item_category_name.apply( lambda x: x.split(" ")[0] ).astype(str)
    item_categories.loc[ (item_categories.type_code == "Игровые")| (item_categories.type_code == "Аксессуары"), "type_code" ] = "Игры"
    
    category = []
    for cat in item_categories.type_code.unique():
        if len(item_categories[item_categories.type_code == cat]) > 4:
            category.append(cat)

    item_categories.type_code = item_categories.type_code.apply(lambda x: x if (x in category) else "etc")

    item_categories.type_code = LabelEncoder().fit_transform(item_categories.type_code)
    item_categories["split"] = item_categories.item_category_name.apply(lambda x: x.split("-"))
    item_categories["subtype"] = item_categories.split.apply(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())
    item_categories["subtype_code"] = LabelEncoder().fit_transform(item_categories["subtype"])
    item_categories = item_categories[["item_category_id", "subtype_code", "type_code"]]
    return


def preprocess_items(items):
    def name_correction(x):
        x = x.lower()
        x = x.partition('[')[0]
        x = x.partition('(')[0]
        x = re.sub('[^A-Za-z0-9А-Яа-я]+', ' ', x)
        x = x.replace('  ', ' ')
        x = x.strip()
        return x

    items["name1"], items["name2"] = items.item_name.str.split("[", 1).str
    items["name1"], items["name3"] = items.item_name.str.split("(", 1).str

    items["name2"] = items.name2.str.replace('[^A-Za-z0-9А-Яа-я]+', " ").str.lower()
    items["name3"] = items.name3.str.replace('[^A-Za-z0-9А-Яа-я]+', " ").str.lower()
    items = items.fillna('0')

    items["item_name"] = items["item_name"].apply(lambda x: name_correction(x))
    items.name2 = items.name2.apply( lambda x: x[:-1] if x !="0" else "0")

    items["type"] = items["name2"].apply(lambda x: x[0:8] if x.split(" ")[0] == "xbox" else x.split(" ")[0] )
    items.loc[(items.type == "x360") | (items.type == "xbox360") | (items.type == "xbox 360") ,"type"] = "xbox 360"
    items.loc[ items.type == "", "type"] = "mac"
    items.type = items.type.apply( lambda x: x.replace(" ", "") )
    items.loc[ (items.type == 'pc' )| (items.type == 'pс') | (items.type == "pc"), "type" ] = "pc"
    items.loc[ items.type == 'рs3' , "type"] = "ps3"

    group_sum = items.groupby(["type"]).agg({"item_id": "count"})
    group_sum = group_sum.reset_index()

    drop_cols = []
    for cat in group_sum.type.unique():
        if group_sum.loc[(group_sum.type == cat), "item_id"].values[0] <40:
            drop_cols.append(cat)

    items["name2"] = items["name2"].apply( lambda x: "etc" if (x in drop_cols) else x )

    items["name2"] = LabelEncoder().fit_transform(items["name2"])
    items["name3"] = LabelEncoder().fit_transform(items["name3"])

    items = items.drop(["type"], axis = 1)
    items.drop(["item_name", "name1"],axis = 1, inplace= True)
    return

def main():
    test = pd.read_csv("test.csv")
    shops = pd.read_csv("shops.csv")
    items = pd.read_csv("items_old.csv")
    sales_train = pd.read_csv("sales_train.csv")
    item_categories = pd.read_csv("item_categories.csv")

    remove_outliers(sales_train)

    drop_shops(sales_train, test, shops)
    add_shop_city_category(shops)

    drop_item_categories(sales_train,items,item_categories)
    add_item_subcategories(item_categories)

    preprocess_items(items)
    embed()
    pickle.dump(test, open('test_preprocessed.pickle', 'wb'), protocol=4)
    pickle.dump(shops, open('shops_preprocessed.pickle', 'wb'), protocol=4)
    pickle.dump(items, open('items_preprocessed.pickle', 'wb'), protocol=4)
    pickle.dump(item_categories, open('item_categories_preprocessed.pickle', 'wb'), protocol=4)
    pickle.dump(sales_train, open('sales_train_preprocessed.pickle', 'wb'), protocol=4)

    return


if __name__ == "__main__":
    main()
