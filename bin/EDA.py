#!/usr/bin/env python3
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

"""
Exploratory Data Analysis of Kaggle's Predict Future Sales Dataset.
This script produces a PDF file with different types of plots of data.

SCRIPT: EDA.py

Exploratory Data Analysis of Kaggle's Predict Future Sales Dataset.
This script produces a PDF file with different types of plots of data.

    FILES IN: 
        "sales_train.csv"
        "items.csv"
        "item_categories.csv"
        "shops.csv"


    FILES OUT: 
        'EDA.pdf'
"""


FIG_SIZE_X = 18
FIG_SIZE_Y = 10

MONTHS = ['Jan13', 'Feb13', 'Mar13', 'Apr13','May13','Jun13', 'Jul13', 'Aug13', 'Sep13', 'Oct13','Nov13', 'Dec13',
             'Jan14', 'Feb14', 'Mar14', 'Apr14','May14','Jun14', 'Jul14', 'Aug14', 'Sep14', 'Oct14','Nov14', 'Dec14',
             'Jan15', 'Feb15', 'March15', 'Apr15','May15','Jun15', 'Jul15', 'Aug15', 'Sep15', 'Oct15']




def create_hist_series(output_file, sales_trainset, group_key_size, properites_dict):
    groupedby_blocknum = sales_trainset.groupby(["date_block_num",properites_dict['group_by']])
    group_tuple_keys   = list(groupedby_blocknum.groups.keys())
    num_months         = len(sales_trainset["date_block_num"].unique())
    num_tuple_groups   = len(group_tuple_keys)
    month_dict         = dict(zip(range(0,num_months),MONTHS))

    key_index      = 0
    all_group      = []
    max_key_index  = num_tuple_groups -1

    for i in range(0,num_months):
        all_group_key_month = dict.fromkeys(range(0,group_key_size),0)
        while( (group_tuple_keys[key_index][0] == i) & (key_index < max_key_index) ) :
            z = group_tuple_keys[key_index][1]
            all_group_key_month[z] = groupedby_blocknum.get_group((i,z)).count()[0]
            key_index += 1
        all_group.append(all_group_key_month)

    fig, axs = plt.subplots(12, 3, figsize=(FIG_SIZE_X,45))
    fig.tight_layout(pad = 4.0)
    for m in range(0,12):
        k = m
        max_row = 3 if m < 9 else 2
        for i in range(0,max_row):
            axs[m][i].set_ylim(0,properites_dict["ylim"])
            axs[m][i].set_xlim(0,properites_dict["xlim"])
            values_list = list(all_group[k].values())
            axs[m][i].bar(range(group_key_size), values_list, color= properites_dict["color"] )
            axs[m][i].set_title("Month: " + MONTHS[k])
            for a, txt in enumerate(values_list):
                if values_list[a]> 5000:
                    axs[m,i].text(a,values_list[a],txt )
            k+=12
    for ax in axs.flat:
        ax.set(xlabel = properites_dict["x_label"], ylabel=properites_dict["y_label"])
    output_file.savefig()



def create_rows_barcharts(output_file, list_barcharts_dict, num_figures):
 
    fig, axs = plt.subplots(num_figures, figsize=(FIG_SIZE_X,FIG_SIZE_Y+4))
    fig.tight_layout(pad = 4.0)

    for i in range(num_figures):
    	axs[i].bar(list_barcharts_dict[i]["width"],list_barcharts_dict[i]["data"], color="green")
    	axs[i].set_title(list_barcharts_dict[i]["title"],fontsize = 24)
    	axs[i].set(xlabel = list_barcharts_dict[i]["x_label"], ylabel = "Sales")
    	for j,v in enumerate(list_barcharts_dict[i]["data"]):
    		if (v > list_barcharts_dict[i]["max_val"]):
    			axs[i].text(j,list_barcharts_dict[i]["data"][j],int(v))
    output_file.savefig()


def create_scatter_plot_items_sold(output_file, sales_trainset):

    nitems_sold = pd.DataFrame(sales_trainset["item_id"].value_counts())
    nitems_sold = nitems_sold.reset_index()
    nitems_sold = nitems_sold.rename(columns= {"index": "item_id","item_id":"sold_count" })
    colors      = np.random.rand((len(nitems_sold)-1))

    plt.figure(figsize=(FIG_SIZE_X,FIG_SIZE_Y))
    plt.scatter(nitems_sold["item_id"][1:],nitems_sold["sold_count"][1:],c = colors)
    plt.title('Number of items sold of each product type')
    plt.xlabel('Product item')
    plt.ylabel('Number of items sold')

    output_file.savefig()


def create_scatter_plot_sales_month(output_file, sales_trainset):

    nitems_sold_month = pd.DataFrame(sales_trainset["date_block_num"].value_counts())
    nitems_sold_month = nitems_sold_month.reset_index()
    nitems_sold_month = nitems_sold_month.rename(columns= {"index": "month","date_block_num":"monthly_sold_count" })
    nitems_sold_month = nitems_sold_month.sort_values("month")

    colors = ['orange', 'violet','cyan', 'blue', 'purple', 'pink', 'red', 'teal','y', 'brown', 'grey', 'black',
         'orange', 'violet','cyan', 'blue', 'purple', 'pink', 'red', 'teal','y', 'brown', 'grey', 'black',
         'orange', 'violet','cyan', 'blue', 'purple', 'pink', 'red', 'teal','y', 'brown']

    fig, ax = plt.subplots(figsize=(FIG_SIZE_X,FIG_SIZE_Y))

    ax.scatter(nitems_sold_month["month"],nitems_sold_month["monthly_sold_count"],c= colors)
    nitems_sold_month_list      = nitems_sold_month["month"].tolist()
    nitems_sold_month_sold_list = nitems_sold_month["monthly_sold_count"].tolist()

    for i, txt in enumerate(MONTHS):
        ax.annotate(txt, (nitems_sold_month_list[i], nitems_sold_month_sold_list[i]))
    plt.title('NUMBER OF ITEMS SOLD EACH MONTH')
    plt.xlabel('MONTHS')
    plt.ylabel('Number of items sold')
    plt.xticks([i for i in range(0,34)], MONTHS, rotation = 80)
    output_file.savefig()



def create_boxplot(output_file, plot_data, x_min, x_max,plt_title):

	plt.figure(figsize = (FIG_SIZE_X,FIG_SIZE_Y-4))
	plt.xlim(x_min, x_max)
	plt.title(plt_title)
	sns.boxplot( x = plot_data )
	output_file.savefig()

def create_first_page(pdf):
    firstPage = plt.figure(figsize=(FIG_SIZE_X,FIG_SIZE_Y))
    text      = 'Exploratory Data Analysis of Kaggle\'s Future Sales Dataset'
    firstPage.text(0.5,0.5,text, size=24, ha="center")
    pdf.savefig()

def prepare_data_row_barcharts(sales_trainset):
    monthly_sales = sales_trainset.groupby(["date_block_num"])["item_cnt_day"].agg(["sum"]).reset_index()
    shop_sales    = sales_trainset.groupby(["shop_id"])["item_cnt_day"].agg(["sum"]).reset_index().rename(columns={'sum':'total_sale'})
    list_barcharts_dict = []
    list_barcharts_dict.append({"width": MONTHS, "data": monthly_sales["sum"].values, "title": "Number of Items Sold Each Month",
        "x_label": "MONTHS", "max_val":1000})
    list_barcharts_dict.append({"width": range(shop_sales.shape[0]), "data": shop_sales["total_sale"].values, "title": "Total Sales in Each Store",
     "x_label":"SHOP ID", "max_val":60000})

    return list_barcharts_dict   

def main():

    # Read in the data for the analysis
    sales_trainset = pd.read_csv("sales_train.csv")
    items          = pd.read_csv("items.csv")
    categories     = pd.read_csv("item_categories.csv")
    shops          = pd.read_csv("shops.csv")

    # Merge the files by Item, Shop and Category ids
    sales_trainset = pd.merge(sales_trainset, items , on = "item_id")
    sales_trainset = pd.merge(sales_trainset, categories , on = "item_category_id")
    sales_trainset = pd.merge(sales_trainset, shops, on = "shop_id")

    # Create pdf file for some of the graphs
    pdf = PdfPages('EDA.pdf')
    create_first_page(pdf)

    # Create scatter plot of items sold and monthly total
    create_scatter_plot_items_sold(pdf, sales_trainset)
    create_scatter_plot_sales_month(pdf, sales_trainset)

    # Dict for data
    categ_dict = {"group_by" : "item_category_id", "color": "orange", "ylim": 32000, "xlim" : 85,
                   "x_label" : "Category of Item's Id", "y_label" : "# of Items Sold in the Category" }
    shops_dict = {"group_by" : "shop_id", "color": "violet", "ylim": 12000, "xlim" : 61,
                   "x_label" : "Shop Id", "y_label":"# of Items Sold in the Shop"}  

    create_hist_series(pdf, sales_trainset, len(categories), categ_dict)
    create_hist_series(pdf, sales_trainset, len(shops), shops_dict)

    create_boxplot(pdf, sales_trainset.item_cnt_day, -50,800,"Count of Items Sold per Day")
    create_boxplot(pdf, sales_trainset.item_price, sales_trainset.item_price.min(),
    	sales_trainset.item_price.max()*1.1,"Prices of Items")

    list_barcharts_dict = prepare_data_row_barcharts(sales_trainset)
    create_rows_barcharts(pdf, list_barcharts_dict, num_figures = 2)
    
    pdf.close()



if __name__ == "__main__":
    main()
