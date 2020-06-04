#!/usr/bin/env python3

import re
import pickle
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

"""
Preprocesses data for the Future Sales Predictions

	FILES IN: 
		'items.csv',
		'item_categories.csv',
		'shops.csv'


	FILES OUT: 
		'tenNN_items.pickle'
		'items_nlp.pickle'
		'items_clusters.pickle'
		'shops_nlp.pickle'
		'threeNN_shops.pickle'

 """

# -----------------           NLP       -------------------------


"""
WHY id index mapping?  
Some of the categories have been removed, it results in some of the items being removed
e.g. we do not have item_id with id 1, hence need for this mapping between order in indices matrix and ids
"""
def find_kNN(embedding_vec, num_neighbors, col_names, embedding_cols, col_id):
    nbrs               = NearestNeighbors(n_neighbors=num_neighbors, algorithm="ball_tree").fit(embedding_vec[embedding_cols])
    distances, indices = nbrs.kneighbors(embedding_vec[embedding_cols])

    id_map_dict = {}
    for i in range(embedding_vec.shape[0]):
	id_map_dict[i] = embedding_vec[col_id][i]

    for i in range(indices.shape[0]):
        for j in range(indices.shape[1]):
	    index = indices[i][j]
	    indices[i][j] = id_map_dict[index]
	
    return pd.DataFrame(indices, columns = col_names)


def clean_text(words):
    words = words.split()
    words_clean = []
    for word in words:
        word  = word.lower()
        word  = re.sub(r"[^a-zA-Z0-9 ]",r"",word)
        words_clean.append(word)
        
    return " ".join(words_clean)


def create_embeddings(vectorizer, data, embedding_dim, col_embed, col_id):
    vectors                      = vectorizer.fit_transform(data[col_embed])
    feature_names_item_cat       = vectorizer.get_feature_names()
    tf_idf_items                 = pd.DataFrame(vectors.todense())
    X_embedded                   = TSNE(n_components = embedding_dim).fit_transform(vectors)
    items_embedded_df            = pd.DataFrame(X_embedded)
    items_embedded_df[col_id]    = data[col_id]
	
    return items_embedded_df


def kmeans_cluster(embedded_df, embed_cols, col_id, num_clusters):
    kmeans_model = KMeans(n_clusters=num_clusters, random_state=1).fit(embedded_df[embed_cols])
    labels       = kmeans_model.labels_
    embedded_df["cluster_memberships"] = labels
	
    return embedded_df[[col_id, "cluster_memberships"]]


# -----------------           NLP CALLERS      -------------------------

def items_nlp(items, categories):
    item_and_category   = pd.merge(items, categories, on="item_category_id")
    item_and_category["item_category_name"] = item_and_category["item_name"] + " " + item_and_category["item_category_name"].str.lower()
    vectorizer          = TfidfVectorizer()
    items_embedded_df   = create_embeddings(vectorizer, item_and_category, 3, "item_category_name", "item_id")
    col_names_items     = ["item_id", "1NN_id", "2NN_id", "3NN_id", "4NN_id", "5NN_id", "6NN_id", "7NN_id", "8NN_id", "9NN_id", "10NN_id"]
    embedding_cols      = [0, 1, 2]
    tenNN_items         = find_kNN(items_embedded_df, 11, col_names_items, embedding_cols, "item_id")
    item_cluster_member = kmeans_cluster(items_embedded_df, embedding_cols, "item_id", 18)
	
    return tenNN_items, items_embedded_df, item_cluster_member


def shops_nlp(shops):
    vectorizer           = CountVectorizer()
    processed_shop_names = []
    for i in range(shops.shape[0]):
	processed_shop_names.append(clean_text(shops["shop_name"][i]))
    shops["clean_shop_name"] = pd.Series(processed_shop_names)  
    shops_embedded_df = create_embeddings(vectorizer, shops, 2, "clean_shop_name", "shop_id")
    col_names_shops   = ["shop_id", "1NN_shop", "2NN_shop", "3NN_shop"]
    threeNN_shops     = find_kNN(shops_embedded_df, 4, col_names_shops, [0,1], "shop_id")
	
    return threeNN_shops, shops_embedded_df


def main():
    # Read in the data for the analysis
    items             = pd.read_csv("items.csv")
    categories        = pd.read_csv("item_categories.csv")
    shops             = pd.read_csv("shops.csv")

    tenNN_items, items_embedded_df, items_clusters = items_nlp(items, categories)
    threeNN_shops, shops_embedded_df               = shops_nlp(shops)

    pickle.dump(tenNN_items, open("tenNN_items.pickle", "wb"), protocol=4)
    pickle.dump(items_embedded_df, open("items_nlp.pickle", "wb"), protocol=4)
    pickle.dump(items_clusters, open("items_clusters.pickle", "wb"), protocol=4)

    pickle.dump(shops_embedded_df, open("shops_nlp.pickle", "wb"), protocol=4)
    pickle.dump(threeNN_shops, open("threeNN_shops.pickle", "wb"), protocol=4)


if __name__ == "__main__":
    main()
