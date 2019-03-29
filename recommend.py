# -*- coding: utf-8 -*-
"""
Spyder Editor

"""
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
ds = pd.read_csv("sample-data.csv") #you can plug in your own list of products as csv file
tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
######ngram (1,3) can be explained as follows#####
#ngram(1,3) encompasses uni gram, bi gram and tri gram
#consider the sentence "The ball fell"
#ngram (1,3) would be the, ball, fell, the ball, ball fell, the ball fell
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[-()\"#/@;:<>{}+=~|.?,]", "", text) 
    return text

tfidf_matrix = tf.fit_transform(ds['description'])
cosine_similarities = cosine_similarity(tfidf_matrix,tfidf_matrix)

results = {} # dictionary created to store the result in a dictionary format (ID : (Score,item_id))

for idx, row in ds.iterrows(): #iterates through all the rows
    # the below code 'similar_indice' stores similar ids based on cosine similarity. sorts them in ascending order. [:-5:-1] is then used so that the indices with 
    #most similarity are got. 0 means no similarity and 1 means perfect similarity
    similar_indices = cosine_similarities[idx].argsort()[:-5:-1] #stores 5 most similar products, you can change it as per your needs
    similar_items = [(cosine_similarities[idx][i], ds['id'][i]) for i in similar_indices]
    results[row['id']] = similar_items[1:]
    
#below code 'function item(id)' returns a row matching the id along with Product Title. Initially it is a dataframe, then we convert it to a list
def item(id):
    return ds.loc[ds['id'] == id]['product_name'].tolist()[0]
def recommend(id, num):
    if (num == 0):
        print("Unable to recommend any product as you have not chosen the number of products to be recommended")
    elif (num==1):
        print("Recommended for you, " + str(num) + " product similar to " + item(id))
        
    else :
        print("Recommended for you, " + str(num) + " products similar to " + item(id))
        
    print("----------------------------------------------------------")
    recs = results[id][:num]
    for rec in recs:
        print("You may also like to see: " + item(rec[1]) + " (score:" + str(rec[0]) + ")")

#the first argument in the below function to be passed is the id of the book, second argument is the number of books you want to be recommended
recommend(5,3)

