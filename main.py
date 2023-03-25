# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import uvicorn
from fastapi import FastAPI, Query
import numpy as np
import pickle
import pandas as pd
from typing import Union
from sklearn.preprocessing import StandardScaler ,LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
app = FastAPI()
pickle_in = open("knn-pickle.pkl", "rb")
knn = pickle.load(pickle_in)
dataRecommendation = pd.read_csv('aaa.csv')


@app.get('/predict')
async def Furniture_recommendation(furniture_name: Union[str, None] = Query(default=None, max_length=100)):
    recommendation_result = list(knn.kneighbors([dataRecommendation[furniture_name].values], 5 + 1))
    recommendation_result = pd.DataFrame(np.vstack((recommendation_result[1], recommendation_result[0])),
                                         index=['FurnitureId', 'Cosine_Similarity (degree)']).T
    recommendation_result = recommendation_result.drop([0]).reset_index(drop=True)
    return recommendation_result


df=pd.read_csv('data_furniture.csv')
num_cols = ['price', 'depth', 'height', 'width']
cat_cols = ['category']
scaler = StandardScaler()
encoder = LabelEncoder()
df[num_cols] = scaler.fit_transform(df[num_cols])
df[cat_cols] = df[cat_cols].apply(encoder.fit_transform)

# Apply K-means clustering
n_clusters = 10 # Choose the optimal number of clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df['cluster'] = kmeans.fit_predict(df[num_cols + cat_cols])




@app.get("/similar_products/{product_id}")
async def get_similar_products(product_id: int):
    product = df.loc[df['item_id'] == product_id]
    cluster_id = product['cluster'].values[0]
    cluster = df.loc[df['cluster'] == cluster_id]
    similarity_scores = cosine_similarity(product[num_cols + cat_cols], cluster[num_cols + cat_cols])
    similarity_df = pd.DataFrame({'item_id': cluster['item_id'], 'similarity': similarity_scores[0]})
    similarity_df = similarity_df.sort_values('similarity', ascending=False)
    similar_products = similarity_df.iloc[1:4]['item_id'].values.tolist()
    return similar_products

# . Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
