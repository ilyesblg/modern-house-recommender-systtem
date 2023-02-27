# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import uvicorn
from fastapi import FastAPI, Query
import numpy as np
import pickle
import pandas as pd
from typing import Union

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




# . Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
