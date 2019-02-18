# %matplotlib inline

# import itertools
# import warnings
# warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set_style("darkgrid")

import math as mt
import time
import re
from datetime import datetime

from pymongo import MongoClient
from bson.json_util import dumps

from surprise import Reader, Dataset, SVD, evaluate
from scipy.sparse import csr_matrix, csc_matrix
from sparsesvd import sparsesvd
from scipy.sparse.linalg import *

#content filtering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import util.Recommenders_pro as Recommenders
import util.Evaluation_pro as Evaluation
from util.mongodb_data import mongodb_import


class blog_train_model(mongodb_import):
    def __init__(self):
        
        self.mongodb_interface = mongodb_import()
        self.super_db = self.mongodb_interface.super_mongodb_interface
        
        self.users, self.views, self.blogs, self.views_pivot, self.content_data  = self.load_model_mongo()
        
        #popularity model
        self.popularity_model = self.train_popularity_model()
        
        ## constants defining the dimensions of our user rating matrix
        self.MAX_PID = self.views_pivot.shape[1]
        self.MAX_UID = self.views_pivot.shape[0]
        
        self.coumpute_SVD()
        self.cosine_sim = self.content_filtering()
        
        
    
    #loading users, blogs and ratings dataset and making pivot table
    def load_model_mongo(self):
        
        views = self.super_db.read_mongo(self.super_db.views_db)
        users = self.super_db.read_mongo(self.super_db.users_db, index="userid")
        blogs = self.super_db.read_mongo(self.super_db.blogs_db, index="blogid")
        content = self.super_db.read_mongo(self.super_db.content_db)

        views["viewedcount"] = views["viewedcount"].astype(float)

        # ratings to pivot table
        views_pivot = pd.pivot_table(views, values="viewedcount", index="userid", columns="blogid")
        return users, views, blogs, views_pivot, content
    
    
    ##POPULARITY MODEL
    
    ## create an instance of popularity recommenders 
    def train_popularity_model(self):
        
        pm = Recommenders.popularity_recommender_py()
        pm.create(self.views, 'userid', 'blogid')
        return pm

#         blogs.loc[pm.recommend("")["movieId"][:5].values]["title"].to_dict()
    
    #predicting popularity rating model
    def predict_popularityModel(self):
        return self.blogs.loc[self.popularity_model.recommend("")["blogid"][:6].values]
    
    
    
    
    ## CONTENT BASED FILTERING
    
    # convert tfidf vectorizer from the content blog tags
    def content_filtering(self):
        
        # tfidf = TfidfVectorizer()
        # data = tfidf.fit_transform(self.blogs["tags"]).toarray()

        data = self.content_data
    
        # creating a Series for the blog titles so they are associated to an ordered numerical
        # list I will use in the function to match the indexes
        self.indices = pd.Series(self.blogs.index)
        #cosine similarity with items
        return cosine_similarity(data, data)
    
    #  defining the function that takes in movie title 
    # as input and returns the top 10 recommended movies
    def content_recommendations(self, idx, cosine_sim):

        # creating a Series with the similarity scores in descending order
        score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)

        # getting the indexes of the 10 most similar movies
        return list(score_series.iloc[1:11].index)
    
    # predict the content filtering blogs
    def content_predict(self, item_id):
        
        cnt_recomment = self.content_recommendations(item_id, self.cosine_sim)
        return self.blogs.iloc[cnt_recomment]
         
    
    
    ## PERSONALIZATION WITH SINGULAR VALUE DECOMPOSITION
    
    def coumpute_SVD(self):
        #Used in SVD calculation (number of latent factors)
        K=2

        #Initialize a sample user rating matrix
        urm = self.views_pivot.fillna(0).values

        urm = csc_matrix(urm, dtype=np.float32)

        #Compute SVD of the input user ratings matrix
        self.U, self.S, self.Vt = self.computeSVD(urm, K)
        
        self.urm = urm
        self.K = K
    
    def predict_estimated_Ratings(self, user):

        #Test user set as user_id 4 with ratings [0, 0, 5, 0]
        uTest = [user]
        print("User id for whom recommendations are needed: %d" % uTest[0])

        #Get estimated rating for test user
        print("Predictied ratings:")
        
        uTest_SVD_recommendation = self.computeEstimatedRatings(self.urm, self.U, self.S, self.Vt, uTest, self.K, True, self.MAX_UID, self.MAX_PID)
        
        return self.blogs.iloc[uTest_SVD_recommendation[:10]]
        
         
    
    
        # mTest_recommended_items = computeEstimatedRatings(urm_m, U_m, S_m, Vt_m, mTest, K, True, MAX_UID_m, MAX_PID_m)
        # print(uTest_recommended_items[:5])
        # print(mTest_recommended_items[:25])


    #Compute SVD of the user ratings matrix
    def computeSVD(self, urm, k):
        U, s, Vt = sparsesvd(urm, k)
    
        dim = (len(s), len(s))
        S = np.zeros(dim, dtype=np.float32)
        for i in range(0, len(s)):
            S[i, i] = mt.sqrt(s[i])
    
        U = csc_matrix(np.transpose(U), dtype=np.float32)
        S = csc_matrix(S, dtype=np.float32)
        Vt = csc_matrix(Vt, dtype=np.float32)
    
        return U, S, Vt

    #Compute estimated rating for the test user
    def computeEstimatedRatings(self, urm, U, S, Vt, uTest, K, test, MAX_UID, MAX_PID):
        rightTerm = S*Vt
    
        estimatedRatings = np.zeros(shape=(MAX_UID, MAX_PID), dtype=np.float16)
        for userTest in uTest:
            prod = U[userTest,:]*rightTerm
        
            estimatedRatings[userTest,:] = prod.todense()
            recom = (-estimatedRatings[userTest,:]).argsort()[:250]
        return recom