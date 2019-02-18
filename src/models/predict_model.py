import pandas as pd
import numpy as np

from models.train_model import blog_train_model

class blog_predict_model():
    
    #inital the model
    def __init__(self):
#         print('')
        self.train_model = blog_train_model()
        self.users = self.train_model.users.reset_index()
        self.blogs = self.train_model.blogs.reset_index()
        self.views_pivot = self.train_model.views_pivot.reset_index()
    
    
    def popularity_model(self):
        
        return self.train_model.predict_popularityModel()["title"].to_dict()
        # blogs.loc[pm.recommend("")["movieId"][:5].values]["title"].to_dict()
    
    def SVD_model(self, user):
        
        return self.train_model.predict_estimated_Ratings(user)['title'].to_dict()
    
    def content_filtering(self, item_id):
        
        return self.train_model.content_predict(item_id)['title'].to_dict()
    
    def predict_model(self, blogId, userId):

        userIdx = self.views_pivot[self.views_pivot['userid'] == userId].index

        try:
            userIdx = userIdx[0]
            item_SVD = self.SVD_model(userIdx)
        except:
            item_SVD = None

        blogIdx = self.blogs[self.blogs["blogid"] == blogId].index[0]

        return {"item_SVD":item_SVD, "item_popular":self.popularity_model(), "item_similar":self.content_filtering(blogIdx)}

    def userList_model(self):

        return self.train_model.users[["username"]].to_dict()
