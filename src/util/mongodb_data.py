from pymongo import MongoClient
from bson.json_util import dumps
import re
from datetime import datetime
import pandas as pd
import numpy as np

class mongodb_interface():
    def __init__(self):
        
        self._host="localhost"
        self._port=27017
        self._username = None
        self._password = None
        self._dbname = "Blog_Recommendation"
        
        #db connection to mongo
        self._db = self._connect_mongo(host=self._host, port=self._port, username=self._username, password=self._password, db=self._dbname)
        
        self.users_db = self._db.Users
        self.views_db = self._db.Views
        self.blogs_db = self._db.Blogs
        self.content_db = self._db.content_similarity
    
    
    
    def _connect_mongo(self, host='localhost', port=27017, username=None, password=None, db="Blog_Recommendation"):
        """ A util for making a connection to mongo """
    
        if username and password:
            mongo_uri = 'mongodb://%s:%s@%s:%s/%s' % (username, password, host, port, db)
            conn = MongoClient(mongo_uri)
        else:
            conn = MongoClient(host, port)


        return conn[db]


    def read_mongo(self, collection, query={}, selection=None, no_id=True, index=None):
        """ Read from Mongo and Store into DataFrame """
    
        # Make a query to the specific DB and Collection
        cursor = collection.find(query, selection)

        # Expand the cursor and construct the DataFrame
        df =  pd.DataFrame(list(cursor))

        if index != None:
            df = df.set_index(index)

        # Delete the _id
        if no_id:
            del df['_id']

        return df
    
    def insert_mongo(self, collection, data={}, structure=()):
        """ insert from store DataFrame into mongo """
        
        if type(data) == dict:
            #insert to db with one record
            return collection.insert_one(structure(data))
        elif type(data) == list:
            data_many = []
            for blg_data in data:
                data_many.append(structure(blg_data))
        
            return collection.insert_many(data_many)
        else:
            return "check structure and data type fields with keys"
    
    def update_mongo(self):
        print()
        

class mongodb_import(mongodb_interface):
    def __init__(self):
        self.super_mongodb_interface = mongodb_interface()

        print()


    def _get(self, id_data):

        if id_data == "users_db":
            return self.super_mongodb_interface.users_db,  self.dict_users
        elif id_data == "blogs_db":
            return self.super_mongodb_interface.blogs_db,  self.dict_blogs
        elif id_data == "views_db":
            return self.super_mongodb_interface.views_db, self.dict_views
        else:
            return ''






    #users data structure to mongo
    def dict_users(self, data):
        return {
            "userid":data.get("userid", ""),
            "username":data.get("username", ""),
           "type": data.get("type", ""),
            "bio": data.get("bio", ""),
#         "interest"
            "createdat": data.get("createdat", datetime.now())
        }

    #blogs data structure to mongo
    def dict_blogs(self, data):
        return {
            "blogid":data.get("blogid", ""),
            "title":data.get("title", ""),
            "subtitle": data.get("subtitle", ""),
            "content": data.get("content", ""),
            "createrid": data.get("createrid", ""),
            "tags": data.get("tags", ""),
            "createrdate": data.get("createrdate", datetime.now())
        }

    #views data structure to mongo
    def dict_views(self, data):
        return {
            "blogid":data.get("blogid", ""),
            "userid":data.get("userid", ""),
            "viewedcount": data.get("viewedcount", ""),
            "lastviewed": data.get("lastviewed", datetime.now())
        }
    
    
    def insert_structure(self, data, collection):

        collection, structure = self._get(collection)

        insert_ststus = self.insert_mongo(collection=collection, data=data, structure=structure)

        try:
            if insert_ststus.acknowledged:
                return "updated successfully"
            else:
                return "parsing xml error"
        except:
            return insert_ststus
    
    def query_insert_views(self, data):
        return ({
            "blogid": data["blogid"],
            "userid": data["userid"]
            }, {
            "$inc": { 
                "seq": int(data["viewedcount"]) 
                },
            "$set": {
                "lastviewed":datetime.now()
            }})

    
    def insert_views_distinct(self, data):
        if type(data) == dict:
            query = self.query_insert_views(data)
            return self.views_db.find_one_and_update(query[0], query[1])
        elif type(data) == list:
            data_many = []
            for blg_blogs in data:
                query = self.query_insert_views(blg_blogs)
                self.views_db.find_one_and_update(query[0], query[1])
#             data_many.append(dict_blogs(blg_blogs))
#         blogs_db.insert_many(data_many)
            return 'updated'
        else:
            return "check structure and data type fields with keys"
    
#     def insert_users(self, data):
        
#         return self.insert_mongo(collection=self.users_db, data=data, structure=self.dict_users)
        
# #         if type(data) == dict:
# #             return users_db.insert_one(dict_users(data))
# #         elif type(data) == list:
# #             data_many = []
# #             for blg_user in data:
# #                 data_many.append(dict_users(blg_user))
        
# #             return users_db.insert_many(data_many)
# #         else:
# #             return "check structure and data type fields with keys"


#     def insert_views(self, data):
        
#         return self.insert_mongo(collection=self.views_db, data=data, structure=self.dict_views)
        
# #         if type(data) == dict:
# #             return views_db.insert_one(dict_views(data))
# #         elif type(data) == list:
# #             data_many = []
# #             for blg_views in data:
# #                 data_many.append(dict_views(blg_views))
        
# #             return views_db.insert_many(data_many)
# #         else:
# #             return "check structure and data type fields with keys"


#     def insert_blogs(self, data):
        
#         return self.insert_mongo(collection=self.blogs_db, data=data, structure=self.dict_blogs)
        
# #         if type(data) == dict:
# #             return blogs_db.insert_one(dict_blogs(data))
# #         elif type(data) == list:
# #             data_many = []
# #             for blg_blogs in data:
# #                 data_many.append(dict_blogs(blg_blogs))
        
# #             return blogs_db.insert_many(data_many)
# #         else:
# #             return "check structure and data type fields with keys"




# class mongo_export(mongodb_interface):
#     def __init__(self):
#         print()
    

# insert from the pandas dataframe to mongodb
# insert_blogs(list(blogs.T.to_dict().values()))
# insert_users(list(users.T.to_dict().values()))
# insert_views(list(viewes.T.to_dict().values()))
        