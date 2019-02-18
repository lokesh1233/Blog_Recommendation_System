from flask import Flask, render_template, request
import json
from bson.json_util import dumps
import threading


app=Flask(__name__,  template_folder='template')
@app.route("/")
def home():
    return render_template("Blog.html")



## recommend with popular and user similarities with SVD
from models.predict_model import  blog_predict_model
blog_recommendation = blog_predict_model()
# blog_recommendation = ''

#
# def printit():
#     blog_recommendation = blog_predict_model()
#     threading.Timer(600.0, printit).start()
#     print("Hello, World!")
#
# printit()

#Blog recommendation using popular data and user similar with svd
@app.route('/recommend_popular_SVD/<movieId>/<userId>',methods = ['GET'])
def recommend_popular_userSVD_df(movieId, userId):
    if request.method == 'GET':
        rec_movies = blog_recommendation.predict_model(movieId, userId)
        # rec_users = recommendMovies.predict_SVD(jsonData["user"])

        return dumps(rec_movies)
    else:
        return ''


#Blog recommendation displaying user list and updating user list data
@app.route('/userListSet',methods = ['GET', "POST"])
def userListData():
    if request.method == 'GET':
        rec_movies = blog_recommendation.userList_model()
        # rec_users = recommendMovies.predict_SVD(jsonData["user"])

        return dumps(rec_movies)
    elif  request.method == 'POST':
        rec_movies = blog_recommendation.train_model.mongodb_interface.insert_structure(json.loads(request.data), "users_db")
        # rec_users = recommendMovies.predict_SVD(jsonData["user"])

        return dumps(rec_movies)
    else:
        return ''



#Blog recommendation updating blogs list data
@app.route('/blogsListSet',methods = ['POST'])
def blogsListData():
    if request.method == 'POST':
        rec_movies = blog_recommendation.train_model.mongodb_interface.insert_structure(json.loads(request.data), "blogs_db")
        # rec_users = recommendMovies.predict_SVD(jsonData["user"])

        return dumps(rec_movies)
    else:
        return ''
    


if __name__ =="__main__":
    app.run(debug=True, host='0.0.0.0')