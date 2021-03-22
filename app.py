# aiml code also imported
import os
from flask import Flask
from flask import redirect
from flask import request
from flask import url_for
from flask import session
from flask import render_template
from flask_sqlalchemy import SQLAlchemy
from tensorflow.python.framework import ops
from main import clean_up_sentence
from main import bag_of_words
from main import aiml
import pickle
import json
import random
import tensorflow as tf
import tflearn
import numpy as np
import codecs
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

# flask initialisation
app = Flask(__name__)
app.secret_key = "123"

# chat list to display in uiux end
chat_list = []

@app.route("/", methods=["GET", "POST"])
def default():
    return redirect(url_for("index"))

# display and process the chat endpoints
@app.route("/index", methods=["GET", "POST"])
def index():
    # checking if post request is done
    if request.form:
        query = request.form["query"]
        chat_list.append(query)
        chat_list.append(aiml(query))
        session["chat"] = chat_list
        return render_template("index.html", chat=session["chat"], len=len(session["chat"]))
    else:
        return render_template("index.html", chat=[])

# app is running in debug mode
if __name__ == "__main__":
    app.run(debug=True)