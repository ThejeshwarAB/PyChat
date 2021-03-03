import os
# from main import *
from flask import Flask
from flask import redirect
from flask import request
from flask import url_for
from flask import session
from flask import render_template
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.secret_key = "123"

@app.route("/", methods=["GET", "POST"])
def default():
    return redirect(url_for("index"))


@app.route("/index", methods=["GET", "POST"])
def index():
	chat_list = ["",""]
	# response = ""
	if request.form:
		print(request.form["query"])

		# chat_list[request.form["query"],response]

		# chat_list=[request.form["query"],request.form["query"]]
	return render_template("index.html",chat=chat_list)

if __name__ == "__main__":
    app.run(debug=True)
