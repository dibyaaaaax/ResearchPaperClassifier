from flask import Flask,render_template
from Classifier.predictor import run
from flask import Flask, request, redirect, url_for, render_template, send_from_directory, after_this_request
from flask import json
from flask import jsonify
app = Flask(__name__)
@app.route('/')
def home():
    return render_template("index.html")

@app.route("/handler",methods=['POST'])
def handler():

	if request.method == 'POST':
		query = request.get_json()
		print("Query: ",query["aid"])
		
		ans = run(query["aid"])
		print("Answer: ",ans)
	return json.jsonify({ 
        'ans': ans 
    }) 	

if __name__ == '__main__':
    app.run(debug=True)