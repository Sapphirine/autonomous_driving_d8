import flask
import os
import cv2
import re
import numpy as np
import glob
import random
import testfunction
import lane_detection
from PIL import Image
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
#import matplotlib
#import matplotlib.pyplot as plt
from flask import Flask, render_template , url_for ,request
from werkzeug.utils import secure_filename

#initializing app
app = Flask(__name__, static_url_path='/static')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
#creating dictionary for basename
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'static/Inpainting-Simple')
#UPLOAD_FOLDER = os.path.basename("static/Inpainting-Simple")
app.config["UPLOAD_FOLDER"]= UPLOAD_FOLDER
#DISPLAY_FOLDER = os.path.join(APP_ROOT, 'static/tampered')
#UPLOAD_FOLDER = os.path.basename("static/Inpainting-Simple")
#app.config["DISPLAY_FOLDER"]= DISPLAY_FOLDER
#def function of inpainting
@app.route("/",methods=["GET","POST"])
def home_page():
	print("hey")
	return render_template("home.html")

#for the first upload page
@app.route("/upload",methods=["GET","POST"])
def upload_file():
	print("hey par2")
	return render_template("index.html")


@app.route("/getLane",methods=["GET","POST"])
def post_feathering():
	#print("Hey posting the advanced lane image")
	file = request.files["im"]
	file.filename = "input.jpg"
	file.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename)))
	dst = testfunction.detect_lanes(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename)))
	return render_template("getLane.html", image_name="../static/lane/output.jpg")




if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=2908)	
