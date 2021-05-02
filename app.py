from flask import Flask, render_template, request,session 
from wtforms import Form, TextAreaField, validators
import pickle
import tensorflow as tf 
import os
import numpy as np
import cv2
from flask_session.__init__ import Session
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
app.config['SESSION_TYPE'] = 'filesystem'
app.secret_key ="super secret key"


######## Preparing the Classifier
cur_dir = os.path.dirname(__file__)
word_index = pickle.load(open(os.path.join(cur_dir,
'pkl_objects', 'word_index.pkl'),
'rb'))
index_word = pickle.load(open(os.path.join(cur_dir,
'pkl_objects', 'index_word.pkl'),
'rb'))
image_features_extract_model =  tf.keras.models.load_model(os.path.join(cur_dir,'image_features_extract_model'))
caption_model =  tf.keras.models.load_model(os.path.join(cur_dir,'model'))


def load_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    
    return img

def generateCaption(img_path):
	maxlen = 61
	max_tokens = 8922
	photo = load_image(img_path)
	photo = tf.reshape(photo,(1,photo.shape[0],photo.shape[1],photo.shape[2]))
	photo = image_features_extract_model(photo)
	photo = tf.reshape(photo,(-1,1,2048))
	# print ("photo {}".format(photo.shape))
	in_text = 'ss'
	for i in range(61):
	# print ('in text : {}'.format(in_text))
		sequence = np.array([word_index[w] for w in in_text.split() if w in word_index])
		sequence = pad_sequences([sequence],maxlen = maxlen,padding='post',truncating='post')
		yhat = caption_model.predict([photo,sequence], verbose=0)
		yhat = np.argmax(yhat.reshape(-1,max_tokens+1),axis=1)
		output = yhat[i]
		if index_word[output] == 'ee':
			break
	# print ("output : {}".format(output))
		in_text  +=  ' ' + index_word[output]

	return "" if in_text == 'ss' else in_text[3:]


import os
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = '/path/to/the/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(cur_dir,'static/image')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
	caption = ""
	if request.method == 'POST':
# check if the post request has the file part
		if 'file' not in request.files:
			flash('No file part')
			return redirect(request.url)
		file = request.files['file']
# if user does not select file, browser also
# submit an empty part without filename
		if file.filename == '':
			flash('No selected file')
			return redirect(request.url)
		if file and allowed_file(file.filename):
			# print ("file : {} ".format(type(file)))
			filename = secure_filename(file.filename)
			direc = os.path.join(app.config['UPLOAD_FOLDER'], filename)
			file.save(direc)
			caption = generateCaption(direc)
			return render_template('imageform.html', caption = caption, filename = filename)
 
	return '''
	<!doctype html>
	<title>Image Captioning</title>
	<h1>Image Captioning</h1>
	<h3>Upload New Image</h3>
	<form method=post enctype=multipart/form-data>
	<input type=file name=file>
	<input type=submit value=Upload>
 
	'''
	


 
if __name__ == '__main__':

	app.debug = True
	app.run()

	