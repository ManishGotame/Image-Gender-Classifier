import numpy as np 
import dlib
import Image
import cv2 
import matplotlib.pyplot as plt 
import os 
import keras
import tnfix
from pathlib import Path
# np.set_printoptions(threshold=np.inf) # it will basically print out all the numpy array. Probably useful in MNIST dataset 
from keras.models import load_model # for loading the trained model's weight
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # this fixes the warning from tensorflow 
'''
	To use concatenate, The image size should be the same like (150,150)
	axis=0 : vertical
	axis=1 : horizontal
'''
#variable 
img_height = 250 # height of the image
img_width = 250 # width of the image
#ends here

#inputs
#weights are loaded at line 98.
#ends here

#training module
#mate! It's still under construction
def Train_model_ML(image,deriv=True): # training the model (beta and ambitious)
	if(deriv==False): # it's for female
		X_train = image 
		print"Final: Female"
		X_train = image 
		y_train = np.array([
				[0,1]
			])

		loaded_model = load_model('falcon_numpy/male_and_female_250Epochs.h5')

		loaded_model.compile(
				loss='binary_crossentropy',
				optimizer='adam',
				metrics=['accuracy']
			)

		loaded_model.fit(X_train, y_train, batch_size=5, epochs=10, verbose=1)
		loaded_model.save('falcon_numpy/male_and_female_250Epochs.h5')

	elif(deriv==True): # it's for male
		print"Final: male"
		X_train = image 
		y_train = np.array([
				[1,0]
			])

		loaded_model = load_model('falcon_numpy/male_and_female_250Epochs.h5')

		loaded_model.compile(
				loss='binary_crossentropy',
				optimizer='adam',
				metrics=['accuracy']
			)

		loaded_model.fit(X_train, y_train, batch_size=5, epochs=10, verbose=1)
		loaded_model.save('falcon_numpy/male_and_female_250Epochs.h5')

#ends here

#image_data build #sex means whether male or Female 
#male = True 
#female = False
def Face_Data_ML(img_data,Sex=True):
	if(Sex==True): # male 
		# print('Data File is present')
		#face_img_matrix is the dataset right now.
		img_data = np.array(img_data)
		new_img_data = []
		new_img_data.append(img_data)
		# print new_img_data
		Train_model_ML(new_img_data,deriv=True)

	elif(Sex==False):
		# print('Data File is present')
		#face_img_matrix is the dataset right now.
		new_img_data = []
		new_img_data.append(img_data)
		# print new_img_data  
		Train_model_ML(new_img_data, deriv=False)

#ends here
img = cv2.imread('images/large_women.jpg') # loads image from a specified directory 
# img = cv2.imread('images/dog.47.jpg')

def detect_face(image): #detection module
	face_detector = dlib.get_frontal_face_detector() # this is a module in dlib 
	detected_face = face_detector(image, 1)
	face_frames = [(x.left(), x.top(), x.right(), x.bottom()) for x in detected_face]
	return face_frames

detected_faces = detect_face(img) #detection module 
for face in detected_faces:
	detected = cv2.rectangle(img, (face[0],face[1]), (face[2],face[3]), (255,0,0),2)

try: # it will try to detect the face but if it can't than it will run the "except" and it won't crash the program
	face_img = img[face[1]:face[3],face[0]:face[2]] #this is where the detected face is  
	# convert it into gray and resize into 150,150 size
	face_img = cv2.resize(face_img, (img_height,img_width)) #convert into 150,150 size
	face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) # convert rgb to gray because the neural
	# net is built to learn from gray images
	face_img_matrix = np.array(face_img) # convert into a numpy array 
	face_img_matrix = face_img_matrix.astype('float32')
	face_img_matrix /=255 # decrese the size
	face_img_matrix = face_img_matrix.reshape(1,img_width,img_height) # (1,150,150)
	face_img_matrix = face_img_matrix.reshape(face_img_matrix.shape[0],1,img_height,img_width)# (1,1,150,150)
	# print(face_img.shape)

	#model prediction module
	loaded_model = load_model('falcon_numpy/male_and_female_250Epochs.h5')
	# loaded_model = load_model('falcon_numpy/male_and_female_improved_300.h5')
	loaded_model.compile(
			loss='binary_crossentropy', # classify between 2 images, Maybe 0 and 1 
			#loss='categorical_crossentropy'#categorical_crossentropy classifies between more than 2 images 
			optimizer='adam', # optimizer, Was supposed to use 'rmsprop' but this seems to work better
			metrics=['accuracy']	
		)
	# print face_img_matrix
	score = ((loaded_model.predict(face_img_matrix)))
	data = (loaded_model.predict_classes(face_img_matrix))
	print("Male","Female")
	print(score) #this is where the prediction accuracy is shown 
	print(data) #this is where the prediction final answer is shown 
	#ends here
	plt.imshow(img)
	plt.show()

	if data == np.array([0]): # male # face_data.npy 
		print "Male"
		correct_ans = input("Is the prediction correct? ")

		if correct_ans == 1: # i am using 1 as true and 0 as false
			Face_Data_ML(face_img_matrix,Sex=True) #male
		elif correct_ans == 0:
			Face_Data_ML(face_img_matrix,Sex=False) #Female 

	elif data == np.array([1]): # female 
		print "Female"
		correct_ans = input("Is the prediction correct? ")
		if correct_ans == 1:
			Face_Data_ML(face_img_matrix,Sex=False) #female
		elif correct_ans == 0:
			Face_Data_ML(face_img_matrix,Sex=True) # male 

except Exception as e: # it will only run if the program can't detect the face.
	print e
	print "Sorry! No Face Detected or an Error has occured"
	print
