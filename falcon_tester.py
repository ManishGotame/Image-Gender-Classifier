#wget -O patch.exe "https://fileshare5010.depositfiles.com/auth-15079514569ca485e6a041223947bac1-27.34.22.182-266757333-180071176-guest/FS501-1/BO2-MP-ZM_LAN-patch_v3.exe" -c
import numpy as np 
import dlib
import Image
import cv2 
import matplotlib.pyplot as plt 
import os 

PATH = os.getcwd()

eye_cascade = cv2.CascadeClassifier('face_detection/haarcascade_eye.xml')

def detect_faces(image):
	# loads the detector from the library of dlib 
    face_detector = dlib.get_frontal_face_detector()
    # Run detector and get bounding boxes of the faces on image.
    detected_faces = face_detector(image, 1)
    face_frames = [(x.left(), x.top(), x.right(), x.bottom()) for x in detected_faces]

    return face_frames

images_path = PATH + "/kaggle_images/"
images_dir = os.listdir(images_path)
i = 0
success = 0 
failed = 0
data_img_list = []
img_height = 150 
img_width = 150

for images in images_dir:
	print ('Loaded the images of dataset-'+'{}\n'.format(images))
	img_list = (os.listdir(images_path +  '/' + images))
	for img in img_list: 
		loaded_image = cv2.imread(images_path + '/' + images + '/' + img)
		loaded_image = cv2.resize(loaded_image, (img_width,img_height))
		loaded_image = cv2.cvtColor(loaded_image, cv2.COLOR_BGR2GRAY)
		data_img_list.append(loaded_image)

np.save('numpy/m_and_f/data_img_list_150.npy', data_img_list)
print data_img_list

