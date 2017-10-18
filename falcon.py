import numpy as np 
import dlib
import Image
import cv2 
import matplotlib.pyplot as plt 
import os 

PATH = os.getcwd()

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

for images in images_dir:
	img_list = (os.listdir(images_path +  '/' + images))
	for img in img_list: 
		image = cv2.imread(images_path + '/' + images + '/' + img)
		print image
		detected_faces = detect_faces(image)

		for face in detected_faces:
			detected = cv2.rectangle(image, (face[0],face[1]), (face[2],face[3]), (255,0,0),2)

		try:
			crop_img = image[face[1]:face[3],face[0]:face[2]]
			crop_img = cv2.resize(crop_img,(1200,1200))
			print "Detected"
			data_img_list.append(crop_img)
			cv2.imwrite("kaggle_images/female_images/img" + str(i) + '.jpg', crop_img)
			i = i + 1
			success = success + 1
			# else:
			print("cancelled")
			failed = failed + 1 
		
		except Exception as e: 
			print "Face Detection Unsuccessful"
			failed = failed + 1
print "successful: ",success
print "failed: ", failed

np.save('numpy/img.npy', data_img_list)

'''
	You've to change the directory manually
'''
dh_d = input("Do you Want to rename the images (1/0)?")

if dh_d == 1:
	dhd_a = input("Have you finalized your images? (1/0)")
	if dhd_a == 1 :
		dhh_d = input("Did you make a folder named 'female_images_listed'? (1/0)")
		if dhh_d == 1:
			#takes only 0.5 sec! Sick!!
			images_directory = os.listdir('kaggle_images/female_images_listed')
			for images in images_directory:
				old = os.path.join('kaggle_images/female_images_listed', images)
				new = os.path.join('kaggle_images/female_images_listed_proper', 'female-' + str(i) + '.jpg')
				print old
				os.rename(old, new)
				i = i + 1
else:
	print("No? Okay.")