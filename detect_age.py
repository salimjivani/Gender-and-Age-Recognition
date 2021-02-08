# import libraries and packages
import numpy as np
import argparse
import cv2
import os
import detect_gender

path = 'Aligned_Images/Image_Prediction.jpg'

def detectAgeGender():

	# define the list of age bracket
	AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)",
		"(38-43)", "(48-53)", "(60-100)"]

	# load haar cascade and fasce detection model
	print("[INFO] loading face detector model...")
	prototxtPath = os.path.sep.join(['face_detector', "deploy.prototxt"])
	weightsPath = os.path.sep.join(['face_detector',"res10_300x300_ssd_iter_140000.caffemodel"])
	faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

	# load age detection model
	print("[INFO] loading age detector model...")
	prototxtPath = os.path.sep.join(['age_detector', "age_deploy.prototxt"])
	weightsPath = os.path.sep.join(['age_detector', "age_net.caffemodel"])
	ageNet = cv2.dnn.readNet(prototxtPath, weightsPath)

	# creat BLOB from input image
	image = cv2.imread(path)
	(h, w) = image.shape[:2]
	blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# pass the blob through face detection model
	print("[INFO] computing face detections...")
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# loop over the detections until certain confidence is reached
	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]

		#filter confidence until 50% confidence is reached
		if confidence > 0.5:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# extract Region Of Interrest from the picture
			face = image[startY:endY, startX:endX]
			faceBlob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
				(78.4263377603, 87.7689143744, 114.895847746),
				swapRB=False)

			# make age prediction using highest confidence
			ageNet.setInput(faceBlob)
			preds = ageNet.forward()
			i = preds[0].argmax()
			age = AGE_BUCKETS[i]
			ageConfidence = preds[0][i]		

			# display age estimation
			text = "{}: {:.2f}%".format(age, ageConfidence * 100)		
			print("[INFO] {}".format(text))

			# display gender prediction
			gender  = detect_gender.getGenderPrediction(path)
			print(f'Gender: {gender}')


			# draw bound box around face and print age with comfidence percentage
			y = startY - 10 if startY - 10 > 10 else startY + 10
			cv2.rectangle(image, (startX, startY), (endX, endY),
				(0, 0, 255), 2)
			cv2.putText(image, f'{text}, {gender}', (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

	cv2.imwrite('unprocessed_images_output/0047UP.jpg', image)

	return text,gender