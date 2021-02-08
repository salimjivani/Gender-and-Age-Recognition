# import libraries and packages
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os

def detect_and_predict_age(frame, faceNet, ageNet, minConf=0.5):
	# define the list of age bucket that will be predicted
	AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)",
		"(38-43)", "(48-53)", "(60-100)"]

	# define the list of gender bucket
	GENDER_BUCKETS = ['Male','Female']

	results = []

	# construct BLOB to feed to model
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# pass BLOB thrpugh face detection model
	faceNet.setInput(blob)
	detections = faceNet.forward()

	for i in range(0, detections.shape[2]):
		# extract confidence of probability
		confidence = detections[0, 0, i, 2]

		# filter out weak confidence for prediction
		if confidence > minConf:
			# compute coordinates of bounding box
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# extract the Region of Interest
			face = frame[startY:endY, startX:endX]

			if face.shape[0] < 20 or face.shape[1] < 20:
				continue

			# construct a blob from face ROI
			faceBlob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
				(78.4263377603, 87.7689143744, 114.895847746),
				swapRB=False)

			# make predictions on the age and find the age bucket with highest confidence percent
			ageNet.setInput(faceBlob)
			preds = ageNet.forward()
			i = preds[0].argmax()
			age = AGE_BUCKETS[i]
			ageConfidence = preds[0][i]

			# make predictions on the gender and find the gender bucket
			genderNet.setInput(faceBlob)
			genderPreds = genderNet.forward()
			j = genderPreds[0].argmax()
			gender = GENDER_BUCKETS[j]

			# construct a dictionary to save prediction
			d = {
				"loc": (startX, startY, endX, endY),
				"age": (age, ageConfidence),
				"gender": (gender)
			}
			results.append(d)

	# return our results to the calling function
	return results

# construct the argument parse and parse the arguments ffor passing pictures through terminal
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", required=True,
	help="path to face detector model directory")
ap.add_argument("-a", "--age", required=True,
	help="path to age detector model directory")
ap.add_argument("-g", "--gender", required=True,
	help="path to gender detector model directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load face detector
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load our age detector
print("[INFO] loading age detector model...")
prototxtPath = os.path.sep.join([args["age"], "age_deploy.prototxt"])
weightsPath = os.path.sep.join([args["age"], "age_net.caffemodel"])
ageNet = cv2.dnn.readNet(prototxtPath, weightsPath)

#load gender detector
print("[INFO] loading gender detector model...")
prototxtPath = os.path.sep.join([args["gender"], "gender_deploy.prototxt"])
weightsPath = os.path.sep.join([args["gender"], "gender_net.caffemodel"])
genderNet=cv2.dnn.readNet(prototxtPath,weightsPath)

# initialize the video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# detect faces in the frame, and for each face in the frame, and predict the age
	results = detect_and_predict_age(frame, faceNet, ageNet,
		minConf=args["confidence"])

	# loop over the results
	for r in results:
		# draw the bounding box with age prediction
		text = "{}: {:.2f}%".format(r["age"][0], r["age"][1] * 100)
		(startX, startY, endX, endY) = r["loc"]
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(frame, (startX, startY), (endX, endY),
			(0, 0, 255), 2)
		cv2.putText(frame, f'{text},{r["gender"]}', (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

cv2.destroyAllWindows()
vs.stop()