import cv2

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
genderList=['Male','Female']

faceNet=cv2.dnn.readNet('face_detector\opencv_face_detector_uint8.pb','face_detector\opencv_face_detector.pbtxt')
genderNet=cv2.dnn.readNet('gender_detector\gender_net.caffemodel','gender_detector\gender_deploy.prototxt')

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn,faceBoxes

def getGenderPrediction(args_path):
    video = cv2.VideoCapture(args_path)
    padding = 20
    
    hasFrame,frame = video.read()

    resultImg,faceBoxes = highlightFace(faceNet,frame)

    for faceBox in faceBoxes:
        face = frame[max(0,faceBox[1]-padding):
                min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                :min(faceBox[2]+padding, frame.shape[1]-1)]

        blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        return gender