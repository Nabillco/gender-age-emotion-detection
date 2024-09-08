import cv2
def faceBox(faceNet,frame):
    frameheight=frame.shape[0]
    framewidth=frame.shape[1]
    blob=cv2.dnn.blobFromImage(frame,1.0,(227,227),[104,117,123],swapRB=False)
    faceNet.setInput(blob)
    detection=faceNet.forward()
    bboxs=[]
    for i in range(detection.shape[2]):
        confidence=detection[0,0,i,2]
        if(confidence>0.7):
            x1=int(detection[0,0,i,3]*framewidth)
            y1=int(detection[0,0,i,4]*frameheight)
            x2=int(detection[0,0,i,5]*framewidth)
            y2=int(detection[0,0,i,6]*frameheight)
            bboxs.append([x1,y1,x2,y2])
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),1)
    return frame,bboxs
faceproto="opencv_face_detector.pbtxt"
faceModel="opencv_face_detector_uint8.pb"
ageproto="age_deploy.prototxt"
ageModel="age_net.caffemodel"
genderproto="gender_deploy.prototxt"
genderModel="gender_net.caffemodel"
    
emotionModel = "emotion-ferplus-8.onnx"
    
    
faceNet=cv2.dnn.readNet(faceModel,faceproto)
ageNet=cv2.dnn.readNet(ageModel,ageproto)
genderNet=cv2.dnn.readNet(genderModel,genderproto)
    
emotionNet = cv2.dnn.readNet(emotionModel)
    
    
MODEL_MEAN_VALUES=(78.4263377603,87.7689143744,114.895847746)
ageList=['(0-2)','(4-6)','(8-12)','(15-20)','(25-32)','(38-43)','(48-53)','(60+)']
genderList=['Male','Female']
    
emotionList = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Anger', 'Disgust', 'Fear', 'Contempt']
    
Video=cv2.VideoCapture(0)
padding=20
while True:
    ret,frame=Video.read()
    frame,bboxs=faceBox(faceNet,frame)
    for bbox in bboxs:
        #face=frame[bbox[1]:bbox[3],bbox[0]:bbox[2]]
        face=frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding,frame.shape[1]-1)]
        blob=cv2.dnn.blobFromImage(face,1.0,(227,227),MODEL_MEAN_VALUES,swapRB=False)
        genderNet.setInput(blob)
        genderpred=genderNet.forward()
        gender=genderList[genderpred[0].argmax()]
    
        ageNet.setInput(blob)
        agepred=ageNet.forward()
        age=ageList[agepred[0].argmax()]
    
        emotion_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        emotion_face = cv2.resize(emotion_face, (64, 64))
    
        emotion_blob = cv2.dnn.blobFromImage(emotion_face, scalefactor=1.0 / 255, size=(64, 64),mean=(0, 0, 0), swapRB=False)
        emotionNet.setInput(emotion_blob)
        emotionpred = emotionNet.forward()
        emotion = emotionList[emotionpred[0].argmax()]
    
    
        label = "{}, {}, {}".format(gender, age, emotion)
        cv2.rectangle(frame,(bbox[0],bbox[1]-30),(bbox[2],bbox[1]),(0,255,0),-1)
        cv2.putText(frame,label,(bbox[0],bbox[1]-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),2,cv2.LINE_AA)
    cv2.imshow("Handetectak",frame)
    k=cv2.waitKey(1)
    if(k==ord('q')):
        break
Video.release()
cv2.destroyAllWindows()
