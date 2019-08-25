# -*- coding: utf-8 -*-
"""
Created on Mon Mar 06 13:48:28 2017

@author: hp
"""
"""Classification.... Using the models to predict the emotion in STATIC"""
import cv2, glob, random, math, numpy as np, dlib, itertools
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.externals import joblib
emotions = ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sadness", "surprise"] #Emotion list
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:\Users\hp\Desktop\shape_predictor_68_face_landmarks.dat") #Or set this to whatever you named the downloaded file
#clf = SVC(kernel='linear', probability=True, tol=1e-3)

def get_landmarks(image):
    detections = detector(image, 1)
    for k,d in enumerate(detections): #For all detected face instances individually
        shape = predictor(image, d) #Draw Facial Landmarks with the predictor class
        xlist = []
        ylist = []
        for i in range(1,68): #Store X and Y coordinates in two lists
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))
            
        xmean = np.mean(xlist) #Get the mean of both axes to determine centre of gravity
        ymean = np.mean(ylist)
        xcentral = [(x-xmean) for x in xlist] #get distance between each point and the central point in both axes
        ycentral = [(y-ymean) for y in ylist]

        if xlist[26] == xlist[29]: #If x-coordinates of the set are the same, the angle is 0, catch to prevent 'divide by 0' error in function
            anglenose = 0
        else:
            anglenose = int(math.atan((ylist[26]-ylist[29])/(xlist[26]-xlist[29]))*180/math.pi)

        if anglenose < 0:
            anglenose += 90
        else:
            anglenose -= 90

        landmarks_vectorised = []
        for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
            landmarks_vectorised.append(x)
            landmarks_vectorised.append(y)
            meannp = np.asarray((ymean,xmean))
            coornp = np.asarray((z,w))
            dist = np.linalg.norm(coornp-meannp)
            anglerelative = (math.atan((z-ymean)/(w-xmean))*180/math.pi) - anglenose
            landmarks_vectorised.append(dist)
            landmarks_vectorised.append(anglerelative)

    if len(detections) < 1: 
        landmarks_vectorised = "error"
    return landmarks_vectorised

if __name__ == '__main__':
    prediction_data = []
    prediction_labels = []
    clf = joblib.load("D:/myStuff/final_year_project/newCode/results.pkl") 
    #clf2 = joblib.load("D:/myStuff/final_year_project/newCode/results2.pkl") 
    #clf3 = joblib.load("D:/myStuff/final_year_project/newCode/results3.pkl")
    image = cv2.imread("D:/myStuff/final_year_project/ladki/fear.jpg")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe_image = clahe.apply(gray)
    landmarks_vectorised = get_landmarks(clahe_image)
    if landmarks_vectorised == "error":
        pass
    else:
        prediction_data.append(landmarks_vectorised)
#        prediction_labels.append(emotions.index("anger"))
#        prediction_labels.append(emotions.index("contempt"))
#        prediction_labels.append(emotions.index("disgust"))
#        prediction_labels.append(emotions.index("fear"))
#        prediction_labels.append(emotions.index("happy"))
#        prediction_labels.append(emotions.index("neutral"))
#        prediction_labels.append(emotions.index("sadness"))
        prediction_labels.append(emotions.index("surprise"))
    npar_pred = np.array(landmarks_vectorised)
    pred_lin = clf.score(npar_pred, prediction_labels)
    print pred_lin
    print "For linear SVM: "
    print "Prediction Probabilities"
    print clf.predict_proba(npar_pred)
    print "Predicted Emotion"
    
    print emotions[clf.predict(npar_pred)]
    
    
    print "For Neural Networks: "
    print "Prediction Probabilities"
    #print clf2.predict_proba(npar_pred)
    print "Predicted Emotion"
    #print emotions[clf2.predict(npar_pred)]
#    print pred_lin
#    print clf2.predict(npar_pred)
#    print clf3.predict(npar_pred)