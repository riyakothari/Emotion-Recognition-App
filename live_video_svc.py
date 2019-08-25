# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 08:37:13 2017

@author: hp
"""

"""Moving forward... trying to make a graph of the probabilities"""

import cv2, glob, random, math, numpy as np, dlib, itertools, copy
from cv2 import WINDOW_NORMAL
from sklearn.externals import joblib
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt
#from sklearn.svm import SVC
from PIL import Image

#Set up some required objects
#video_capture = cv2.VideoCapture(0) #Webcam object
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
detector = dlib.get_frontal_face_detector()		#loading the detector model
predictor = dlib.shape_predictor("C:\Users\hp\Desktop\shape_predictor_68_face_landmarks.dat")  	#loading the predictor model
faceCascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')			#loading the face detector
listGlobal = []

def image_as_nparray(image):		#Converts PIL's Image to numpy's array.
    #print "in image as nparray"
    return np.asarray(image)


def nparray_as_image(nparray, mode='RGB'):		#Converts numpy's array of image to PIL's Image.
    #print "array as image"
    return Image.fromarray(np.asarray(np.clip(nparray, 0, 255), dtype='uint8'), mode)


def load_image(source_path):		#Loads RGB image and converts it to grayscale.
    #print "source image"
    source_image = cv2.imread(source_path)
    return cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)

def _normalize_face(face):			#Normalize the color and size of the face detected
    #print "normalise face"
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = clahe.apply(face)
    face = cv2.resize(face, (350, 350))
    return face;

def _locate_faces(image):			#Locate the faces in the image frame using the cascade face detector
    #print "locate faces"
    faces = faceCascade.detectMultiScale(
        image,
        scaleFactor=1.1,
        minNeighbors=15,
        minSize=(70, 70)
    )

    return faces  # list of (x, y, w, h)    

def find_faces(image):				#Find and normalize the faces detected in the locate_face function
    #print "find faces"
    faces_coordinates = _locate_faces(image)
    cutted_faces = [image[y:y + h, x:x + w] for (x, y, w, h) in faces_coordinates]
    normalized_faces = [_normalize_face(face) for face in cutted_faces]
    return zip(normalized_faces, faces_coordinates)


def draw_with_alpha(source_image, image_to_draw, coordinates):			#Draws a partially transparent image over another image.
    #print "draw with alpha"
    x, y, w, h = coordinates
    #print "x ",x," y ",y," w ",w," h ",h
    image_to_draw = image_to_draw.resize((h, w), Image.ANTIALIAS)
    image_array = image_as_nparray(image_to_draw)
    for c in range(0, 3):
        source_image[y:y + h, x:x + w, c] = image_array[:, :, c] * (image_array[:, :, 3] / 255.0) \
                                            + source_image[y:y + h, x:x + w, c] * (1.0 - image_array[:, :, 3] / 255.0)

# def draw_with_beta(source_image, image_to_draw, coordinates):
    # #print "draw with beta"
    # x, y, w, h = coordinates
    # print "ye le le ele ele le le le le lere"
    # print "x ",x," y ",y," w ",w," h ",h
    
    # image_to_draw = image_to_draw.resize((h, w), Image.ANTIALIAS)
    # image_array = image_as_nparray(image_to_draw)
    # for c in range(0, 3):
        # source_image[y:y + h, x:x + w, c] = image_array[:, :, c] * (image_array[:, :, 3] / 255.0) \
                                            # + source_image[y:y + h, x:x + w, c] * (1.0 - image_array[:, :, 3] / 255.0)


def _load_emoticons(emotions):			#Loads emotions images from graphics folder.
    #print "load emotions"
    return [nparray_as_image(cv2.imread('graphics/%s.png' % emotion, -1), mode=None) for emotion in emotions]


def show_webcam_and_run(model, emoticons, window_size=None, window_name='webcam', update_time=16):		
		#Shows webcam image, detects faces and its emotions
		#in real time and draw emoticons over those faces.

    emo = [0,0,0,0,0,0]
    try:
        
        cv2.namedWindow(window_name, WINDOW_NORMAL)
        if window_size:
            width, height = window_size
            cv2.resizeWindow(window_name, width, height)
    
        vc = cv2.VideoCapture(0)
        #print "Video Capture works"
        if vc.isOpened():
            read_value, webcam_image = vc.read()
            #print "vc is opened"
        else:
            print("webcam not found")
            return
        cnt = {}			#A dictionary for storing the image record of a face over time
        cnt[0]=[0,0,0,0,0,0]
        while read_value:
            counter =0
            for normalized_face, (x, y, w, h) in find_faces(webcam_image):
                #print "Length is ",len(normalized_face)
                #print normalized_face
                
                #print "Counter is ",counter
                landmarks_vectorised = get_landmarks(normalized_face)
                npar_pred = np.array(landmarks_vectorised)
                #prediction1 = model.predict_proba(npar_pred)  # do prediction
                prediction2 = model.predict(npar_pred)
                            
    #            if cv2.__version__ != '3.1.0':
    #                prediction1 = prediction1[0]
    
                #print prediction2
                #print "new line"
                image_to_draw = emoticons[prediction2]
                if(prediction2 != 0):
                    cnt[counter][prediction2] = cnt[counter][prediction2]+1
                    #emo[prediction2] = emo[prediction2]+1
                #print"Emotion increase in",prediction2-1
                counter = counter+1
                cnt[counter] = [0,0,0,0,0,0]
                
                #font = cv2.FONT_HERSHEY_SIMPLEX
                #cv2.putText(vc,'Face number',(x,y), font, 4,(255,255,255),2,cv2.LINE_AA)
#                if((x-100 <0) or (y-100 < 0)):
#                    draw_with_alpha(webcam_image, image_to_draw, (x+50, y+50, w, h))
#                else:
#                    draw_with_alpha(webcam_image, image_to_draw, (x+30, y+30, w-40, h-40))
                draw_with_alpha(webcam_image, image_to_draw, (x-50, y-50, w-60, h-60))
                #draw_with_beta(webcam_image, image_to_draw, (x,y,w,h))
            cv2.imshow(window_name, webcam_image)
            read_value, webcam_image = vc.read()
            key = cv2.waitKey(update_time)
            
            if key == 27:  # exit on ESC
                print cnt
                #print" Your emotions were: "
                #print emo
                #print "You were mostly "
                maxi = emotions[calculate_max(emo)]
                #print maxi
#                showGraph(emo)
                showGraph2(cnt)
                break
    
        cv2.destroyWindow(window_name)
    except ValueError, ex:
        print "Error msg, ", str(ex)
        cv2.destroyWindow(window_name)


#def showGraph(emotionVal):
#    emotions2 = ('neutral','anger', 'disgust', 'happy', 'sadness', 'surprise')
#    y_pos = np.arange(len(emotions))
#    plt.bar(y_pos, emotionVal, align='center', alpha=0.5)
#    plt.xticks(y_pos, emotions2)
#    plt.ylabel('emotion')
#    plt.title('Showing graphically')
#    plt.show()
#    print"Graph plotted successfully"
#    return

def showGraph2(counter):				#To draw the graph of count of each emotion
    #emotions2 = ('neutral','anger', 'disgust', 'happy', 'sadness', 'surprise')
    n_groups = 6
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.8
    rect = []
    #print len(counter)
    color1 = ['b','r','g','y']
    for i in range(len(counter)):
        #print "it is i: ",i
        
        rect = plt.bar(index, counter[i], bar_width, 
                  alpha=opacity, color=color1[i],label='face')

    plt.xlabel('Emotion')
    plt.ylabel('Scores')
    plt.title('Emotions by person')
    plt.xticks(index + bar_width, ('neutral','anger', 'disgust', 'happy', 'sadness', 'surprise'))
    plt.legend()
     
    plt.tight_layout()
    plt.show()
#    y_pos = np.arange(len(emotions))
#    plt.bar(y_pos, emotionVal, align='center', alpha=0.5)
#    plt.xticks(y_pos, emotions2)
#    plt.ylabel('emotion')
#    plt.title('Showing graphically')
#    plt.show()
    #print"Graph plotted successfully"
    return


def calculate_max(emotion):				#To calculate the maximum amount of emotion that was detected in a person
    maximum = 0
    maxVal = 0
    for i in range(1,6):
        #print i
        if(emotion[i]>maximum):
            maximum = emotion[i]
            maxVal = i
    
    return maxVal

def get_landmarks(image):			#To get the facial landmarks of the face detected and normalize them
   
    #print "in get_landmarks"
    detections = detector(image, 1)
    # print "image"
    # print image
    # print "Detected list: ",list(detections)
#    for x in list(detections).values():
#        print "sdl"
#        print x
    check_flag=100
    if(len(list(detections))==0):
        print "empty"
        #enu = listGlobal
        check_flag=50
    else:
        print "not empty"
        global listGlobal
        listGlobal = copy.deepcopy(list(detections))
        #enu = enumerate(detections)
#        for x in list(detections).values():
#            if(math.isnan(x)):
        check_flag=100
    #abc = enumerate(detections)
    #print abc
    #try:
    if(check_flag==100):       
        
        for k,d2 in enumerate(detections): #For all detected face instances individually
            #print "in detections"
            shape = predictor(image, d2) #Draw Facial Landmarks with the predictor class
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
            #print "before for"
            for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
                landmarks_vectorised.append(x)
                landmarks_vectorised.append(y)
                meannp = np.asarray((ymean,xmean))
                coornp = np.asarray((z,w))
                dist = np.linalg.norm(coornp-meannp)
                anglerelative = (math.atan((z-ymean)/(w-xmean))*180/math.pi) - anglenose
                landmarks_vectorised.append(dist)
                landmarks_vectorised.append(anglerelative)
            #print "after for"
    
    #    except ValueError, ex:
    #        print "Error it is", str(ex)
            
    
            if len(detections) < 1: 
                landmarks_vectorised = "error"
            return landmarks_vectorised
    else:
        print "The input contained NaNnananananananananaanananananananananananananananananananananananana"
        for k,d2 in enumerate(listGlobal): #For all detected face instances individually
            #print "in detections nan"
            shape = predictor(image, d2) #Draw Facial Landmarks with the predictor class
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
            #print "before for nan"
            for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
                landmarks_vectorised.append(x)
                landmarks_vectorised.append(y)
                meannp = np.asarray((ymean,xmean))
                coornp = np.asarray((z,w))
                dist = np.linalg.norm(coornp-meannp)
                anglerelative = (math.atan((z-ymean)/(w-xmean))*180/math.pi) - anglenose
                landmarks_vectorised.append(dist)
                landmarks_vectorised.append(anglerelative)
            #print "after for nan"
    
    #    except ValueError, ex:
    #        print "Error it is", str(ex)
            
    
            if len(listGlobal) < 1: 
                landmarks_vectorised = "error"
            return landmarks_vectorised
		
		
if __name__ == '__main__':
    emotions = ['neutral', 'anger', 'disgust', 'happy', 'sadness', 'surprise']
    emoticons = _load_emoticons(emotions)
    clf = joblib.load("D:/myStuff/final_year_project/newCode/results_6emo.pkl") 
    window_name = 'WEBCAM (press ESC to exit)'
    show_webcam_and_run(clf, emoticons, window_size=(800, 600), window_name=window_name, update_time=16)
