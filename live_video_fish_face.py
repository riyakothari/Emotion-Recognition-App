"""
This module is the main module in this package. It loads emotion recognition model from a file,
shows a webcam image, recognizes face and it's emotion and draw emotion on the image.
"""


"""Using the Haarcascade classifier for emotion prediction in live feed"""

import cv2
from cv2 import WINDOW_NORMAL
import numpy as np
from sklearn.externals import joblib
#import cv2
if cv2.__version__ == '3.1.0':
    from PIL import Image
else:
    from PIL import Image
#from face_detect import find_faces
#from image_commons import nparray_as_image, draw_with_alpha
faceCascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
def image_as_nparray(image):
    """
    Converts PIL's Image to numpy's array.
    :param image: PIL's Image object.
    :return: Numpy's array of the image.
    """
    return np.asarray(image)


def nparray_as_image(nparray, mode='RGB'):
    """
    Converts numpy's array of image to PIL's Image.
    :param nparray: Numpy's array of image.
    :param mode: Mode of the conversion. Defaults to 'RGB'.
    :return: PIL's Image containing the image.
    """
    return Image.fromarray(np.asarray(np.clip(nparray, 0, 255), dtype='uint8'), mode)


def load_image(source_path):
    """
    Loads RGB image and converts it to grayscale.
    :param source_path: Image's source path.
    :return: Image loaded from the path and converted to grayscale.
    """
    source_image = cv2.imread(source_path)
    return cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
def _normalize_face(face):
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (350, 350))

    return face;
def _locate_faces(image):
    faces = faceCascade.detectMultiScale(
        image,
        scaleFactor=1.1,
        minNeighbors=15,
        minSize=(70, 70)
    )

    return faces  # list of (x, y, w, h)    

def find_faces(image):
    faces_coordinates = _locate_faces(image)
    cutted_faces = [image[y:y + h, x:x + w] for (x, y, w, h) in faces_coordinates]
    normalized_faces = [_normalize_face(face) for face in cutted_faces]
    return zip(normalized_faces, faces_coordinates)

def nparray_as_image(nparray, mode='RGB'):
    """
    Converts numpy's array of image to PIL's Image.
    :param nparray: Numpy's array of image.
    :param mode: Mode of the conversion. Defaults to 'RGB'.
    :return: PIL's Image containing the image.
    """
    return Image.fromarray(np.asarray(np.clip(nparray, 0, 255), dtype='uint8'), mode)

def draw_with_alpha(source_image, image_to_draw, coordinates):
    """
    Draws a partially transparent image over another image.
    :param source_image: Image to draw over.
    :param image_to_draw: Image to draw.
    :param coordinates: Coordinates to draw an image at. Tuple of x, y, width and height.
    """
    x, y, w, h = coordinates
    print "ye le le ele ele le le le le lere"
    print "x ",x," y ",y," w ",w," h ",h
    image_to_draw = image_to_draw.resize((h, w), Image.ANTIALIAS)
    image_array = image_as_nparray(image_to_draw)
    for c in range(0, 3):
        source_image[y:y + h, x:x + w, c] = image_array[:, :, c] * (image_array[:, :, 3] / 255.0) \
                                            + source_image[y:y + h, x:x + w, c] * (1.0 - image_array[:, :, 3] / 255.0)



def _load_emoticons(emotions):
    """
    Loads emotions images from graphics folder.
    :param emotions: Array of emotions names.
    :return: Array of emotions graphics.
    """
    return [nparray_as_image(cv2.imread('graphics/%s.png' % emotion, -1), mode=None) for emotion in emotions]


def show_webcam_and_run(model, emoticons, window_size=None, window_name='webcam', update_time=10):
    """
    Shows webcam image, detects faces and its emotions in real time and draw emoticons over those faces.
    :param model: Learnt emotion detection model.
    :param emoticons: List of emotions images.
    :param window_size: Size of webcam image window.
    :param window_name: Name of webcam image window.
    :param update_time: Image update time interval.
    """
    cv2.namedWindow(window_name, WINDOW_NORMAL)
    if window_size:
        width, height = window_size
        cv2.resizeWindow(window_name, width, height)

    vc = cv2.VideoCapture(0)
    print "helllo"
    if vc.isOpened():
        read_value, webcam_image = vc.read()
        print "ere too"
    else:
        print("webcam not found")
        return

    while read_value:
        for normalized_face, (x, y, w, h) in find_faces(webcam_image):
            prediction = model.predict(normalized_face)  # do prediction
            if cv2.__version__ != '3.1.0':
                prediction = prediction[0]

            image_to_draw = emoticons[prediction]
            draw_with_alpha(webcam_image, image_to_draw, (x, y, w, h))

        cv2.imshow(window_name, webcam_image)
        read_value, webcam_image = vc.read()
        key = cv2.waitKey(update_time)

        if key == 27:  # exit on ESC
            break

    cv2.destroyWindow(window_name)


if __name__ == '__main__':
    emotions = ['neutral', 'anger', 'disgust', 'happy', 'sadness', 'surprise']
    emoticons = _load_emoticons(emotions)

    # load model
    """if cv2.__version__ == '3.1.0':
        fisher_face = cv2.face.createFisherFaceRecognizer()
    else:
        fisher_face = cv2.createFisherFaceRecognizer()
    fisher_face.load('models/emotion_detection_model.xml')"""

    # use learnt model
    clf = joblib.load("D:/myStuff/final_year_project/newCode/results.pxl")
    print clf
    window_name = 'WEBCAM (press ESC to exit)'
    show_webcam_and_run(clf, emoticons, window_size=(1600, 1200), window_name=window_name, update_time=8)
