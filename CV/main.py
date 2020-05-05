# Code for CV Client which continuously sends data to server by using keywords like GLOBAL and USERINPUT
import socket
from socket import AF_INET
from socket import SOCK_STREAM
from socket import SOL_SOCKET
from socket import SO_REUSEADDR
import threading
import time
import pygame
from keras.preprocessing.image import img_to_array
import imutils
import cv2
import cv2 as cv
from tensorflow.keras.models import load_model
import numpy as np
import dlib
from imutils import face_utils
from PIL import Image, ImageEnhance
from imutils.contours import sort_contours
import requests
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import os
import sys
from collections import deque
import math
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon as sPolygon
from copy import deepcopy
from itertools import groupby

# global variables
face_coords_client = None
nlp_client = None
blender_client = None
face_coords_startit = False
face_coords = [0, 0]
latest_camera_frame = None
last_written_frame = 10
frame_ID = 0
ocrStart = False
ret = False
blenderConnected = False
NLPConnected = False




#data from face detect
globalFaceObj = None

#data from ocr
formData = 0
newOcrResults = None
count = 0

#data from gesture
personInFrame = None
newGesture = None
gestureGlobal = None

def fetchFrame():
    global ret
    print("starting IP camera handler")
    global latest_camera_frame
    global last_written_frame
    global frame_ID
    cam = cv2.VideoCapture('http://username:password@192.168.0.101:8080/video')
    # cam = cv2.VideoCapture(0)
    print("Starting to write frames")
    while True:
        ret, latest_camera_frame = cam.read()
        if ret:
            frame_ID += 1
        if not ret:
            print("Lost connection to 'http://username:password@192.168.0.101:8080/video'")
            break;
        if not frame_ID % 10:
            cv2.imwrite("./img/"+str(frame_ID)+".jpg", latest_camera_frame)
            last_written_frame = frame_ID

        cv2.imshow("realtime", cv2.resize(latest_camera_frame, (480,480)))
        cv2.waitKey(1)


# face detection stuff
face_landmark_path = './shape_predictor_68_face_landmarks.dat'
K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
     0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
     0.0, 0.0, 1.0]
D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]
cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)
dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)
object_pts = np.float32([[6.825897, 6.760612, 4.402142],
                         [1.330353, 7.122144, 6.903745],
                         [-1.330353, 7.122144, 6.903745],
                         [-6.825897, 6.760612, 4.402142],
                         [5.311432, 5.485328, 3.987654],
                         [1.789930, 5.393625, 4.413414],
                         [-1.789930, 5.393625, 4.413414],
                         [-5.311432, 5.485328, 3.987654],
                         [2.005628, 1.409845, 6.165652],
                         [-2.005628, 1.409845, 6.165652],
                         [2.774015, -2.080775, 5.048531],
                         [-2.774015, -2.080775, 5.048531],
                         [0.000000, -3.116408, 6.097667],
                         [0.000000, -7.415691, 4.070434]])
reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                           [10.0, 10.0, -10.0],
                           [10.0, -10.0, -10.0],
                           [10.0, -10.0, 10.0],
                           [-10.0, 10.0, 10.0],
                           [-10.0, 10.0, -10.0],
                           [-10.0, -10.0, -10.0],
                           [-10.0, -10.0, 10.0]])
line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
              [4, 5], [5, 6], [6, 7], [7, 4],
              [0, 4], [1, 5], [2, 6], [3, 7]]
def get_head_pose(shape):
    image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                            shape[39], shape[42], shape[45], shape[31], shape[35],
                            shape[48], shape[54], shape[57], shape[8]])

    ret_val, rotation_vec, translation_vec = cv2.solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs)

    reprojectdst, ret_val = cv2.projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix, dist_coeffs)

    reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))

    # calc euler angle
    rotation_mat, ret_val = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    ret_val, ret_val, ret_val, ret_val, ret_val, ret_val, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

    return reprojectdst, euler_angle
def faceThread():
    print("Starting face detection thread")
    global globalFaceObj
    global ret
    global latest_camera_frame
    detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
    emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'

    print("Loading face detection models")
    # hyper-parameters for bounding boxes shape
    # loading models
    face_detection = cv2.CascadeClassifier(detection_model_path)
    emotion_classifier = load_model(emotion_model_path, compile=False)
    EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised",
                "neutral"]
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    data = {}

    # feelings_faces = []
    # for index, emotion in enumerate(EMOTIONS):
    # feelings_faces.append(cv2.imread('emojis/' + emotion + '.png', -1))

    # starting video streaming

    # camera = cv2.VideoCapture(0)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(face_landmark_path)
    print("Face detection models loaded. Starting processing")
    while ret:
        processFrameID = frame_ID
        if not processFrameID % 2:
            data['Faces'] = []
            frame = latest_camera_frame

            # ret, frame = camera.read()
            detected = False
            EulerStr = {}
            # Getting Euler angle
            if ret:
                detected = True
                face_rects = detector(frame, 0)
                if len(face_rects) > 0:
                    shape = predictor(frame, face_rects[0])
                    shape = face_utils.shape_to_np(shape)
                    reprojectdst, euler_angle = get_head_pose(shape)
                    X = str(euler_angle[0])
                    Y = str(euler_angle[1])
                    Z = str(euler_angle[2])

                    EulerStr = {
                        "X": str(X),
                        "Y": str(Y),
                        "Z": str(Z)
                    }

            # reading the frame
            frame = imutils.resize(frame, width=300)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                                    flags=cv2.CASCADE_SCALE_IMAGE)

            canvas = np.zeros((250, 300, 3), dtype="uint8")
            frameClone = frame.copy()
            if len(faces) > 0:
                detected = True
                faces = sorted(faces, reverse=True,
                               key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
                (fX, fY, fW, fH) = faces

                # Extract the ROI of the face from the grayscale image, resize it to a fixed 28x28 pixels, and then prepare
                # the ROI for classification via the CNN
                roi = gray[fY:fY + fH, fX:fX + fW]
                roi = cv2.resize(roi, (64, 64))
                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                preds = emotion_classifier.predict(roi)[0]
                emotion_probability = np.max(preds)
                label = EMOTIONS[preds.argmax()]
            else:
                continue

            for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
                # construct the label text
                text = "{}: {:.2f}%".format(emotion, prob * 100)

                # draw the label + probability bar on the canvas
                # emoji_face = feelings_faces[np.argmax(preds)]

                w = int(prob * 300)
                cv2.rectangle(canvas, (7, (i * 35) + 5),
                              (w, (i * 35) + 35), (0, 0, 255), -1)
                cv2.putText(canvas, text, (10, (i * 35) + 23),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (255, 255, 255), 1)
                cv2.putText(frameClone, label, (fX, fY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
                cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                              (0, 0, 255), 2)

            # Display Results
            cv2.imshow('your_face', frameClone)
            cv2.imshow("Probabilities", canvas)

            FaceingFalg = -1

            if ret:
                dect_face = detector(frame)
                if dect_face:
                    FaceingFalg = 1  # Looking at camera
                    for x in dect_face:
                        cv2.rectangle(frame, (x.left(), x.top()), (x.right(), x.bottom()), (0, 255, 0))
                    cv2.putText(frame, "Person is facing camera", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    eyes = eye_cascade.detectMultiScale(frame, 1.3, 5)
                    for (x1, y1, w, h) in eyes:
                        cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 1)
                else:
                    FaceingFalg = 0  # Not looking at camera
                    cv2.putText(frame, "Person is not facing camera", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # cv2.imshow("demo", frame)
            else:
                print("Error in reading from camera")

            if detected:
                globalFaceObj = {
                    "face_towards_camera": str(FaceingFalg),
                    "emotion": label,
                    "Euler angle": EulerStr,
                    "frameID" : processFrameID
                }
                print(globalFaceObj);
                data.clear()
            else:
                print("no face")
            cv2.waitKey(1)
    cv2.destroyAllWindows()


#ocr stuff
def get_image_width_height(image):
    image_width = image.shape[1]  # current image's width
    image_height = image.shape[0]  # current image's height
    return image_width, image_height
def calculate_scaled_dimension(scale, image):
    # http://www.pyimagesearch.com/2014/01/20/basic-image-manipulations-in-python-and-opencv-resizing-scaling-rotating-and-cropping/
    image_width, image_height = get_image_width_height(image)
    ratio_of_new_with_to_old = scale / image_width
    dimension = (scale, int(image_height * ratio_of_new_with_to_old))
    return dimension
def scale_image(image, size):
    if image.size > 0:
        image_resized_scaled = cv2.resize(
            image,
            calculate_scaled_dimension(
                size,
                image
            ),
            interpolation=cv2.INTER_AREA

        )
        return image_resized_scaled
def detect_box(image, cropIt=True):
    # https://stackoverflow.com/questions/36982736/how-to-crop-biggest-rectangle-out-of-an-image/36988763
    # Transform colorspace to YUV
    image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    image_y = np.zeros(image_yuv.shape[0:2], np.uint8)
    image_y[:, :] = image_yuv[:, :, 0]

    # Blur to filter high frequency noises
    image_blurred = cv2.GaussianBlur(image_y, (3, 3), 0)
    # show_image(image_blurred, window_name)

    # Apply canny edge-detector
    edges = cv2.Canny(image_blurred, 100, 300, apertureSize=3)
    # show_image(edges, window_name)

    # Find extrem outer contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mx = (0, 0, 0, 0)  # biggest bounding box so far
    mx_area = 0

    for c in contours:
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        # convert all coordinates floating point values to int
        box = np.int0(box)

        x, y, w, h = cv2.boundingRect(c)
        area = w * h

        if area > mx_area:
            mx = x, y, w, h
            mx_area = area

        # cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
        # show_image(image, window_name)

    x, y, w, h = mx

    roi = image[y:y + h, x:x + w]
    if (roi.shape[1] == 0 or roi.shape[0] == 0):
        print("Couldn't detect edges in image")
        return 0

    cv2.imwrite("cropped_img.jpeg", roi)
    return 1
def show_image(image, window_name):
    # Show image
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, image)
    image_width, image_height = get_image_width_height(image)
    cv2.resizeWindow(window_name, image_width, image_height)

    # Wait before closing
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def resizeAndDisplayImage(image, scalePercent=10):
    dimensions = (int(image.shape[1] * scalePercent / 100), int(image.shape[0] * scalePercent / 100))
    resizedImage = cv.resize(image, dimensions, interpolation=cv.INTER_AREA)
    # cv.imshow('Resized Image', resizedImage)
def remove_outliers(data, m=2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / (mdev if mdev else 1.)
    return data[s < m]
def increase_contrast(filepath):
    im = Image.open("cropped_img.jpeg")
    enhancer = ImageEnhance.Sharpness(im)
    enhanced_im = enhancer.enhance(10.0)
    filename = "enhanced_img.png"
    enhanced_im.save("enhanced_img.png")
    return filename
def box_extraction(img_for_box_extraction_path, cropped_dir_path):
    img = cv2.imread(img_for_box_extraction_path, 0)  # Read the image

    (thresh, img_bin) = cv2.threshold(img, 128, 255,
                                      cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # Thresholding the image
    # img_bin = cv2.bitwise_not(img_bin)
    # cv2.imshow()
    img_bin = 255 - img_bin  # Invert the image
    cv2.imwrite("Image_bin.png", img_bin)

    # Defining a kernel length
    kernel_length = np.array(img).shape[1] // 40

    # A verticle kernel of (1 X kernel_length), which will detect all the verticle lines from the image.
    verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
    # A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    # A kernel of (3 X 3) ones.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # Morphological operation to detect verticle lines from an image
    img_temp1 = cv2.erode(img_bin, verticle_kernel, iterations=3)
    verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=3)
    cv2.imwrite("verticle_lines.jpg", verticle_lines_img)
    # Morphological operation to detect horizontal lines from an image
    img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=3)

    horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)

    cv2.imwrite("horizontal_lines.jpg", horizontal_lines_img)
    # Weighting parameters, this will decide the quantity of an image to be added to make a new image.
    alpha = 0.5
    beta = 1.0 - alpha
    # This function helps to add two image with specific weight parameter to get a third image as summation of two image.
    img_final_bin = cv2.addWeighted(verticle_lines_img, alpha, horizontal_lines_img, beta, 0.0)
    img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)
    (thresh, img_final_bin) = cv2.threshold(img_final_bin, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # For Debugging
    # Enable this line to see verticle and horizontal lines in the image which is used to find boxes
    cv2.imwrite("img_final_bin.jpg", img_final_bin)
    # Find contours for image, which will detect all the boxes
    contours, hierarchy = cv2.findContours(
        img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Sort all the contours by top to bottom.
    (contours, boundingBoxes) = sort_contours(contours, method="top-to-bottom")
    idx = 0
    width = []
    height = []
    hor = 1
    for c in contours:
        # Returns the location and width,height for every contour
        x, y, w, h = cv2.boundingRect(c)
        if np.average(h) > np.average(w):
            y, x, h, w = cv2.boundingRect(c)
            hor = 0

        # print(w,h)
        # If the box height is greater then 20, width is >80, then only save it as a box in "cropped/" folder.
        if (w > 40 and h > 4) and w > (2 * h):  ##10 1, 2 2, 485 4

            idx += 1
            if (hor):
                new_img = img[y:y + h, x:x + w]
            else:
                new_img = img[x:x + w, y:y + h]

            width.append(w)
            height.append(h)
            # cv2.imshow('image', new_img)
            # cv2.waitKey(0)

    if idx > 0:
        data_points_w = np.asarray(width)
        data_points_h = np.asarray(height)
        if (count > 5):
            remove_outliers(data_points_w, 2.)
        else:
            remove_outliers(data_points_w, 1.)
        remove_outliers(data_points_h, 1.)

        if (data_points_w.size >= 5 and data_points_h.size >= 5):
            # print("Document is in correct format")
            return 1
        else:
            sqwsw = 0
            # print("Detected format is incorrect")
    else:
        sqwsw = 0
        # print("Detected format is incorrect")
def contour_detection(filePath):
    img = cv.imread(filePath, 0)
    imgRGB = cv.imread(filePath, 1)
    dim = (img.shape[1], img.shape[0])

    cv2.imshow("processessing", imgRGB)
    cv.waitKey(500)

    # extend threshold function to optimize for all lighting conditions
    # these values tend to work well for non-flash images
    _, thresh = cv.threshold(img, 135, 255, cv.THRESH_BINARY_INV)
    resizeAndDisplayImage(thresh)


    # classify all detected shapes as valid/invalid question/answer fields aspect ratio approach doesnt work when
    # photograph is taken at an angle/image is rotated
    #
    # possible fix - take extreme points of each box and use them to fix
    # the skewness OR use rhombus dimensions to work it out
    num = 0
    contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    FieldDimensions = []
    field = 0
    count = 0
    detected = 0
    for cnt in contours:
        # print(cv.contourArea(cnt))

        if cv.contourArea(cnt) > 150:
            num += 1
        if cv.contourArea(cnt) > 1000:
            # print(cv.contourArea(cnt))
            approx = cv.approxPolyDP(cnt, 0.009 * cv.arcLength(cnt, True), True)

            rect = cv.boundingRect(cnt)
            img = cv.drawContours(imgRGB, [approx], 0, (0, 0, 255), 2)
            # cv.imshow("image",img)
            # cv.waitKey(0)
            FieldDimensions.append(rect)
            count += 1
            # print(len(approx))
            if len(approx) == 4:
                field += 1
    if num == 11:
        # print("Document is in correct format")
        detected = 1
        # print(detected)

    # print(num)
    cv.imwrite("contours.jpeg", img)
    return detected
def ocrThread():
    print("Starting OCR thread")
    global count
    global formData
    global newOcrResults
    while True:
        try:
            lastProcessed = -1
            global last_written_frame
            while True:
                missing_env = False
                detected = 0
                count = 0
                size_max_image = 500
                check = 0
                while last_written_frame == lastProcessed:
                    time.sleep(0.1)
                processFrameID = last_written_frame
                # print("reading " + str(processFrameID))
                filename = "./img/" +str(processFrameID)+'.jpg'
                lastProcessed = processFrameID
                # filename = "./img/10.jpg"
                org_img = filename
                # _____________________________CONTOUR DETECTION______________________________________
                detected = contour_detection(filename)
                flag = 1
                val = 0
                if detected == 0:

                    # _____________________________CROP IMAGE______________________________________
                    image = cv2.imread(filename)
                    image = scale_image(image, size_max_image)
                    vu = detect_box(image, True)

                    if vu == 1:
                        image = "cropped_img.jpeg"
                        # _____________________________ENHANCE______________________________________

                        filename = increase_contrast("cropped_img.jpeg")

                        # _____________________________CROP IMAGE______________________________________
                        image = cv2.imread(filename)
                        image = scale_image(image, size_max_image)
                        detect_box(image, True)
                        # cv2.imwrite("enhanced_img.png",image)

                        # _____________________________CONTOUR DETECTION______________________________________
                        filePath = 'cropped_img.jpeg'
                        detected = contour_detection(filePath)

                        # _____________________________BOX DETECTION______________________________________

                        if detected == 0:
                            val = box_extraction("contours.jpeg", "./Cropped/")
                        image_path = "cropped_img.jpeg"


                else:
                    image_path = org_img
                    flag = 0

                if val == 1 or detected == 1:

                    # Add your Computer Vision subscription key and endpoint to your environment variables.
                    if 'COMPUTER_VISION_ENDPOINT' in os.environ:
                        endpoint = os.environ['COMPUTER_VISION_ENDPOINT']
                    else:
                        print("From Azure Cogntivie Service, retrieve your endpoint and subscription key.")
                        print(
                            "\nSet the COMPUTER_VISION_ENDPOINT environment variable, such as \"https://westus2.api.cognitive.microsoft.com\".\n")
                        missing_env = True

                    if 'COMPUTER_VISION_SUBSCRIPTION_KEY' in os.environ:
                        subscription_key = os.environ['COMPUTER_VISION_SUBSCRIPTION_KEY']
                    else:
                        print("From Azure Cogntivie Service, retrieve your endpoint and subscription key.")
                        print(
                            "\nSet the COMPUTER_VISION_SUBSCRIPTION_KEY environment variable, such as \"1234567890abcdef1234567890abcdef\".\n")
                        missing_env = True

                    if missing_env:
                        print("**Restart your shell or IDE for changes to take effect.**")
                        sys.exit()

                    t = time.localtime()
                    current_time = time.strftime("%H:%M:%S", t)
                    # print(current_time)

                    ocr_url = endpoint + "/vision/v3.0-preview/read/analyze"
                    # Set the langauge that you want to recognize. The value can be "en" for English, and "es" for Spanish
                    language = "en"
                    image_data = open(image_path, "rb").read()
                    headers = {'Ocp-Apim-Subscription-Key': subscription_key,
                               'Content-Type': 'application/octet-stream'}
                    params = {'visualFeatures': 'Categories,Description,Color'}
                    response = requests.post(ocr_url, headers=headers, params=params, data=image_data)
                    response.raise_for_status()

                    # Extracting text requires two API calls: One call to submit the
                    # image for processing, the other to retrieve the text found in the image.

                    # Holds the URI used to retrieve the recognized text.
                    operation_url = response.headers["Operation-Location"]

                    # The recognized text isn't immediately available, so poll to wait for completion.
                    analysis = {}
                    poll = True
                    while (poll):
                        response_final = requests.get(
                            response.headers["Operation-Location"], headers=headers)
                        analysis = response_final.json()

                        # print(json.dumps(analysis, indent=4))

                        time.sleep(1)
                        if ("analyzeResult" in analysis):
                            poll = False
                        if ("status" in analysis and analysis['status'] == 'failed'):
                            poll = False

                    polygons = []
                    if ("analyzeResult" in analysis):
                        # Extract the recognized text, with bounding boxes.
                        polygons = [(line["boundingBox"], line["text"])
                                    for line in analysis["analyzeResult"]["readResults"][0]["lines"]]
                    t1 = time.localtime()
                    current_time1 = time.strftime("%H:%M:%S", t1)
                    # print(current_time1)
                    # Display the image and overlay it with the extracted text.
                    i = 0
                    size = 30
                    text = ["" for x in range(size)]
                    image = Image.open(image_path)
                    ax = plt.imshow(image)
                    for polygon in polygons:
                        vertices = [(polygon[0][i], polygon[0][i + 1])
                                    for i in range(0, len(polygon[0]), 2)]
                        text[i] = polygon[1]
                        i += 1
                        patch = Polygon(vertices, closed=True, fill=False, linewidth=2, color='y')
                        ax.axes.add_patch(patch)
                        # plt.text(vertices[0][0], vertices[0][1], text, fontsize=20, va="top")

                    if (text[0] == "COURSE DROP FORM"):
                        flag = 1
                    if (text[0].upper() == "COURSE CODE"):
                        flag = 0
                    student_data = {

                        text[0 + flag]: text[1 + flag],
                        text[2 + flag]: text[3 + flag],
                        text[4 + flag]: text[5 + flag],
                        text[6 + flag]: text[7 + flag],
                        text[8 + flag]: text[9 + flag],
                        "Detection Time": current_time
                    }
                    if len(student_data) == 6:
                        newOcrResults = True
                        print(student_data)
                        formData = deepcopy(student_data)
        except:
            # e = sys.exc_info()[0]
            # print(e)
            print("ocr crashed again. Restarting.")
            pass


#gesture stuff
def getRelativeDirection(old, new, distanceRef, ls = False):
    if ls:
        sens = 2
    else:
        sens = 5

    if distance(old, new) > sens:
        a = calculateAngle(old, new)
        # print("distance: " + str(distance(old, new)))
        # print(a)
        if -45 < a < 45:
            return 1
        if a < -135 or a > 135:
            return 2
        if -135 < a < -45:
            return 3
        if 45 < a < 135:
            return 4
    return -1
def detectGuesture(s):
    parsed = []
    for i in range(1, 29):
        if s[i] == s[i + 1]:
            n = 0
        elif s[i] != s[i + 1] and s[i] == s[i - 1]:
            parsed.append(s[i])

    y = [x[0] for x in groupby(parsed)]

    stt = (str(y).replace(', ', ''))
    print(stt)


    if '2121' in stt or '1212' in stt:
        return('UD')
    if '3434' in stt or '4343' in stt:
        return('LR')

    return "none"



class circularlist(object):
    def __init__(self, size, data = []):
        """Initialization"""
        self.index = 0
        self.size = size
        self._data = list(data)[-size:]

    def append(self, value):
        """Append an element"""
        if len(self._data) == self.size:
            self._data[self.index] = value
        else:
            self._data.append(value)
        self.index = (self.index + 1) % self.size

    def __getitem__(self, key):
        """Get element by index, relative to the current index"""
        if len(self._data) == self.size:
            return(self._data[(key + self.index) % self.size])
        else:
            return(self._data[key])

    def __repr__(self):
        """Return string representation"""
        return self._data.__repr__() + ' (' + str(len(self._data))+' items)'
    def __contains__(self, key):
        return key in self._data
    def potentialGesture(self):
        return -1 not in self._data and -2 not in self._data
def calculateAngle(x, y):
    if x[0] == 0 or x[1] == 0 or y[1] == 0 or y[0] == 0:
        return -1000
    else:
        return int(math.degrees(math.atan2((x[0] - y[0]), (x[1] - y[1]))))
def torsoHipAdjustment(hip, mid, increase):
    Tdistance = abs(mid[0] - hip[0])
    if increase:
        return (hip[0] + Tdistance, hip[1])
    else:
        return (hip[0] - Tdistance, hip[1])
def distance(x, y, euq=True, vrtcl=True):
    if euq:
        return int(math.sqrt((x[0] - y[0]) ** 2 + ((x[1] - y[1]) ** 2)))
    if vrtcl:
        return abs(x[0] - y[0])
    else:
        return abs(x[1] - y[1])
def isStraight(x):
    if type(x) == int:
        return abs(x) > 160
    elif type(x) == list:
        flag = True;
        for a in x:
            if flag == False:
                break;
            flag = abs(a) > 160
        return flag
#limb list lUA, LFA, RUA, RFA, LT, LS, RT, RS
def updateLimbLengths(c, lenHere, recalib = False, rate = 0.6):
    if not recalib:
        lenHere[0] = distance(c[5], c[6])
        lenHere[1] = distance(c[6], c[7])
        lenHere[2] = distance(c[2], c[3])
        lenHere[3] = distance(c[3], c[4])
        lenHere[4] = distance(c[12], c[13])
        lenHere[5] = distance(c[13], c[14])
        lenHere[6] = distance(c[9], c[10])
        lenHere[7] = distance(c[10], c[11])
    else:
        lenHere[0] = lenHere[0] * (1 - rate) + rate * distance(c[5], c[6])
        lenHere[1] = lenHere[1] * (1 - rate) + rate * distance(c[6], c[7])
        lenHere[2] = lenHere[2] * (1 - rate) + rate * distance(c[2], c[3])
        lenHere[3] = lenHere[3] * (1 - rate) + rate * distance(c[3], c[4])
        lenHere[4] = lenHere[4] * (1 - rate) + rate * distance(c[12], c[13])
        lenHere[5] = lenHere[5] * (1 - rate) + rate * distance(c[13], c[14])
        lenHere[6] = lenHere[6] * (1 - rate) + rate * distance(c[9], c[10])
        lenHere[7] = lenHere[7] * (1 - rate) + rate * distance(c[10], c[11])
    return lenHere
def calculateExtensions(rLen, fLen):
    extensions = [0 for i in range(8)]
    for i in range(8):
        if fLen[i] < rLen[i] and rLen[i] != 0 and fLen[i] != 0:
            extensions[i] = int(((math.sqrt(rLen[i]**2 - fLen[i]**2)) / rLen[i]) * 100)
        else:
            extensions[i] = 0

    return extensions
def torsoHipAdjustment(hip, mid, increase):
    Tdistance = abs(mid[0] - hip[0])
    if increase:
        return (hip[0] + Tdistance, hip[1])
    else:
        return (hip[0] - Tdistance, hip[1])

def getMessage(sockConn, size):
    initSize = size
    fetchData = 0
    remainingData = initSize
    finalStr = ''
    while fetchData < initSize:
        data = sockConn.recv(remainingData).decode()
        finalStr = finalStr + data
        fetchData = len(data)
        remainingData = initSize - fetchData
    return finalStr
def azure_keypoint_detection():
    printAngles = False
    global face_coords
    global personInFrame
    global gestureGlobal
    leftHandGesture = None
    rightHandGesture = None
    host = '0.0.0.0'
    port = 10000
    DISPLAY = pygame.display.set_mode((int(1080 * 1), int(1080 * 1)), 0, 32)
    mySocket = socket.socket()
    mySocket.bind((host, port))
    fc = 1
    # limb list lUA, LFA, RUA, RFA, LT, LS, RT, RS
    referenceLimbLength = [0 for x in range(8)]
    referenceExists = False
    frameLimbLength = [0 for x in range(8)]

    noseTrack = circularlist(15, [-2 for j in range(15)])
    rightHandTrack = circularlist(15, [-2 for j in range(30)])
    leftHandTrack = circularlist(15, [-2 for j in range(30)])

    coords = [[0, 0] for x in range(25)]
    lastCoords = [[0, 0] for x in range(25)]

    print("Waiting for connection from openpose@azure")
    mySocket.listen(1)
    conn, addr = mySocket.accept()
    print("Azure server recieved connection from: " + str(addr))

    while True:
        pygame.event.get()

        #get data from azure
        sizeof = int((getMessage(conn, 7)))
        data = getMessage(conn, sizeof)
        fc += 1

        #check if anyone in frame
        if data == "no person":
            if personInFrame:
                print("no person in frame")
            personInFrame = False
        else:
            personInFrame = True
            cords = data.split(':')
            for jdx in range(0, 50, 2):
                coords[jdx//2][0] = int(cords[jdx])
                coords[jdx//2][1] = int(cords[jdx + 1])

            for tdx, point in enumerate(coords):
                pygame.draw.circle(DISPLAY, (255, 255, 255), (int(point[0] * 1), int(point[1] * 1)), 5)

            pygame.draw.polygon(DISPLAY, (200, 200, 10),
                                [coords[5], coords[2], torsoHipAdjustment(coords[9], coords[8], False),
                                 torsoHipAdjustment(coords[12], coords[8], True)], 1)

            torsoPolygon = sPolygon([coords[5], coords[2], torsoHipAdjustment(coords[9], coords[8], True),
                                    torsoHipAdjustment(coords[12], coords[8], False)])

            #update global face coords variable
            face_coords[0] = (int(coords[0][0]) - 540) / 540
            face_coords[1] = (int(coords[0][1]) - 540) / 540
            #
            pygame.draw.line(DISPLAY, (155, 155, 0), coords[8], coords[1], 1)
            torsoAngle = calculateAngle(coords[1], coords[8])
            if printAngles: print("Torso angle: " + str(torsoAngle))

            # hand distance and relative angle (0 being flat/in line) 7 : 4
            handRelativeAngle = calculateAngle(coords[4], coords[7])
            if printAngles: print("Hand angles: " + str(handRelativeAngle))

            # right forearm angle
            rightForearmAngle = calculateAngle(coords[3], coords[4])
            if printAngles: print("Right forearm angle: " + str(rightForearmAngle))

            # right upperarm angle
            rightUpperarmAngle = calculateAngle(coords[2], coords[3])
            if printAngles: print("Right upperarm angle: " + str(rightUpperarmAngle))

            # left upper arm angle
            leftUpperarmAngle = calculateAngle(coords[5], coords[6])
            if printAngles: print("Right forearm angle: " + str(leftUpperarmAngle))

            # left forearm angle
            leftForearmAngle = calculateAngle(coords[6], coords[7])
            if printAngles: print("Left forearm angle: " + str(leftForearmAngle))

            # right hand height
            rightHandHeight = coords[12][1] - coords[4][1]
            # print("right hand height: " + str(rightHandHeight))

            # left hand height
            leftHandHeight = coords[12][1] - coords[7][1]
            # print("Left hand height: " + str(leftHandHeight))

            #facing front
            rlDiff = coords[9][0] - coords[12][0]
            if rlDiff <= 0:
                forwardFacing = True
            else:
                forwardFacing = False


            # hands crossed
            rlDiff = coords[4][0] - coords[7][0]
            if rlDiff <= 0:
                hc = False
            else:
                hc = True
            # print("hands crossed: " + str(hc))

            # check if left upperarm covers torso
            LUAT = torsoPolygon.contains(Point(coords[6]))
            # print("left upper arm in front of chest: " + str(LUAT))

            # check if right upperarm covers torso
            RUAT = torsoPolygon.contains(Point(coords[3]))
            # print("right upper arm in front of chest: " + str(RUAT))

            # leftThigh
            leftThighAngle = calculateAngle(coords[12], coords[13])
            if printAngles: print("Left thigh angle: " + str(leftThighAngle))

            # leftThigh
            rightThighAngle = calculateAngle(coords[9], coords[10])
            if printAngles: print("right thigh angle: " + str(rightThighAngle))

            # left shin
            leftShinAngle = calculateAngle(coords[13], coords[14])
            if printAngles: print("Left shin angle: " + str(leftShinAngle))

            # right shin
            rightShinAngle = calculateAngle(coords[10], coords[11])
            if printAngles: print("right shin angle: " + str(rightShinAngle))

            ## check if relaxed/normal body pos
            # print('relaxed: ', end="")
            relaxedState = (isStraight(
                [leftForearmAngle, rightForearmAngle, leftUpperarmAngle, rightUpperarmAngle, leftShinAngle,
                 leftThighAngle,
                 rightShinAngle, rightThighAngle])
                            and rightHandHeight < 15 and leftHandHeight < 15)

            if relaxedState:
                if not referenceExists:
                    referenceLimbLength = updateLimbLengths(coords, referenceLimbLength, False)
                    referenceExists = True
                else:
                    referenceLimbLength = updateLimbLengths(coords, referenceLimbLength, True)

            # calculate limb extension
            elif (referenceExists):
                frameLimbLength = updateLimbLengths(coords, frameLimbLength)
                extensions = calculateExtensions(referenceLimbLength, frameLimbLength)
                types = ["lUA", "LFA", "RUA", "RFA", "LT", "LS", "RT", "RS"]
                for i in range(8):
                    print(types[i] + " " + str(extensions[i]), end=" | ")

            if coords[4][1] < coords[0][1]:
                rRaised = True
            else:
                rRaised = False
            if coords[7][1] < coords[0][1]:
                lRaised = True
            else:
                lRaised = False



            #compute torso length
            #use this to normalize everything
            spineLength = distance(coords[1], coords[8])

            #check if neck moves more than noise -> kill all patterns
            if distance(coords[1], lastCoords[1]) > 20:
                noseTrack = circularlist(15, [-2 for j in range(15)])
                leftHandTrack = circularlist(15, [-2 for j in range(30)])
                rightHandTrack = circularlist(15, [-2 for j in range(30)])
            else:
                noseTrack.append(getRelativeDirection(lastCoords[1], coords[1], spineLength, True))
                leftHandTrack.append(getRelativeDirection(lastCoords[7], coords[7], spineLength))
                rightHandTrack.append(getRelativeDirection(lastCoords[4], coords[4], spineLength))

            if noseTrack.potentialGesture():
                for j in range(15):
                    print(noseTrack[j], end = " ")
                print(" : NOSE")

            if leftHandTrack.potentialGesture():
                # for j in range(30):
                #     print(leftHandTrack[j], end = " ")
                # print(" : LH")
                leftHandGesture = (detectGuesture(leftHandTrack))

            if rightHandTrack.potentialGesture():
                for j in range(30):
                    print(rightHandTrack[j], end = " ")
                print(" : RH")
                rightHandGesture = (detectGuesture(rightHandTrack))

            obj = {
                "rightHandHeight ": rightHandHeight,
                "leftHandHeight ": leftHandHeight,
                "hands crossed ": hc,
                "relaxed": relaxedState,
                "upper body upright": isStraight(torsoAngle),
                "standing": isStraight([leftShinAngle, rightShinAngle, leftThighAngle, rightThighAngle]),
                "left arm raised": lRaised,
                "right arm raised": rRaised,
                "facing front": forwardFacing,
                "leftHandGesture": leftHandGesture,
                "rightHandGesture": rightHandGesture
            }
            gestureGlobal = obj
            print()
            print(obj)

            lastCoords = deepcopy(coords)
            pygame.display.update()
            pygame.time.wait(1)
            DISPLAY.fill((0, 0, 0))

    # conn.close()


#blender/nlp stuff
def init_server_face_coords():
    # LOCALHOST = '192.168.0.170'
    LOCALHOST = '0.0.0.0'
    PORT = 10010
    global face_coords_startit
    global face_coords_client
    cserver = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    cserver.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    cserver.bind((LOCALHOST, PORT))
    print("coords server waiting for client request..")
    time.sleep(0.5)
    cserver.listen(1)
    clientsock, clientAddress = cserver.accept()
    face_coords_client = clientsock
    print('coords server got connection from {}'.format(clientAddress))
    face_coords_startit = True
    print("Send face coords: " + str(face_coords_startit))

def send_face_coords():
    global face_coords
    global face_coords_client
    global face_coords_startit
    while not face_coords_startit:
        a = 0
    while True:
        precision = 10000
        convertedCoords = [0, 0]

        convertedCoords[0] = int(face_coords[0]  * precision) + precision
        convertedCoords[1] = int(face_coords[1] * -1 * precision) + precision

        # sending X
        face_coords_client.sendall(convertedCoords[0].to_bytes(4, byteorder='big'))
        face_coords_client.recv(2048).decode()

        # sending Y
        face_coords_client.sendall(convertedCoords[1].to_bytes(4, byteorder='big'))
        face_coords_client.recv(2048).decode()

        # send data after every 0.1 seconds
        time.sleep(0.1)
def init_server_blender_nlp():
    global blenderConnected

    # LOCALHOST = '192.168.0.170'
    LOCALHOST = '0.0.0.0'
    PORT = 10005
    bserver = socket.socket(AF_INET, SOCK_STREAM)
    bserver.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
    bserver.bind((LOCALHOST, PORT))
    global nlp_client
    global blender_client
    global NLPConnected

    for i in range(2):
        time.sleep(0.5)
        print("Blender/NLP server waiting for client request..")
        bserver.listen(1)
        clientsock, clientAddress = bserver.accept()
        input_client = clientsock.recv(1024)
        print(input_client)
        if input_client.decode().upper() == 'NLP':
            nlp_client = clientsock
            NLPConnected = True
            print('Got NLP connection')
            nlp_client.sendall(bytes('connected ya beatches', 'UTF-8'))
        elif input_client.decode().upper() == 'BLENDER':
            blender_client = clientsock
            blenderConnected = True
            print('Got blender connection')
            blender_client.sendall(bytes('connected ya beatches', 'UTF-8'))
        else:
            print('invalid Connection string')
            clientsock.sendall(bytes('invalid Connection string', 'UTF-8'))

def getEmotionString(str):
    if str=='happy':
        return "posture, I am very happy today."
    if str=='scared':
        return "posture, I do not feel very comfortable."
    if str=='sad':
        return "posture, Everything has made me sad."
    if str=='surprised':
        return "posture, I did not expect this behaviour"
    if str=='neutral':
        return "posture, This is all very normal to me."
    if str=='disgust':
        return "posture, This disgusts me."
    if str=='angry':
        return "posture, I feel angry."

def communicate():

    global blenderConnected
    global NLPConnected
    global newOcrResults
    global formData
    global globalFaceObj
    global frame_ID
    global gestureGlobal
    lastSentState = 2
    sentPersonInFrame = False
    lastNLPSendTime = 0
    lastFormSendTime = 0
    formRetries = 3
    lastSentGesture = -1
    while not blenderConnected or not NLPConnected:
        time.sleep(0.2)

    print("starting to send states")

    while True:
        time.sleep(0.1)
        #blender thinks no one in frame
        if not sentPersonInFrame:
            if personInFrame:
                if lastSentState != 1:
                    print("sending person in frame (1)")
                    user_input = "1"
                    lastSentState = 1
                    blender_client.sendall(bytes('update_state', 'UTF-8'))
                    blender_client.sendall(int(user_input).to_bytes(1, byteorder='big'))
                    print(blender_client.recv(1))
                    sentPersonInFrame = True

        elif sentPersonInFrame:
            if lastSentState != 2 and not personInFrame:
                print("sending no person in frame (2)")
                lastSentState = 2
                user_input = "2"
                blender_client.sendall(bytes('update_state', 'UTF-8'))
                blender_client.sendall(int(user_input).to_bytes(1, byteorder='big'))
                print(blender_client.recv(1))
                sentPersonInFrame = False

            #do remaining logic -> blender knows person is in frame

            #check if form data is present
            checkData = deepcopy(formData)
            if newOcrResults:
                print("new ocr")
                # if len(checkData) == 6 and 'COURSE CODE' in checkData and 'ROLL NUMBER' in checkData and time.time() - lastFormSendTime > 10 :
                print("sending valid form data (5)")
                # user_input = "form, my roll number is {0} and course Id is {1}"\
                #     .format(checkData['ROLL NUMBER'], checkData['COURSE CODE'])
                user_input = "form, my roll number is 17L-4125 and course Id is CS 549"

                nlp_client.sendall(bytes('cv_input', 'UTF-8'))
                nlp_client.sendall(bytes(user_input, 'UTF-8'))
                print(nlp_client.recv(1))
                lastNLPSendTime = time.time()

                lastSentState = 5
                user_input = "5"
                blender_client.sendall(bytes('update_state', 'UTF-8'))
                blender_client.sendall(int(user_input).to_bytes(1, byteorder='big'))
                blender_client.recv(1)
                lastFormSendTime = time.time()
                newOcrResults = False

                # elif len(checkData) == 6 and time.time() - lastFormSendTime > 10 :
                #     if formRetries > 0:
                #         formRetries -= 1
                #     else:
                #         print("sending invalid form state (6)")
                #         lastSentState = 6
                #         user_input = "6"
                #         blender_client.sendall(bytes('update_state', 'UTF-8'))
                #         blender_client.sendall(int(user_input).to_bytes(1, byteorder='big'))
                #         blender_client.recv(1)
                #         formRetries = 3
                #         lastFormSendTime = time.time()
                # newOcrResults = False

            #randomly send face expressions
            # if time.time() - lastNLPSendTime > 10 and frame_ID - globalFaceObj['frameID'] < 20:
            #     print("sending face expression")
            #     user_input = getEmotionString(globalFaceObj['emotion'])
            #     nlp_client.sendall(bytes('cv_input', 'UTF-8'))
            #     nlp_client.sendall(bytes(user_input, 'UTF-8'))
            #     lastNLPSendTime = time.time()

            #if gesture present send it
            if time.time() - lastNLPSendTime > 3 and not gestureGlobal['standing']:
                print("sending not standing to NLP")
                user_input = "posture, I feel tired."
                nlp_client.sendall(bytes('cv_input', 'UTF-8'))
                nlp_client.sendall(bytes(user_input, 'UTF-8'))
                print(nlp_client.recv(1))
                lastNLPSendTime = time.time()

            # if time.time() - lastNLPSendTime > 3  and not gestureGlobal['upper body upright']:
            #     user_input = "posture, I don't think I want to talk to you."
            #     nlp_client.sendall(bytes('cv_input', 'UTF-8'))
            #     nlp_client.sendall(bytes(user_input, 'UTF-8'))
            #     print(nlp_client.recv(1))
            #     lastNLPSendTime = time.time()

            if time.time() - lastNLPSendTime > 3 and not gestureGlobal['facing front']:
                print("sending not looking at to NLP")
                user_input = "posture, I do not think I want to talk to you."
                nlp_client.sendall(bytes('cv_input', 'UTF-8'))
                nlp_client.sendall(bytes(user_input, 'UTF-8'))
                print(nlp_client.recv(1))
                lastNLPSendTime = time.time()

            if time.time() - lastNLPSendTime > 3 and gestureGlobal['leftHandGesture'] == 'UD':
                print("sending left yes gesture to NLP")
                user_input = "posture, Yes."
                nlp_client.sendall(bytes('cv_input', 'UTF-8'))
                nlp_client.sendall(bytes(user_input, 'UTF-8'))
                print(nlp_client.recv(1))
                lastNLPSendTime = time.time()

            if time.time() - lastNLPSendTime > 3 and gestureGlobal['leftHandGesture'] == 'LR':
                print("sending left no gesture to NLP")
                user_input = "posture, No."
                nlp_client.sendall(bytes('cv_input', 'UTF-8'))
                nlp_client.sendall(bytes(user_input, 'UTF-8'))
                print(nlp_client.recv(1))
                lastNLPSendTime = time.time()




if __name__ == '__main__':
    # server = threading.Thread(target=init_server_blender_nlp, args=())
    # server.start()
    # server.join()
    #
    # server2 = threading.Thread(target=init_server_face_coords, args=())
    # server2.start()
    # server2.join()

    # start ipcamera reader
    camera_thread = threading.Thread(target=fetchFrame, args=())
    camera_thread.start()
    time.sleep(2)
    # #
    #start ocr
    ocr = threading.Thread(target=ocrThread, args=())
    ocr.start()
    #
    #
    #start face emotion
    # face = threading.Thread(target=faceThread, args=())
    # face.start()
    # # # #
    # #start gesture recognition
    # azure_datareader = threading.Thread(target=azure_keypoint_detection, args=())
    # azure_datareader.start()
    # #
    # # #start blender/nlp thread
    # #
    # networking = threading.Thread(target=communicate, args=())
    # networking.start()
    #
    # # setup face coord threads
    # coords_send = threading.Thread(target=send_face_coords, args=())
    # coords_send.start()

