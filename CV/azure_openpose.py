import sys
import cv2
import os
from sys import platform
import argparse
import time
import socket
import numpy as np



def packageArray(a):
    out = ""
    for row in a[0]:
        out = out + str(int(row[0])) + ' : ' + str(int(row[1])) + ':'
        # print(str(int(row[0])) + ':'+ str(int(row[1])))
    return out


host = '43.245.206.104'
port = 10000

mySocket = socket.socket()
mySocket.connect((host, port))

try:
    # Import Openpose (Windows/Ubuntu/OSX)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        # Windows Import
        if platform == "win32":
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append(dir_path + '/../../python/openpose/Release');
            os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' + dir_path + '/../../bin;'
            import pyopenpose as op
        else:
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append('../../python');
            # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
            # sys.path.append('/usr/local/python')
            from openpose import pyopenpose as op
    except ImportError as e:
        print(
            'Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e

    parser = argparse.ArgumentParser()
    # parser.add_argument("--video", default="./tv/extension.avi", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
    args = parser.parse_known_args()

    params = dict()
    params["model_folder"] = "../../../models/"
    params["net-resolution"] = "-1x160"
    params["number_people_max"] = "1"
    params["process_real_time"] = "1"
    # params["scale_number"] = "4"
    # params["scale_gap"] = "0.25"
    params["model_pose"] = "BODY_25"
    # params["write-keypoint"] = "3"
    # params["face"] = True
    params["disable_multi_thread"] = 1
    params["render_pose"] = 0
    params["display"] = 0

    # Add others in path?
    for i in range(0, len(args[1])):
        curr_item = args[1][i]
        if i != len(args[1]) - 1:
            next_item = args[1][i + 1]
        else:
            next_item = "1"
        if "--" in curr_item and "--" in next_item:
            key = curr_item.replace('-', '')
            if key not in params:  params[key] = "1"
        elif "--" in curr_item and "--" not in next_item:
            key = curr_item.replace('-', '')
            if key not in params: params[key] = next_item

    np.set_printoptions(formatter={'float': lambda x: "{0:0.0f}".format(x)})

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Process Image
    datum = op.Datum()
    # imageToProcess = cv2.imread(args[0].image_path)

    camera = cv2.VideoCapture('http://username:password@43.245.206.104:10001/video') #username password have been redacted
    # camera = cv2.VideoCapture('./tv/sample 1.avi')
    fc = 0

    # mySocket.sendall("init".encode())

    while True:
        startTime = time.time()
        ret, frame = camera.read()
        if ret:
            fc += 1
            print("frame " + str(fc))
        else:
            break
        datum.cvInputData = frame
        opWrapper.emplaceAndPop([datum])
        print("fps: ", end='')
        print(1 / (time.time() - startTime))

        if datum.poseKeypoints.size == 1:
            data = "no person"
        else:
            data = packageArray(datum.poseKeypoints)

        # get message size
        dtSize = len(str(data))

        mySocket.sendall(str(dtSize).rjust(7).encode())
        # print("sent length:" + (str(dtSize).rjust(7)))
        mySocket.sendall(data.encode())

        # print("Face keypoints: \n" + str(datum.faceKeypoints))
        # print("Left hand keypoints: \n" + str(datum.handKeypoints[0]))
        # print("Right hand keypoints: \n" + str(datum.handKeypoints[1]))
        # cv2.imshow("OpenPose 1.5.1 - Tutorial Python API", datum.cvOutputData)
        cv2.waitKey(1)
except Exception as e:
    print(e)
    sys.exit(-1)

print('done')
