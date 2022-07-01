from operator import imatmul
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import resize
from tensorflow.keras.models import load_model
from tensorflow.python import framework
from sudokusolver import printing
from tensorflow.compat.v1.keras.backend import set_session
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from multiprocessing import Pipe, process
import multiprocessing
from tensorflow.keras.models import load_model
import os

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
model = load_model('Model_test_3.h5')
corner = None
pts1 = []
max_contour = 0
best_count  = 0
kernel = np.ones((4,3),np.uint8)

def video_captur(p_input,p_input2):

    cap = cv2.VideoCapture(0)

    while True:
        _,frame = cap.read()
        frame = cv2.resize(frame,(500,500))
        cv2.circle(frame,(125,400),9,(0,255,0),-1)
        cv2.circle(frame,(125,100),9,(0,255,0),-1)
        cv2.circle(frame,(475,400),9,(0,255,0),-1)
        cv2.circle(frame,(475,100),9,(0,255,0),-1)
        pts = [[ 125,400],[125,100],[475,100],[475,100]]
        #cropped = image[100:500,125:475]
        p_input.send(frame)
        p_input2.send(frame)
        
def image_process(p_output,p_input3):
    global pts1
    global corner
    while True:
        frame_ = p_output.recv()
        gray = cv2.cvtColor(frame_,cv2.COLOR_BGR2GRAY)
        blur = cv2.blur(frame_,(3,3))
        # blur_ = cv2.GaussianBlur(gray,(9,9),-1)
        # frame_ = cv2.adaptiveThreshold(blur_,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,19,2)
        cannny = cv2.Canny(blur,127,255,-1,apertureSize=3)

        contour,_ = cv2.findContours(cannny,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

        try:

            for c in contour:
                (x,y,w,h) = cv2.boundingRect(c)
                if cv2.contourArea(c)> 30000:
                    best_contour = cv2.contourArea(c)
                    cv2.rectangle(frame_,(x,y),(x+w,y+h),(0,255,0),3)
                    best_count = c
                    approx = cv2.approxPolyDP(c, 0.009 * cv2.arcLength(c, True), True)
                    if len(approx) == 4:
                        corner = approx
            try:
                n = corner.ravel()
                i = 0
                for j in n:
                    if i % 2 == 0:
                        x = n[i]
                        y = n[i+1]
                        pts1.append([x,y])
                    i = i + 1
            except Exception as e:
                print(e)
 

                #COORDINATES OF CORNERS OF SUDOKU BOARD

            pts1_ = np.float32([[pts1[1][1],pts1[1][0]],[pts1[2][1],pts1[2][0]],[pts1[0][0],pts1[0][1]],[pts1[3][1],pts1[3][0]]])

            pts1_ = np.float32([[pts1[1][0],pts1[1][1]],[pts1[0][0],pts1[0][1]],[pts1[2][0],pts1[2][1]],[pts1[3][0],pts1[3][1]]])

            pts2 = np.float32([[0,0],[500,0],[0,500],[500,500]])

            # for i in pts1:
            #     cv2.circle(frame,i,3,(0,255,255),3)
            pts1 = []

            #FOR TOP VIEW

            matrix = cv2.getPerspectiveTransform(pts1_,pts2)

            imgOutput =cv2.warpPerspective(frame_,matrix,(500,500))

            #CONVERTING IMAGEOUTPUT TO GRAYYSCALE

            imgOutput = cv2.cvtColor(imgOutput,cv2.COLOR_BGR2GRAY)
            
            kernel = np.ones((3,3), np.uint8)

            #TO THICKEN THE IMAGES

            img_dilation = cv2.dilate(imgOutput, kernel, iterations=1)

            #FOR THINNING
            img_erosion = cv2.erode(img_dilation, kernel, iterations=1)

            frame_ = cv2.adaptiveThreshold(img_erosion,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,11,3)

            #NO NEED TO CHANGE THE BELOW CODE IT SPLITS THE IMAGES AS 9X9
            edge_h = np.shape(frame_)[0]
            edge_w = np.shape(frame_)[1]
            celledge_h = edge_h // 9
            celledge_w = edge_w // 9
            tempgrid = []
            coord = []


            image_o = cv2.adaptiveThreshold(imgOutput,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,19,3)

            horizontal = np.copy(image_o)
            vertical = np.copy(image_o)

            cols = horizontal.shape[1]
            rows = horizontal.shape[0]
            vertical_size = cols//13
            horizontal_size = rows // 13

            horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size,1))
            horizontal = cv2.erode(horizontal, horizontalStructure)
            horizontal = cv2.dilate(horizontal, horizontalStructure)

            vertical_struct = cv2.getStructuringElement(cv2.MORPH_RECT, (1, horizontal_size))
            vertical  = cv2.erode(vertical, vertical_struct)
            vertical = cv2.dilate(vertical, vertical_struct)




            combined = cv2.bitwise_or(horizontal,vertical)
            combined = cv2.bitwise_not(combined)
            combined = cv2.bitwise_and(combined,image_o)
            kernel = np.ones((4,5),np.uint8)
            opening = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
            for i in range(celledge_h, edge_h + 1, celledge_h):
                for j in range(celledge_w, edge_w + 1, celledge_w):

                    rows = combined[i - celledge_h:i]
                    coord.append([i,j])
                    tempgrid.append([rows[k][j - celledge_w:j] for k in range(len(rows))])
                    
            finalgrid = []
            

            for i in range(0, len(tempgrid) - 8, 9):

                finalgrid.append(tempgrid[i:i + 9])

            for i in range(9):
                for j in range(9):
                    finalgrid[i][j] = np.array(finalgrid[i][j]) 
 

                    image = cv2.morphologyEx(finalgrid[i][j], cv2.MORPH_OPEN, kernel)
                    image  = image[15:73,25:75]
                    
                    image = cv2.erode(image,kernel,iterations = 1)
                    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
                    image = cv2.resize(image,dsize=(24,24))
                    image = cv2.copyMakeBorder(image,3,3,3,3,0)
                    kernel = np.ones((2,2),np.uint8)
                    #image = cv2.dilate(image,kernel,iterations = 1)


                    image = np.array(image).reshape(-1,30,30,1)
                    image_ = image/255
                    p_input3.send(image_)
            p_input.send(image_)
            
            

        except Exception as e:
            print(e)

def prediction(p_output3):

    image = p_output3.recv()
    y_predicted = []
    image_ = image/255
    y_predicted.append(np.argmax(model.predict(image_)))



def Show_image(p_output2):
    while True:
        image = p_output2.recv()
        cv2.imshow('frame',image)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    p_output,p_input = Pipe()
    p_output2,p_input2 = Pipe()
    p_output3,p_input3 = Pipe()

    p1 = multiprocessing.Process(target=video_captur, args=(p_input,p_input2))
    p4 = multiprocessing.Process(target=Show_image, args=(p_output,))

    p2 = multiprocessing.Process(target=image_process, args=(p_output2,p_input3))
    p3 = multiprocessing.Process(target=prediction, args=(p_output3,))
    

    p1.start()
    p2.start()
    p3.start()
    p4.start()