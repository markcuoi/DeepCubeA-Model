###############################################
##      Open CV and Numpy integration        ##
###############################################
import pickle

import pyrealsense2 as rs
import numpy as np
import cv2
from itertools import *
from scipy import stats
from datetime import datetime
import matplotlib.pyplot as plt

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# Section for color
lower_white = np.array([130, 130, 130])
upper_white = np.array([255, 255, 255])

lower_yellow = np.array([5, 130, 90])
upper_yellow = np.array([120, 255, 230])

lower_green = np.array([0, 90, 0])  # 135 mid
upper_green = np.array([130, 220, 45])

lower_blue = np.array([100, 20, 0])
upper_blue = np.array([255, 150, 30])

lower_orange = np.array([0, 20, 160])
upper_orange = np.array([120, 135, 255])

lower_red = np.array([0, 0, 50])
upper_red = np.array([99, 99, 155])


def detect_face(img):

    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # kernel = np.ones((2, 2), np.uint8)
    # gray = cv2.dilate(gray, kernel, iterations=1)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1,1))
    # gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    # gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    # gray = cv2.erode(gray,kernel,iterations = 4)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)) #(8,8)
    # gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    #gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    gray = cv2.morphologyEx(gray, cv2.MORPH_ELLIPSE, kernel)
    gray = cv2.adaptiveThreshold(gray,10,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,21,5)

    #cv2.imshow('dilation', gray)
    contours, hierarchy = cv2.findContours(gray,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)

    i = 0
    contour_id = 0
    count = 0
    blob_colors = []

    for contour in contours:
        A1 = cv2.contourArea(contour)
        contour_id = contour_id + 1
        #7000 and 500
        if A1 < 5000 and A1 > 800:

            perimeter = cv2.arcLength(contour, True)
            epsilon = 0.01 * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            hull = cv2.convexHull(contour)  # show the images

            if cv2.norm(((perimeter / 4) * (perimeter / 4)) - A1) < 150:
                # if cv2.ma
                count = count + 1
                x, y, w, h = cv2.boundingRect(contour)
                val = (50 * y) + (10 * x)

                blob_color = np.array(cv2.mean(img[y:y + h, x:x + w])).astype(int) # avarage color RGB
                cv2.drawContours(img, [contour], -1, (209, 206, 0), 3)
                cv2.drawContours(img, [approx], -1, (209, 206, 0), 3)
                # cv2.drawContours(img,[contour],0,(255, 255, 0),2)
                # cv2.drawContours(img, [approx], 0, (255, 255, 0), 2)
                blob_color = np.append(blob_color, val)
                blob_color = np.append(blob_color, x)
                blob_color = np.append(blob_color, y)

                blob_color = np.append(blob_color, w)
                blob_color = np.append(blob_color, h)

                blob_colors.append(blob_color)

    if len(blob_colors) > 0:
        #print(blob_colors)
        blob_colors = np.asarray(blob_colors)
        blob_colors = blob_colors[blob_colors[:, 4].argsort()]
    face = np.array([0,0,0,0,0,0,0,0,0])
    if len(blob_colors) == 9:
        for i in range(9):
            if (blob_colors[i][0] >= lower_white[0] and blob_colors[i][0] <= upper_white[0]) and \
                    (blob_colors[i][1] >= lower_white[1] and blob_colors[i][1] <= upper_white[1]) and \
                    (blob_colors[i][2] >= lower_white[2] and blob_colors[i][2] <= upper_white[2]):
                blob_colors[i][3] = 1
                face[i] = 1
                #print('white')
            elif (blob_colors[i][0] >= lower_yellow[0] and blob_colors[i][0] <= upper_yellow[0]) and \
                    (blob_colors[i][1] >= lower_yellow[1] and blob_colors[i][1] <= upper_yellow[1]) and \
                    (blob_colors[i][2] >= lower_yellow[2] and blob_colors[i][2] <= upper_yellow[2]):
                blob_colors[i][3] = 2
                face[i] = 2
                #print('yellow')
            elif (blob_colors[i][0] >= lower_green[0] and blob_colors[i][0] <= upper_green[0]) and \
                    (blob_colors[i][1] >= lower_green[1] and blob_colors[i][1] <= upper_green[1]) and \
                    (blob_colors[i][2] >= lower_green[2] and blob_colors[i][2] <= upper_green[2]):
                blob_colors[i][3] = 3
                face[i] = 3
                #print('green')
            elif (blob_colors[i][0] >= lower_blue[0] and blob_colors[i][0] <= upper_blue[0]) and \
                    (blob_colors[i][1] >= lower_blue[1] and blob_colors[i][1] <= upper_blue[1]) and \
                    (blob_colors[i][2] >= lower_blue[2] and blob_colors[i][2] <= upper_blue[2]): #135 122
                blob_colors[i][3] = 4
                face[i] = 4
                #print('blue')
            elif (blob_colors[i][0] >= lower_orange[0] and blob_colors[i][0] <= upper_orange[0]) and \
                    (blob_colors[i][1] >= lower_orange[1] and blob_colors[i][1] <= upper_orange[1]) and \
                    (blob_colors[i][2] >= lower_orange[2] and blob_colors[i][2] <= upper_orange[2]):
                blob_colors[i][3] = 5
                face[i] = 5
                #print('orange')
            elif (blob_colors[i][0] >= lower_red[0]  and blob_colors[i][0] <= upper_red[0]) and \
                    (blob_colors[i][1] >= lower_red[1] and blob_colors[i][1] <= upper_red[1]) and \
                    (blob_colors[i][2] >= lower_red[2] and blob_colors[i][2] <= upper_red[2]):
                blob_colors[i][3] = 6
                face[i] = 6
                #print('red')

        #print(face)
        if np.count_nonzero(face) == 9:
            return face, blob_colors
        else:
            return [0,0], blob_colors
    else:
        return [0,0,0], blob_colors
        #break

def find_face(text):
    faces = []
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        img = np.asanyarray(color_frame.get_data())
        #display(img)
        img = cv2.putText(img, text, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        face, blob_colors = detect_face(img)
        if len(face) == 9:
            faces.append(face)
            if len(faces) == 20:
                face_array = np.array(faces)
                detected_face = stats.mode(face_array)[0]
                detected_face = np.asarray(detected_face)
                detected_face = np.squeeze(detected_face).tolist()
                #print(detected_face)
                #cv2.imwrite("filename.png", img)
                faces.clear()
                return detected_face
        cv2.imshow('plot2',img)
        cv2.waitKey(1)

def change_color_id(temp):
    changed_list_id = temp.copy()
    changed_list_id[0] = temp[6]
    changed_list_id[1] = temp[3]
    changed_list_id[2] = temp[0]
    changed_list_id[3] = temp[7]
    changed_list_id[4] = temp[4]
    changed_list_id[5] = temp[1]
    changed_list_id[6] = temp[8]
    changed_list_id[7] = temp[5]
    changed_list_id[8] = temp[2]
    return changed_list_id

def num_to_string(temp):
    changed_list = temp.copy()
    for i in range(len(temp)):
        if temp[i] == 1:
            changed_list[i] = 'white'
        elif temp[i] == 2:
            changed_list[i] = 'yellow'
        elif temp[i] == 3:
            changed_list[i] = 'green'
        elif temp[i] == 4:
            changed_list[i] = 'blue'
        elif temp[i] == 5:
            changed_list[i] = 'orange'
        elif temp[i] == 6:
            changed_list[i] = 'red'
    return changed_list


def cam_stream():
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        img = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8)) #(8,8)
        # gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        #gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        gray = cv2.morphologyEx(gray, cv2.MORPH_ELLIPSE, kernel)
        gray = cv2.adaptiveThreshold(gray,10,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,21,4)

        #cv2.imshow('dilation', gray)
        contours, hierarchy = cv2.findContours(gray,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)

        i = 0
        contour_id = 0
        count = 0
        blob_colors = []

        for contour in contours:
            A1 = cv2.contourArea(contour)
            contour_id = contour_id + 1
            #7000 and 500
            if A1 < 5000 and A1 > 600:

                perimeter = cv2.arcLength(contour, True)
                epsilon = 0.01 * perimeter
                approx = cv2.approxPolyDP(contour, epsilon, True)
                hull = cv2.convexHull(contour)  # show the images

                if cv2.norm(((perimeter / 4) * (perimeter / 4)) - A1) < 150:
                    # if cv2.ma
                    count = count + 1
                    x, y, w, h = cv2.boundingRect(contour)
                    val = (50 * y) + (10 * x)

                    blob_color = np.array(cv2.mean(img[y:y + h, x:x + w])).astype(int) # avarage color RGB
                    cv2.drawContours(img, [contour], -1, (209, 206, 0), 3)
                    cv2.drawContours(img, [approx], -1, (209, 206, 0), 3)
        cv2.imshow('plot2',img)
        cv2.waitKey(1)

def get_state(total):
    while True:
        temp_num = find_face(text="")
        change_id = change_color_id(temp_num)
        temp=num_to_string(change_id)
        print(total)
        if ((total[0] !=0) and (total[9] !=0) and(total[18] !=0) and(total[27] !=0) and(total[36] !=0) and (total[45] !=0)) :
            output(total)
            total = [0 for i in total if i !=0]
            #print('total check ', total)


        elif temp[4] == 'white':
            # total = [0 for i in total if i !=0]
            for index, i in enumerate(total):
                if (index >=9 and index<=53):
                    total[index] = 0

            white = temp
            total[0:9] = white
            #total = [0 for i in total[9:54] if i !=0]

            print('this is white',white)
        elif temp[4] == 'yellow':
            yellow = temp
            total[9:18] = yellow
            print('this is yellow') #            print('this is yellow',yellow)
        elif temp[4] == 'green':
            green = temp
            total[18:27] = green
            print('this is green')
        elif temp[4] == 'blue':
            blue = temp
            total[27:36] = blue
            print('this is blue')
        elif temp[4] == 'orange':
            orange = temp
            total[36:45] = orange
            print('this is orange')
        elif temp[4] == 'red':
            red = temp
            total[45:54] = red
            print('this is red')

        print(total)
        # MIDDLE COLOR CHECK: check the middle color and the corresponding color array
    print(total)
    print("\n\n")

def process():
    faces=[]
    total = 54*[0]
    white = []
    yellow = []
    blue = []
    green = []
    orange = []
    red = []
    print('hi process')
    get_state(total)


def output(total):
    corner4 = list(permutations(['white','blue','orange']))
    corner3 = list(permutations(['white','blue','red']))
    corner2 = list(permutations(['white','green','orange']))
    corner1 = list(permutations(['white','green','red']))

    corner8 = list(permutations(['yellow','blue','red']))
    corner7 = list(permutations(['yellow','blue','orange']))
    corner6 = list(permutations(['yellow','green','red']))
    corner5 = list(permutations(['yellow','green','orange']))

    side3 = list(permutations(['white','blue']))
    side4 = list(permutations(['white','red']))
    side1 = list(permutations(['white','green']))
    side2 = list(permutations(['white','orange']))

    side7 = list(permutations(['blue','orange']))
    side8 = list(permutations(['blue','red']))
    side5 = list(permutations(['green','red']))
    side6 = list(permutations(['green','orange']))

    side11 = list(permutations(['yellow','blue']))
    side12 = list(permutations(['yellow','orange']))
    side9 = list(permutations(['yellow','green']))
    side10 = list(permutations(['yellow','red']))

    # CREATE CORNERS AND SIDES LIST:

    corners = []
    sides = []

    # FILL IN THE CORNERS AND SIDES:
    corners.append([total[0], total[26], total[47]])
    corners.append([total[2], total[20], total[44]])
    corners.append([total[6], total[29], total[53]])
    corners.append([total[8], total[35], total[38]])

    corners.append([total[9], total[18], total[42]])
    corners.append([total[11], total[24], total[45]])
    corners.append([total[15], total[33], total[36]])
    corners.append([total[17], total[27], total[51]])


    sides.append([total[1], total[23]])
    sides.append([total[5], total[41]])
    sides.append([total[7], total[32]])
    sides.append([total[3], total[50]])

    sides.append([total[25], total[46]])
    sides.append([total[19], total[43]])
    sides.append([total[34], total[37]])
    sides.append([total[28], total[52]])

    sides.append([total[10], total[21]])
    sides.append([total[14], total[48]])
    sides.append([total[16], total[30]])
    sides.append([total[12], total[39]])

    print('Corners: ',corners)
    print('Sides: ',sides)
    #for corner in corners:
        #print(corner, 'hey this is print')

    for corner in corners:
        temp = (corner[0],corner[1],corner[2])
        if (temp in corner1):
            for i in range(3):
                if corner[i] == 'white':
                    corner[i] = 0
                elif corner[i] == 'green':
                    corner[i] = 26
                else:
                    corner[i] = 47
            #print(total)
        if (temp in corner2):
            for i in range(3):
                if corner[i] == 'white':
                    corner[i] = 2
                elif corner[i] == 'green':
                    corner[i] = 20
                else:
                    corner[i] = 44
            #print(total)
        if (temp in corner3):
            for i in range(3):
                if corner[i] == 'white':
                    corner[i] = 6
                elif corner[i] == 'blue':
                    corner[i] = 29
                else:
                    corner[i] = 53
            #print(total)
        if (temp in corner4):
            for i in range(3):
                if corner[i] == 'white':
                    corner[i] = 8
                elif corner[i] == 'blue':
                    corner[i] = 35
                else:
                    corner[i] = 38
            #print(total)

        if (temp in corner5):
            for i in range(3):
                if corner[i] == 'yellow':
                    corner[i] = 9
                elif corner[i] == 'green':
                    corner[i] = 18
                else:
                    corner[i] = 42
            #print(total)

        if (temp in corner6):
            for i in range(3):
                if corner[i] == 'yellow':
                    corner[i] = 11
                elif corner[i] == 'green':
                    corner[i] = 24
                else:
                    corner[i] = 45
            #print(total)

        if (temp in corner7):
            for i in range(3):
                if corner[i] == 'yellow':
                    corner[i] = 15
                elif corner[i] == 'blue':
                    corner[i] = 33
                else:
                    corner[i] = 36
            #print(total)

        if (temp in corner8):
            for i in range(3):
                if corner[i] == 'yellow':
                    corner[i] = 17
                elif corner[i] == 'blue':
                    corner[i] = 27
                else:
                    corner[i] = 51
            #print(total)
    for side in sides:
        temp = (side[0], side[1])
        if temp in side1:
            for i in range(2):
                if side[i] == 'white':
                    side[i] = 1
                else:
                    side[i] = 23

        if temp in side2:
            for i in range(2):
                if side[i] == 'white':
                    side[i] = 5
                else:
                    side[i] = 41

        if temp in side3:
            for i in range(2):
                if side[i] == 'white':
                    side[i] = 7
                else:
                    side[i] = 32

        if temp in side4:
            for i in range(2):
                if side[i] == 'white':
                    side[i] = 3
                else:
                    side[i] = 50

        if temp in side5:
            for i in range(2):
                if side[i] == 'green':
                    side[i] = 25
                else:
                    side[i] = 46

        if temp in side6:
            for i in range(2):
                if side[i] == 'green':
                    side[i] = 19
                else:
                    side[i] = 43

        if temp in side7:
            for i in range(2):
                if side[i] == 'blue':
                    side[i] = 34
                else:
                    side[i] = 37

        if temp in side8:
            for i in range(2):
                if side[i] == 'blue':
                    side[i] = 28
                else:
                    side[i] = 52

        if temp in side9:
            for i in range(2):
                if side[i] == 'yellow':
                    side[i] = 10
                else:
                    side[i] = 21

        if temp in side10:
            for i in range(2):
                if side[i] == 'yellow':
                    side[i] = 14
                else:
                    side[i] = 48

        if temp in side11:
            for i in range(2):
                if side[i] == 'yellow':
                    side[i] = 16
                else:
                    side[i] = 30

        if temp in side12:
            for i in range(2):
                if side[i] == 'yellow':
                    side[i] = 12
                else:
                    side[i] = 39


    total[0], total[26], total[47] = corners[0]
    total[2], total[20], total[44] = corners[1]
    total[6], total[29], total[53] = corners[2]
    total[8], total[35], total[38] = corners[3]

    total[9], total[18], total[42] = corners[4]
    total[11], total[24], total[45] = corners[5]
    total[15], total[33], total[36] = corners[6]
    total[17], total[27], total[51] = corners[7]

    total[1], total[23] = sides[0]
    total[5], total[41] = sides[1]
    total[7], total[32] = sides[2]
    total[3], total[50] = sides[3]

    total[25], total[46] = sides[4]
    total[19], total[43] = sides[5]
    total[34], total[37] = sides[6]
    total[28], total[52] = sides[7]

    total[10], total[21] = sides[8]
    total[14], total[48] = sides[9]
    total[16], total[30] = sides[10]
    total[12], total[39] = sides[11]

    total[4], total[13], total[22], total[31], total[40], total[49] = 4, 13, 22, 31, 40, 49
    # for i in total:
    #     if isinstance(i,str):
    #         get_state(total)
    #     else:
    #         break


    print('get state successfulllllllllllllllllllllllllllllllllllllllll')
    clear = open("/home/mark/Desktop/Cube/Rubik/Rubik/model_datas/cube_test.txt","r+")
    clear.truncate(0)
    clear.close()
    total_temp = []
    for i in total:
        total_temp.append(i)
    total_temp = map(str, total_temp)
    total_temp = ",".join(total_temp)
    with open("/home/mark/Desktop/Cube/Rubik/Rubik/model_datas/cube_test.txt", "a+") as write_output:
        write_output.write(str(total_temp))
        write_output.close()
    print(total)
    #print('Sorted: ',sorted(total))
    filename = '/home/mark/Desktop/Cube/Rubik/Rubik/data/cube3/handing/samples/test.pkl'

    with open(filename, 'wb') as f:
        pickle.dump(total, f)

def main():

    process()


if __name__ == "__main__":
    main()
