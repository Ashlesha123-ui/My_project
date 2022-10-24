import sys
import os
import cv2 as cv
import numpy as np


def main(argv):
    ## [load]##
    default_file =  r'C:\Users\yashasbd\Music\Medium pipe final\Input\3.jpg'
    path_list = default_file.split(os.sep)
    #print (path_list)
    #print (path_list[2])
    outimg = path_list[2]
    print(outimg)
    filename = argv[0] if len(argv) > 0 else default_file
    print(filename)
    # Loads an image
    src = cv.imread(filename, cv.IMREAD_COLOR)
    

    # Check if image is loaded fine##
    if src is None:
        print ('Error opening image!')
        print ('Usage: hough_circle.py [image_name -- default ' + default_file + '] \n')
        return -1
       
    ## [load]##
    ## [convert_to_gray]##
    ## [Convert it to gray]##
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    ## [convert_to_gray]##
    ## [reduce_noise]##
    ## Reduce the noise to avoid false circle detection##
    gray = cv.GaussianBlur(gray,(5,5),0)
    cv.BORDER_DEFAULT
    gray = cv.medianBlur(gray, 5)
    gray = cv.bilateralFilter(gray,5,5,5)
    th2 = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_MEAN_C,\
                    cv.THRESH_BINARY,11, 2)
    rows = gray.shape[0]
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows /100,
                               param1=100, param2=30,
                               minRadius=1, maxRadius=40)
    
    if circles is not None:
       circles = np.uint16(np.around(circles))
       number=1
       font = cv.FONT_HERSHEY_SIMPLEX
       height= 40
       
    for cont, i in enumerate(circles, start=1):
      circles = np.round(circles[0,:]).astype("int")
      circles = sorted(circles, key=lambda v: [v[0], v[1]])

      NUM_ROWS = 1000
      sorted_cols = []
    for k in range(0, len(circles), NUM_ROWS):
        col = circles[k:k+NUM_ROWS]
        sorted_cols.extend(sorted(col, key=lambda v: v[1]))
        circles = sorted_cols  

    for i in circles[0: ]:
        x, y= i[0], i[1]
        number=str(number)
        numbersize= cv.getTextSize(number, font, 1, 2)[0]


        radius= i[2]
        cv.circle(src, (x, y), radius, (255, 0, 255), 3)
        cv.rectangle(src, (x - 12, y - 12), (x + 12, y + 12), (0, 128, 255), -1)   
         
        # cv.putText(src, number, (x - 5, y + 5) , font, 0.5,(255, 255, 255), -1)
        
        number=int(number)
        number+=1

    r = number-1       
    print(r)
    cv.imshow('detecting', src)
    print("i cam here.....!!!!", src)
    print()
    print()
    optimage ="./output/" +outimg
    # print (optimage)
    cv.imwrite(r'output.jpg',src)
    cv.waitKey(0)

    return 0

