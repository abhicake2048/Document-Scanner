import cv2
import numpy as np
from imutils.perspective import four_point_transform
import pytesseract

cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)

pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"

count = 0
scale = 1
font = cv2.FONT_HERSHEY_SIMPLEX

wid , hei = 800 , 600
cap.set(cv2.CAP_PROP_FRAME_WIDTH,wid)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,hei)


def image_processing(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray,160,255,cv2.THRESH_BINARY)

    return threshold


def scan_detection(image):
    global document_c
    document_c = np.array([[0,0],[wid,0],[wid,hei],[0,hei]])

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    _,threshold = cv2.threshold(blur,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours,_ = cv2.findContours(threshold,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours,key=cv2.contourArea,reverse=True)

    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
            peri = cv2.arcLength(contour,True)
            approx = cv2.approxPolyDP(contour,0.015 * peri, True)
            if area > max_area and len(approx) == 4:
                document_c = approx
                max_area=area
    cv2.drawContours(frame,[document_c],-1,(0,255,0),3)


def center_text(image,text):
    text_size = cv2.getTextSize(text,font,2,5)[0]
    text_x = (image.shape[1] - text_size[0])//2
    text_y = (image.shape[0] + text_size[1])//2
    cv2.putText(image,text,(text_x,text_y), font,2,(255,0,255),5,cv2.LINE_AA)


while True:
    _,frame = cap.read()
    frame = cv2.rotate(frame,cv2.ROTATE_180)
    frame_copy= frame.copy()

    scan_detection(frame)

    cv2.imshow("input",frame)
    warped = four_point_transform(frame_copy,document_c.reshape(4,2))
    cv2.imshow('warped',warped)

    processed = image_processing(warped)
    processed = processed[10:processed.shape[0]-10 , 10:processed.shape[1] - 10]
    cv2.imshow("processed",processed)


    

    pressed_key = cv2.waitKey(1) & 0xFF

    if pressed_key == 27:
        break

    elif pressed_key == ord('s'):
        cv2.imwrite("output/scanned_" + str(count)+ ".jpg" ,processed)
        count = count + 1

        center_text(frame,"scan saved")
        cv2.imshow("input",frame)
        cv2.waitKey(500)

    elif pressed_key == ord('o'):
        file = open("output/recognized_" + str(count - 1)+ ".txt" ,"w")
        ocr_txt = pytesseract.image_to_string(warped)
        #print(ocr_txt)
        file.write(ocr_txt)
        file.close()

        center_text(frame,"text saved")
        cv2.imshow("input",frame)
        cv2.waitKey(500)


cv2.destroyAllWindows()