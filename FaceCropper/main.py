import cv2
import os

def getLargestFace(faces):
    largestArea = 0
    largestFace = faces

    for (x, y, w, h) in faces:
        area = (w-x) * (h-y)
        if area > largestArea:
            largestArea = area
            largestFace = [[x, y, w, h]]

    return largestFace



directory = "Uncropped\\Cloth"
# = 3600 #CAN BE USED TO START AT A CERTAIN POSITION

outputFolders = ["None","N95","Surgical","Cloth"] #MAKE THESE FOLDERS IN THE IMAGE PATH
counts = [0,0,0,0]

list = os.listdir(directory)


for i in range(len(list)):
    filename = list[i]
    # Read the input image
    imagePath = directory+ "\\" + filename
    img = cv2.imread(imagePath)
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Load the cascade
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw rectangle around the faces and crop the faces
    #biggestFace = getLargestFace(faces)


    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        #faces = img[y:y + h, x:x + w]
        #cv2.imshow("face", faces)
        #biggestFace = img[y:y + h, x:x + w]
        #cv2.imshow("BiggestFace", biggestFace)
        #cv2.imwrite('face.jpg', faces)



    # Display the output
    #cv2.imwrite('detected.jpg', img)
    cv2.imshow('img', img)
    cv2.waitKey()






