### OutLine of the Algorithm ###

# 1)Capture image frames from your webcam video stream using open cv
# 2)Using Cascade Classifier from open cv detect faces and bound them using rectangles.
# 3)Using the coordinates of these rectangles extract all the faces present in the image and store them in the list.
# 4)Select the largest face image from the list,resize it to 100 x 100.
# 5)Store every 10th image of face into a list.
# 6)Flatten the 100 x 100 x 3 image into a single dimension of 30,000.
# 7)These images will act as training data for face recognition.
# 8)Store them in .npy format for later use.


#importing necessary libraries.
import cv2
import numpy as np

capture = cv2.VideoCapture(0) #Captures the video from your device's webcam

face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml") #The xml file contains the pre-trained weight/features of a cascading classifier trained on thousands of positive and negative face images.

face_data=[]

count=0 #TO display number of images collected.

dataset_path='./data/' #directory to store the training data

file_name=input("Enter the name of the person:")

while True:

	ret,frame=capture.read()

	if(ret==False):
		continue

	faces=face_cascade.detectMultiScale(frame,1.3,5) #This method can detect mulitple faces returns each faces coordinates in form of a tuple

	faces=sorted(faces,key=lambda f:f[2]*f[3],reverse=True) #Sorting the faces in reverse order according to their area.

	
	x_coord,y_coord,width,height=faces[0] #Pick only the image with the largest face.

	cv2.rectangle(frame,(x_coord,y_coord),(x_coord+width,y_coord+height),(255,0,0),2) #This method draws a rectangle around the face

	offset=10 #The boundary to consider while slicing the face from the image.

	face_section=frame[y_coord-offset : y_coord+height+offset , x_coord-offset : x_coord+width+offset] #Extacting the face from image.


	face_section=cv2.resize(face_section,(100,100)) #Resize the image for uniformity.

	

	count=count+1
	if count%10==0: #Select every 10th image
		face_data.append(face_section)

		print(len(face_data),"Images Collected")

		
		


	cv2.imshow("VideoFrame",frame) #Show the whole image.
	cv2.imshow("Face_Section",face_section) #Show the face.

	key_pressed=cv2.waitKey(1) & 0xFF #Convert the 32 bit output from the waitKey method to a 8 bit number because ascii value of char lies between [0,255]
	if key_pressed== ord('q'): #If the letter pressed is q stop collecting images and terminate.
		break

face_data= np.asarray(face_data)


face_data=face_data.reshape((face_data.shape[0],-1)) #Flattening of each image.

print(face_data.shape)

np.save(dataset_path+file_name+'.npy',face_data) #Save the training data to the destination folder

print("data succesfully stored")

capture.release() #Stop capturing the video stream from webcam and close all the windows

cv2.destroyAllWindows()


