### Outline of the Algorithm ###

# 1)Retrieve the training data which was generated earlier. Generate Class ID's for each person whose data has been collected.
# 2)Using a dictionary map the name of each person's whose data has been collected to a unique class ID.
# 2)Concatenate the training data and the training labels.
# 3)Extract the face from the image this will act as the test data.
# 4)Call the KNN algorithm to get the predicted class ID for that particular test image.
# 5)Get the name of the person using the predicted Class ID
# 6)Draw a rectangle and write the predicted name using built in functions of open-cv.
# 7)Terminate the program if user presses the 'q' button.




#importing necessary libraries
import cv2
import numpy as np 
import os

######## K Nearest Neighbour Algorithm CODE ############

def distance(v1,v2):
	#Euclidean
	return np.sqrt(((v1-v2)**2).sum())

def knn(train,test,k=5):
	dist=[]

	for i in range(train.shape[0]):
		ix =train[i,:-1] #Training data
		iy=train[i,-1] #Labels for Training data
		#compute distance from test point
		d=distance(test,ix)
		dist.append([d,iy])
	#Sort based on the distance and get top k

	dk = sorted(dist,key=lambda x: x[0])[:k]
	#Retrieve only labels
	labels=np.array(dk)[:,-1]

	#Get frequencies of each label
	output= np.unique(labels,return_counts=True)
	#Find max frequency and corresponding label
	index=np.argmax(output[1])
	return output[0][index]

########################################################

capture = cv2.VideoCapture(0)

face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

face_data=[]

labels=[] #labels for the given file
names={} #Mapping between id and name of the person

class_id=0

dataset_path='./data/'

#data preparation step:

for file in os.listdir(dataset_path):

	if file.endswith('.npy'):

		names[class_id]=file[:-4] #get the name of the person from file's name
		print("Loaded"+file)
		data_item=np.load(dataset_path+file)
		face_data.append(data_item) #training data
		#create labels for each class
		target=class_id*np.ones((data_item.shape[0])) #each file contains multiple flattened images.
		class_id+=1 
		labels.append(target)


Training_Matrix=np.concatenate(face_data,axis=0)
Labels_Matrix=np.concatenate(labels,axis=0).reshape((-1,1))

Training_set=np.concatenate((Training_Matrix,Labels_Matrix),axis=1)

while True:

	ret,frame=capture.read()

	if(ret==False):
		continue

	faces = face_cascade.detectMultiScale(frame,1.3,5)

	for (x_coord,y_coord,width,height) in faces:

		offset=10
		face_section=frame[y_coord-offset : y_coord+height+offset , x_coord-offset : x_coord+width+offset]
		face_section=cv2.resize(face_section,(100,100))


		pred_class_index=knn(Training_set,face_section.flatten())

		pred_name=names[int(pred_class_index)] 

		cv2.putText(frame,pred_name,(x_coord,y_coord-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2,cv2.LINE_AA)

		cv2.rectangle(frame,(x_coord,y_coord),(x_coord+width,y_coord+height),(0,255,255),2)

	cv2.imshow("Faces",frame)

	key=cv2.waitKey(1) & 0xFF
	if key==ord('q'):
		break

capture.release()
cv2.destroyAllWindows()
