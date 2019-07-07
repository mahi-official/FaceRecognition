# import the necessary packages
from imutils.video import VideoStream
from threading import Thread
import pickle
import face_recognition
import imutils
import time
import cv2
import sys
import numpy as np

students = []
#Note:need to add count as a parameter for faster processing
#------------------------------------------------------------------------------------
filep = open("Attendance.csv", "w+")
filep.truncate()
filep.close()
#------------------------------------------------------------------------------------
#load the known faces and embeddings
print("Loading Data...")
data = pickle.loads(open('Dataset/Encodings/encodings.pickle', "rb").read())

# Load the webcam and give a warmup time of 2 sec.
print("Starting Webcam....")
#vs = cv2.VideoCapture(0)
vs = VideoStream(src=0).start()
time.sleep(2.0)

# loop over frames from the video file stream
while True:
	# grab the frame from the threaded video stream
	frame = vs.read()
	
	# convert the frame to RGB and scale it to 800px for faster processing
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	rgb = imutils.resize(rgb, width=800)
	#find the scaling factor
	r = frame.shape[1] / float(rgb.shape[1])

	# detect the bounding boxes to each face in then compute the facial embeddings for each face
	boxes = face_recognition.face_locations(rgb, model='hog')
	encodings = face_recognition.face_encodings(rgb, boxes)
	
        # initialize the list of names for each face detected
	names = []

	# loop over the facial embeddings
	for encoding in encodings:
		# attempt to match each face in the input image to our known
		# encodings
		matches = face_recognition.compare_faces(data["encodings"], encoding, tolerance =  0.45)
		name = "Unknown"

		# check to see if we have found a match
		if True in matches:
			# find the indexes of all matched faces then initialize a
			# dictionary to count the total number of times each face
			# was matched
			matchedIdxs = [i for (i, b) in enumerate(matches) if b]
			counts = {}

			# loop over the matched indexes and maintain a count for
			# each recognized face face
			for i in matchedIdxs:
				name = data["names"][i]
				counts[name] = counts.get(name, 0) + 1
				if counts[name] > 5:
					continue
			# determine the recognized face with the largest number
			# of votes (note: in the event of an unlikely tie Python
			# will select first entry in the dictionary)
			name = max(counts, key=counts.get)
		
		# update the list of names
		names.append(name)
		if(name != "Unknown" and name not in students):
			students.append(name)
			filep = open('Attendance.csv','a')
			filep.write(name + '\n')
			filep.close()

	# loop over the recognized faces
	for ((top, right, bottom, left), name) in zip(boxes, names):
		# rescale the face coordinates
		top = int(top * r)
		right = int(right * r)
		bottom = int(bottom * r)
		left = int(left * r)

		# draw the predicted face name on the image
		y = top - 15 if top - 15 > 15 else top + 15
		cv2.rectangle(frame, (left, top), (right, bottom),(0, 255, 0), 2)
		cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

	# check to see if we are supposed to display the output frame to
	# the screen
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
#vs.release()
vs.stop()
