from imutils.video import VideoStream
import imutils
import time
import cv2
import os

# load OpenCV's Haar cascade for face detection from disk
detector = cv2.CascadeClassifier('cascades/haarcascades/haarcascade_frontalface_default.xml')

# initialize the video capturing
cap = cv2.VideoCapture(0)
#allow the camera sensor to warm up,
time.sleep(2.00)

print ("Starting video stream...")
print ("Press 's' for saving the screenshot")
print ("Press 'q' to quit")
#vs = VideoStream(src=0).start()
#vs = VideoStream(usePiCamera=True).start()

total = 0

# loop over the frames from the video stream
while (True):
	#Read frame by frame
	ret, frame = cap.read()
	orig = frame.copy()

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	# detect faces in the grayscale frame
	faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
	
	# loop over the face detections and draw them on the frame
	for (x, y, w, h) in faces:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

	# show the output frame
	cv2.imshow('Frame', frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `s` key was pressed, write the *original* frame to disk
	# so we can later process it and use it for face recognition
	if key == ord("s"):
		cv2.imwrite('Dataset/Generator/{}.jpg'.format(str(total).zfill(5)),orig)
		total += 1

	# if the `q` key was pressed, break from the loop
	elif key == ord("q"):
		break
# print the total faces saved and do a bit of cleanup
print("{} face images stored".format(total))
print("Cleaning up...")
cap.release()
cv2.destroyAllWindows()
