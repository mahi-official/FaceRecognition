# import the necessary packages
import face_recognition
import argparse
import pickle
import cv2
import imutils

path = 'Input/00002.jpg'

# load the known faces and embeddings
print("Loading Database")
data = pickle.loads(open('Dataset/Encodings/encodings.pickle', "rb").read())

# load the input image and convert it from BGR to RGB
image = cv2.imread(path)

rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
rgb = imutils.resize(image, width=1000)

#find the scaling factor
r = image.shape[1] / float(rgb.shape[1])

# detect the (x, y)-coordinates of the bounding boxes corresponding
# to each face in the input image, then compute the facial embeddings
# for each face
print("Recognizing faces...")
boxes = face_recognition.face_locations(rgb, model='hog')
encodings = face_recognition.face_encodings(rgb, boxes)

# initialize the list of names for each face detected
names = []

# loop over the facial embeddings
for encoding in encodings:
	# attempt to match each face in the input image to our known
	# encodings
	matches = face_recognition.compare_faces(data["encodings"], encoding, tolerance = 0.45)
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

		# determine the recognized face with the largest number of
		# votes (note: in the event of an unlikely tie Python will
		# select first entry in the dictionary)
		name = max(counts, key=counts.get)
	
	# update the list of names
	names.append(name)
	print (names)

# loop over the recognized faces
for ((top, right, bottom, left), name) in zip(boxes, names):
        #Rescale the face coordinates
	#top = int(top * r)
	#right = int(right * r)
	#bottom = int(bottom * r)
	#left = int(left * r)
	# draw the predicted face name on the image
	cv2.rectangle(rgb, (left, top), (right, bottom), (0, 255, 0), 1)
	y = top - 15 if top - 15 > 15 else top + 15
	cv2.putText(rgb, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
		0.75, (0, 255, 0), 1)

# show the output image
cv2.imshow("Image", rgb)
cv2.waitKey(0)
