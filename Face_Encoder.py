# import the necessary packages
from imutils import paths
import face_recognition
import pickle
import cv2
import os

#grab the paths to the input images in our dataset
imagePaths = list(paths.list_images('Dataset'))

#initialize the list of known encodings and known names
knownEncodings = []
knownNames = []

print ("Starting Process")

#loop over the image paths
for (i, imagePath) in enumerate(imagePaths):

	#extract the person name from the image path
	print("Processing image {}/{}".format(i + 1, len(imagePaths)))

	name = imagePath.split(os.path.sep)[-2]

	#load the input image and convert it to RGB
	image = cv2.imread(imagePath)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	#detect the (x, y)-coordinates of the bounding boxes corresponding to each face in the input image
	boxes = face_recognition.face_locations(rgb, model='hog')

	#compute the facial embedding for the face
	encodings = face_recognition.face_encodings(rgb, boxes)

	# loop over the encodings
	for encoding in encodings:
		# add each encoding + name to our set of known names and encodings
		knownEncodings.append(encoding)
		knownNames.append(name)

# dump the facial encodings + names to disk
data = {"encodings": knownEncodings, "names": knownNames}
f = open("Dataset/Encodings/encodings.pickle", "wb")
f.write(pickle.dumps(data))
f.close()
print ("Done Face Encodings")
