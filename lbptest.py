# import the necessary packages
import numpy as np
import cv2
from skimage import feature
from sklearn.svm import LinearSVC
from imutils import paths
import argparse
class LocalBinaryPatterns:
        def __init__(self, numPoints, radius):

                # store the number of points and radius

                self.numPoints = numPoints

                self.radius = radius

        def describe(self, image, eps=1e-7):

                # compute the Local Binary Pattern representation

                # of the image, and then use the LBP representation

                # to build the histogram of patterns

                lbp = feature.local_binary_pattern(image, self.numPoints,

                        self.radius, method="uniform")

                (hist, _) = np.histogram(lbp.ravel(),

                        bins=np.arange(0, self.numPoints + 3),

                        range=(0, self.numPoints + 2))

                # normalize the histogram

                hist = hist.astype("float")

                hist /= (hist.sum() + eps)

                # return the histogram of Local Binary Patterns

                return hist
# import the necessary packages


# initialize the local binary patterns descriptor along with

# the data and label lists

desc = LocalBinaryPatterns(24, 8)

data = []

labels = []

result = []
A = 0.0
B = 0.0
C = 0.0
D = 0.0
AT = 0.0
BT = 0.0
CT = 0.0
DT = 0.0
# loop over the training images
counter = 0
for imagePath in paths.list_images("H:/python/doll_test/"):

        # load the image, convert it to grayscale, and describe it

        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = desc.describe(gray)
        # extract the label from the image path, then update the
        strr = imagePath.split("/")[-1]
        # label and data lists
        if(strr[0] == "A" ):
                labels.append("A")
                A = A + 1 
        elif(strr[0] == "B" ):
                labels.append("B")
                B = B + 1
        elif(strr[0] == "C" ):
                labels.append("C")
                C = C + 1
        elif(strr[0] == "D" ):
                labels.append("D")
                D = D + 1
        data.append(hist)

# train a Linear SVM on the data
print(labels)
print("A=>" + str(A) + "  B=>" + str(B) + "  C=>" + str(C) + "  D=>" + str(D))
model = LinearSVC(C=50.0, random_state=42,max_iter = 3000)

model.fit(data, labels)
# loop over the testing images

for imagePath in paths.list_images("H:/python/doll_test"):

        # load the image, convert it to grayscale, describe it,

        # and classify it

        image = cv2.imread(imagePath)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        hist = desc.describe(gray)
        
        hist = hist.reshape(-1, 1)
        hist = hist.reshape(1, -1)
        prediction = model.predict(hist)[0]

        # display the image and the prediction
        if(prediction == "A" ):
                if(prediction == labels[counter]):
                        AT = AT + 1 
        elif(prediction == "B" ):
                if(prediction == labels[counter]):
                        BT = BT + 1
        elif(prediction == "C" ):
                if(prediction == labels[counter]):
                        CT = CT + 1
        elif(prediction == "D" ):
                if(prediction == labels[counter]):
                        DT = DT + 1

        counter = counter + 1
        result.append(prediction)
        # cv2.putText(image, prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,

        #         1.0, (0, 0, 255), 3)
        # cv2.imshow("Image", image)

print("AT=>" + str(AT / A) + "  BT=>" + str(BT / B) + "  CT=>" + str(CT / C) + "  DT=>" + str(DT / D))
print("TOTAL=>" + str((AT + BT + CT + DT)/(A + B + C + D)))
cv2.waitKey(0)