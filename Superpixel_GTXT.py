import cv2
import numpy as np
from imutils import paths
import argparse
# image path
ap = argparse.ArgumentParser()

ap.add_argument("-l", "-LoadImg", required=True,help="path to the training images")

args = vars(ap.parse_args())

num_of_superpixels_result = 100
ADD_reg = [[],[],[]]
ADDr = [[],[]]
counter = 0
grid_size = 120
flag = 0
np.set_printoptions(threshold = np.inf)
# image read
img = cv2.imread(args["l"]) # BRG order, uint8 type
print(args["l"])
rows,cols,vector = img.shape
textimg = np.zeros((rows,cols,3),dtype="uint8")
# convert color space
converted_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# set parameters for superpixel segmentation
num_superpixels = 200  # desired number of superpixels
num_iterations = 4     # number of pixel level iterations. The higher, the better quality
prior = 2              # for shape smoothing term. must be [0, 5]
num_levels = 4
num_histogram_bins = 5 # number of histogram bins
height, width, channels = converted_img.shape
# initialize SEEDS algorithm
slic = cv2.ximgproc.createSuperpixelSLIC(converted_img,101,100,50.0)
# run SEEDS
slic.iterate()
# get number of superpixel
num_of_superpixels_result = slic.getNumberOfSuperpixels()
print('Final number of superpixels: %d' % num_of_superpixels_result)
# retrieve the segmentation result
labels = slic.getLabels() # heigqht x width matrix. Each component indicates the superpixel index of the corresponding pixel position
# print(labels)
ADD_reg = [[0 for i in range(3)] for i in range(num_of_superpixels_result)]
ADDr    = [[0 for i in range(2)] for i in range(num_of_superpixels_result)]
for i in range (0 , rows):
    for j in range (0 , cols):
        ADD_reg[int(labels[i][j])][0] = ADD_reg[int(labels[i][j])][0] + 1
        ADD_reg[int(labels[i][j])][1] = ADD_reg[int(labels[i][j])][1] + j
        ADD_reg[int(labels[i][j])][2] = ADD_reg[int(labels[i][j])][2] + i 
f = open('news.txt','w')
f.truncate();
f = open('news.txt','a')
for k in range (0 , num_of_superpixels_result):
    ADDr[k][0] = int(float(ADD_reg[k][1]) / ADD_reg[k][0])
    ADDr[k][1] = int(float(ADD_reg[k][2]) / ADD_reg[k][0])
    f.write("X" + str(ADDr[k][0]) + "Y" + str(ADDr[k][1]) + '\n')
f = open('news.txt','r')
print(f.read())
cv2.destroyAllWindows()