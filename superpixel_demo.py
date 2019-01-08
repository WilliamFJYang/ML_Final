import cv2
import numpy as np
import os
from imutils import paths
file_count = sum(len(files) for _, _, files in os.walk("H:/AAAA"))
num_of_superpixels_result = 100
ADD_reg = [[],[],[]]
ADDr = [[],[]]
print(file_count)
counter = 0
grid_size = 120
flag = 0
for cnt in range (1,file_count+1):
    np.set_printoptions(threshold = np.inf)
    # image read
    img = cv2.imread("H:/AAAA/S("+str(cnt)+").jpg") # BRG order, uint8 type
    print("H:/AAAA/T ("+str(cnt)+").jpg")
    rows,cols,vector = img.shape
    img = cv2.resize(img,(720,int(720*(rows/cols))))
    rows,cols,vector = img.shape
    # print(rows,cols)
    textimg = np.zeros((rows,cols,3),dtype="uint8")
    ##############################################
    # (B,G,R) = cv2.split(img)
    # MB = np.mean(B)
    # MG = np.mean(G)
    # MR = np.mean(R)
    # KB = np.zeros(B.shape,dtype = np.float32)
    # KB.fill((MR + MG + MB) / (3 * MB))
    # KG = np.zeros(G.shape,dtype = np.float32)
    # KG.fill((MR + MG + MB) / (3 * MG))
    # KR = np.zeros(R.shape,dtype = np.float32)
    # KR.fill((MR + MG + MB) / (3 * MR))
    # KB = B * KB
    # KB[KB >255] = 255
    # B = KB.astype(np.uint8)
    # KG = G * KG
    # KG[KG >255] = 255
    # G = KG.astype(np.uint8)
    # KR = R * KR
    # KR[KR >255] = 255
    # R = KR.astype(np.uint8)
    # img = cv2.merge([B,G,R])
    ##############################################
    # cv2.imshow('ImageWindow', img)

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
    seeds = cv2.ximgproc.createSuperpixelSLIC(converted_img,100,75,75.0)


    # run SEEDS
    seeds.iterate()

    # get number of superpixel
    num_of_superpixels_result = seeds.getNumberOfSuperpixels()
    print('Final number of superpixels: %d' % num_of_superpixels_result)


    # retrieve the segmentation result
    labels = seeds.getLabels() # heigqht x width matrix. Each component indicates the superpixel index of the corresponding pixel position
    # print(labels)
    ADD_reg = [[0 for i in range(3)] for i in range(num_of_superpixels_result)]
    ADDr    = [[0 for i in range(2)] for i in range(num_of_superpixels_result)]
    for i in range (0 , rows):
        for j in range (0 , cols):
            ADD_reg[int(labels[i][j])][0] = ADD_reg[int(labels[i][j])][0] + 1
            ADD_reg[int(labels[i][j])][1] = ADD_reg[int(labels[i][j])][1] + j
            ADD_reg[int(labels[i][j])][2] = ADD_reg[int(labels[i][j])][2] + i 

    for k in range (0 , num_of_superpixels_result):
        ADDr[k][0] = int(float(ADD_reg[k][1]) / ADD_reg[k][0])
        ADDr[k][1] = int(float(ADD_reg[k][2]) / ADD_reg[k][0])

    counter = 0
    grid_size = 120
    flag = 0
    ############################################################################################################################################
    while (flag != 1):
        for k in range (0 , num_of_superpixels_result):
            if (ADDr[k][0] - grid_size / 2 >= 0 and ADDr[k][0] + grid_size / 2 < rows) and (ADDr[k][1] - grid_size / 2 >= 0 and ADDr[k][1] + grid_size / 2 < cols):
                counter = counter + 1
                print(str(counter) + "==> X " + str(ADDr[k][0] - int(grid_size / 2)) + ':' + str(ADDr[k][0] + int(grid_size / 2 - 1)) + '  Y ' + str(ADDr[k][1] - int(grid_size / 2)) + ':' + str(ADDr[k][1] + int(grid_size / 2 - 1) ))
                crop_img = img[ADDr[k][0] - int(grid_size / 2 ): ADDr[k][0] + int(grid_size / 2 - 1), ADDr[k][1] - int(grid_size / 2) : ADDr[k][1] + int(grid_size / 2 - 1)]              
                crop_img = cv2.resize(crop_img,(120,120))
                cv2.imwrite("H:/AAAA/TAAA/"+ str(cnt) + "_(" + str(counter) +").jpg",crop_img)
                # cv2.imshow('123', crop_img)
                # cv2.waitKey(0)
        grid_size = int(grid_size * 1.25)
        if(grid_size > rows or grid_size > cols):
            flag = 1
    ############################################################################################################################################

    # draw contour
    mask = seeds.getLabelContourMask(True)

#     # cv2.imshow('MaskWindow', mask)
#     # cv2.waitKey(0)


    # draw color coded image
    color_img = np.zeros((height, width, 3), np.uint8)
    color_img[:] = (0, 0, 255)
    mask_inv = cv2.bitwise_not(mask)
    result_bg = cv2.bitwise_and(img, img, mask=mask_inv)
    result_fg = cv2.bitwise_and(color_img, color_img, mask=mask)
    result = cv2.add(result_bg, result_fg)
    ###############################################
    for k in range (0 , num_of_superpixels_result):
        cv2.putText(textimg, str(k), (ADDr[k][0], ADDr[k][1]), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
    result = cv2.add(result , textimg)
    ###############################################
    cv2.imshow('ColorCodedWindow', result)
    cv2.waitKey(0)
# cv2.destroyAllWindows()