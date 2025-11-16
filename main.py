import cv2
import os
import numpy as np

# creating the results folder if it doesn't exist
if not os.path.exists("results/"):
    os.makedirs("results/")

# creating the image list to store the images
imgList = []

# opening the txt file
file = open("dataset/" + "info.txt", "r")

# reading the txt file content
for line in file:
    #spliting the text line by line
    lines = line.split()
    print(line)

    x = int(lines[4])
    y = 1024 - int(lines[5])  # assigment says "This value should be subtracted from 1024"
    imgName = lines[0] #name of the img
    radius = int(lines[6]) # radius of the tumor in pixels

    imgList.append({'name': imgName, 'x': x, 'y': y, 'radius': radius})

file.close()

print("Number of images:", len(imgList))

# function for region growing with 4-connectivity

# R. Adams and L. Bischof, "Seeded region growing,"
# in IEEE Transactions on Pattern Analysis and Machine Intelligence,
# vol. 16, no. 6, pp. 641-647, June 1994, doi: 10.1109/34.295913.

# I used the algorithm from the paper above and
# I tried to implement it as a function in my code
# but instead of the priority queue I used a regular FIFO queue
# and I compared everything to the seed value to keep it simple

def region_growing_4_connectivity(img, seed_x, seed_y, threshold, max_size):

    height = img.shape[0]
    width = img.shape[1]

    # I keep visited pixels here, I fill it with zeros at the beginning
    visitedPixels= np.zeros((height, width), dtype=bool)

    # I use this to store segmented image, I fill it with zeros for now
    result = np.zeros((height, width), dtype=np.uint8)

    # intensity values at seed point
    seed = int(img[seed_y, seed_x])

    pixels = []
    pixels.append((seed_x, seed_y))
    visitedPixels[seed_y, seed_x] = True

    # loops until queue is empty or until reaches the max size
    count = 0
    while len(pixels) > 0 and count < max_size:
        # next pixel from queue
        x, y = pixels.pop()

        # checking if the current pixel intensity is close to the seed value
        if abs(int(img[y, x]) - seed) <= threshold:
            # adding pixel to the result region
            result[y, x] = 255
            count = count + 1

            # I check 4-neighbors here

            # left neighbor
            if x > 0 and not visitedPixels[y, x - 1]:
                pixels.append((x - 1, y))
                visitedPixels[y, x - 1] = True

            # right neighbor
            if x < width - 1 and not visitedPixels[y, x + 1]:
                pixels.append((x + 1, y))
                visitedPixels[y, x + 1] = True

            # top neighbor
            if y > 0 and not visitedPixels[y - 1, x]:
                pixels.append((x, y - 1))
                visitedPixels[y - 1, x] = True

            # bottom neighbor
            if y < height - 1 and not visitedPixels[y + 1, x]:
                pixels.append((x, y + 1))
                visitedPixels[y + 1, x] = True

    return result


for i in range(len(imgList)):
    imageList = imgList[i]
    name = imageList['name']
    x = imageList['x']
    y = imageList['y']
    radius = imageList['radius']

    # reading images

    # I use imread to read only .pgm files because .jpg is compressed, and we may lose important details
    img = cv2.imread("dataset/" + name + ".pgm", cv2.IMREAD_GRAYSCALE)

    # in my first step I use gaussian to remove noise from the images
    # 9 by 9 kernel worked well for this dataset
    # I also tried 3 by 3 mask, but it didn't blur the images enough
    blurred = cv2.GaussianBlur(img, (9, 9), 0)

    # secondly I used contrast stretching to enhance the intensity differences
    contrast_stretching = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)

    # So after I have done the preprocessing
    # I removed the noise with gaussian blur
    # there were some holes in some regions in segmentation before applying the gaussian blur
    # Also, contrast stretching helped to separate regions in a better way

    # and as the third step of my algorithm I used region growing function here
    # this is where the main segmentation happens

    # I choose threshold value by trying different values I started from 10,12,13,15,18,20
    # I stopped at 20
    # because at 20 in smaller tumors it started doing too much over-segmentation
    # so what I found is:
    # thresh = 20  gives the best result for larger tumors
    # thresh = 10  gives the best result for smaller tumors
    # I picked 12 to make it suitable for both cases
    # but still it makes some under-segmentation for very big tumors

    thresh = 12

    # calling region_growing_4_connectivity function
    segmented_img = region_growing_4_connectivity(contrast_stretching, x, y, thresh, img.shape[0] * img.shape[1])

    # as the final step I used closing to fill gaps I tried several kernel sizes but 12 was enough
    closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))
    segmented_img = cv2.morphologyEx(segmented_img, cv2.MORPH_CLOSE, closing_kernel)

    # I multiplied the original image with the segmentation mask to have only tumor region visible
    tumor_region = cv2.bitwise_and(img, img, mask=segmented_img)

    # drawing the ground truth square
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) # red
    cv2.rectangle(img_color, (x - radius, y - radius), (x + radius, y + radius), (0, 0, 255), 4)

    # saving results
    cv2.imwrite("results/" + name + "_original.png", img)
    cv2.imwrite("results/" + name + "_groundtruth.png", img_color)
    cv2.imwrite("results/" + name + "_segmented.png", segmented_img)
    cv2.imwrite("results/" + name + "_tumorregion.png", tumor_region)

print("\nResults saved in 'results' folder")