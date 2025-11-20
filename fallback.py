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
    imgName = lines[0]  #name of the img
    radius = int(lines[6])  # radius of the tumor in pixels
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

def find_seed_simple(img):
    """En parlak noktayı seed olarak seçer"""
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(img)
    seed_x = max_loc[0]
    seed_y = max_loc[1]
    return seed_x, seed_y

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
    
    # Otomatik seed bulma - en parlak noktayı seç
    x_auto, y_auto = find_seed_simple(contrast_stretching)
    
    # and as the third step of my algorithm I used region growing function here
    # this is where the main segmentation happens
    # I choose threshold value by trying different values I started from 10,12,13,15,18,20
    # I stopped at 20
    # because at 20 in smaller tumors it started doing too much over-segmentation
    # so what I found is:
    # thresh = 20 gives the best result for larger tumors
    # thresh = 10 gives the best result for smaller tumors
    # I picked 12 to make it suitable for both cases
    # but still it makes some under-segmentation for very big tumors
    thresh = 12
    
    # calling region_growing_4_connectivity function with automatic seed
    segmented_img = region_growing_4_connectivity(contrast_stretching, x_auto, y_auto, thresh, img.shape[0] * img.shape[1])
    
    # as the final step I used closing to fill gaps I tried several kernel sizes but 12 was enough
    closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))
    segmented_img_closed = cv2.morphologyEx(segmented_img, cv2.MORPH_CLOSE, closing_kernel)
    
    # I multiplied the original image with the segmentation mask to have only tumor region visible
    tumor_region = cv2.bitwise_and(img, img, mask=segmented_img_closed)
    
    # drawing the ground truth square
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # red
    cv2.rectangle(img_color, (x - radius, y - radius), (x + radius, y + radius), (0, 0, 255), 4)
    
    # Tüm adımları tek bir görüntüde birleştir
    img_color_step = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    blurred_color = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
    contrast_color = cv2.cvtColor(contrast_stretching, cv2.COLOR_GRAY2BGR)
    cv2.circle(contrast_color, (x_auto, y_auto), 5, (0, 255, 0), -1)
    segmented_color = cv2.cvtColor(segmented_img, cv2.COLOR_GRAY2BGR)
    closed_color = cv2.cvtColor(segmented_img_closed, cv2.COLOR_GRAY2BGR)
    tumor_color = cv2.cvtColor(tumor_region, cv2.COLOR_GRAY2BGR)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_color_step, "1. Original", (10, 40), font, 1.2, (255, 255, 255), 2)
    cv2.putText(blurred_color, "2. Gaussian Blur", (10, 40), font, 1.2, (255, 255, 255), 2)
    cv2.putText(contrast_color, "3. Contrast + Seed", (10, 40), font, 1.2, (255, 255, 255), 2)
    cv2.putText(segmented_color, "4. Region Growing", (10, 40), font, 1.2, (255, 255, 255), 2)
    cv2.putText(closed_color, "5. Morphological Close", (10, 40), font, 1.2, (255, 255, 255), 2)
    cv2.putText(tumor_color, "6. Tumor Region", (10, 40), font, 1.2, (255, 255, 255), 2)
    cv2.putText(img_color, "7. Ground Truth", (10, 40), font, 1.2, (255, 255, 255), 2)
    
    row1 = np.hstack([img_color_step, blurred_color, contrast_color, segmented_color])
    row2 = np.hstack([closed_color, tumor_color, img_color, np.zeros_like(img_color)])
    
    combined = np.vstack([row1, row2])
    
    # saving results
    cv2.imwrite("results/" + name + "_pipeline.png", combined)

print("\nResults saved in 'results' folder")