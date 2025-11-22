import cv2
import os
import numpy as np
import math

# I am extracting largest connected component because it is the breast region, 
# and I tried to remove white square labels in some images
def largest_connected_component(binary):
    bin_u8 = (binary > 0).astype('uint8') * 255
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_u8, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(binary, dtype=np.uint8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = 1 + int(np.argmax(areas))
    return (labels == largest_idx).astype(np.uint8)

# I am cutting the muscle parts using two triangles at 26 degree angle 
# (I tried 30 degree but in some it cuts tumors too) 
def remove_muscle(img, angle_deg=26):
    height, width = img.shape[:2] 
    ang = math.radians(26) # converting angle to radians

    # left line
    x0_left, y0_left = 0, height
    x1_left = int(height * math.tan(ang))
    y1_left = 0

    # right line
    x0_right, y0_right = width, height
    x1_right = int(width - height * math.tan(ang))
    y1_right = 0

    # to keep other areas, I created a V shape
    v_shape = np.array([
        [x0_left, y0_left],
        [x1_left, y1_left],
        [x1_right, y1_right],
        [x0_right, y0_right]
    ], np.int32)

    # mask to keep only the area inside the V shape
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, v_shape, 255)

    cleaned = cv2.bitwise_and(img, img, mask=mask)
    return cleaned, mask

# finding the seed algorithm
def find_seed(img, params):
    # all pre-processing is done in main loop before calling this function 
    # Gaussian blur, contrast stretching, muscle removal with triangle mask

    # In the first step I used otsu thresholding to separate breast region from background
    _, m = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # and I extracted largest connected component which is the breast region
    m = largest_connected_component(m > 0).astype(np.uint8) * 255 
    # breast mask
    breast_mask_u8 = m.astype(np.uint8)
    breast_only = cv2.bitwise_and(img, img, mask=m)

    breast_without_muscle = breast_only.copy()

    # I used erosion to remove vessels in the breast
    # I tried different sizes for the disk but 81 worked well
    erosion_radius = params.get('erosion_radius', 81) 
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_radius*2+1, erosion_radius*2+1))
    _, no_muscle_binary_img = cv2.threshold(breast_without_muscle, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # I made sure image is in binary before erosion
    no_muscle_binary_img = (no_muscle_binary_img > 0).astype(np.uint8)
    eroded = cv2.erode((no_muscle_binary_img*255).astype(np.uint8), disk)

    # I applied mean filter to remove small bright pixels which remained after erosion
    # I tried different sizes for the disk but 81 worked well
    mean_radius = params.get('mean_filter_radius', 81)
    mean_ksize = (mean_radius, mean_radius)
    mean_filtered = cv2.blur(breast_without_muscle, mean_ksize)

    # min filter to set bright regions to lower values to increase the contrast between bright regions
    # I tried different sizes for the disk but 21 worked well
    min_radius = params.get('min_filter_radius', 21)
    disk_min = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (min_radius, min_radius))
    min_filtered = cv2.erode(breast_without_muscle, disk_min)

    # I picked the brightest pixel in min_filtered image as the seed pixel for my region growing algorithm
    masked_min = cv2.bitwise_and(min_filtered, min_filtered, mask=breast_mask_u8)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(masked_min)
    seed_x, seed_y = int(max_loc[0]), int(max_loc[1])

    # returning all results for showing the steps
    return seed_x, seed_y, breast_mask_u8, breast_only, breast_without_muscle, no_muscle_binary_img, eroded, mean_filtered, min_filtered

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


# creating the results folder if it doesn't exist
if not os.path.exists("results/"):
    os.makedirs("results/")

# creating the image list to store the images
imgList = []

# opening the info file
file = open("dataset/" + "info.txt", "r")

# reading the file content
for line in file:
    #spliting the text line by line
    lines = line.split()
    print(line)
    x = int(lines[4])
    y = 1024 - int(lines[5])  # assigment says "This value should be subtracted from 1024"
    imgName = lines[0]  # name of the img
    radius = int(lines[6])  # radius of the tumor in pixels
    imgList.append({'name': imgName, 'x': x, 'y': y, 'radius': radius})

file.close()

print("Number of images:", len(imgList))

# reading data for each image in the list
for i in range(len(imgList)):
    imageList = imgList[i]
    name = imageList['name']
    x = imageList['x']
    y = imageList['y']
    radius = imageList['radius']

    print(f"Processing image {i+1}/{len(imgList)}: {name}...")
    
    # reading images
    # I use imread to read only .pgm files because .jpg is compressed, and we may lose important details
    img = cv2.imread("dataset/" + name + ".pgm", cv2.IMREAD_GRAYSCALE)
    
    # in my first step I use gaussian to remove noise from the images
    # 9 by 9 disk worked well for this dataset
    # I also tried 3 by 3 mask, but it didn't blur the images enough
    blurred = cv2.GaussianBlur(img, (9, 9), 0)
    
    # secondly I used contrast stretching to enhance the intensity differences
    contrast_stretching = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)

    # thirdly I removed the muscle part and white label apperaing in the some images with triangle mask
    muscle_removed, angle_mask = remove_muscle(contrast_stretching)

    # I use this preprocessed image for the finding_seed
    img_for_finding_seed = muscle_removed

    # I set the parameters for the finding_seed here
    parameters_for_finding_seed = {'inner_thresh_offset': 20, 'erosion_radius': 12, 'min_filter_radius': 21}

    # x_coordinate_of_seed and y_coordinate_of_seed are the coordinates of the seed point returned by the find_seed function
    x_coordinate_of_seed, y_coordinate_of_seed, breast_mask_u8, breast_only, breast_without_muscle, no_muscle_binary_img, eroded, mean_filtered, min_filtered = find_seed(img_for_finding_seed, parameters_for_finding_seed)

    # region growing part 

    # I decided to set region growing threshold to 12 after several trys
    thresh = 12
    # mask to limit region growing only to breast area without muscle
    mask_for_rg = (breast_without_muscle > 0).astype(np.uint8) * 255
    # applying mask to contrast stretched image for region growing 
    contrast_masked = cv2.bitwise_and(contrast_stretching, contrast_stretching, mask=mask_for_rg)
    # calling region growing function
    segmented_img = region_growing_4_connectivity(contrast_masked, x_coordinate_of_seed, y_coordinate_of_seed, thresh, img.shape[0] * img.shape[1])

    # after region growing I applied morphological closing to fill small holes in the segmented region
    closing_disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))
    segmented_img_closed = cv2.morphologyEx(segmented_img, cv2.MORPH_CLOSE, closing_disk)

    # so I extracted the tumor region from the original image for visualization
    tumor_region = cv2.bitwise_and(img, img, mask=segmented_img_closed)

    # drawing ground truth circle on original image
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.circle(img_color, (x, y), radius, (0, 0, 255), 4)

    # Saving all steps in one image for finding seed
    # I created a grid to show all steps
    font = cv2.FONT_HERSHEY_SIMPLEX

    # 1. Original
    step1 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.putText(step1, "1. Original", (10, 40), font, 1.2, (255, 255, 255), 2)

    # 2. Gaussian Blurred
    step2 = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
    cv2.putText(step2, "2. Gaussian Blurred", (10, 40), font, 1.2, (255, 255, 255), 2)

    # 3. Contrast Stretched
    step3 = cv2.cvtColor(contrast_stretching, cv2.COLOR_GRAY2BGR)
    cv2.putText(step3, "3. Contrast Stretched", (10, 40), font, 1.2, (255, 255, 255), 2)

    # 4. Triangle Mask
    step4 = cv2.cvtColor(angle_mask, cv2.COLOR_GRAY2BGR)
    cv2.putText(step4, "4. Triangle Mask (Binary)", (10, 40), font, 1.2, (255, 255, 255), 2)

    # 5. Triangle Mask Applied
    step5 = cv2.cvtColor(muscle_removed, cv2.COLOR_GRAY2BGR)
    cv2.putText(step5, "5. Triangle Mask Applied", (10, 40), font, 1.2, (255, 255, 255), 2)

    # 6. Breast Region Extracted
    step6 = cv2.cvtColor(breast_only, cv2.COLOR_GRAY2BGR)
    cv2.putText(step6, "6. Breast Region (Otsu)", (10, 40), font, 1.2, (255, 255, 255), 2)

    # 7. Erosion
    # I created grayscale erosion for showing
    erosion_radius_ = parameters_for_finding_seed.get('erosion_radius', 12)
    disk_vis = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_radius_*2+1, erosion_radius_*2+1))
    eroded_grayscale = cv2.erode(breast_without_muscle, disk_vis)
    step7 = cv2.cvtColor(eroded_grayscale, cv2.COLOR_GRAY2BGR)
    cv2.putText(step7, "7. Erosion", (10, 40), font, 1.2, (255, 255, 255), 2)

    # 8. Min Filtered
    step8 = cv2.cvtColor(min_filtered, cv2.COLOR_GRAY2BGR)
    cv2.putText(step8, "8. Min Filtered", (10, 40), font, 1.2, (255, 255, 255), 2)

    # 9. Finding Seed (the brightest point in image)
    step9 = cv2.cvtColor(min_filtered, cv2.COLOR_GRAY2BGR)
    cv2.circle(step9, (x_coordinate_of_seed, y_coordinate_of_seed), 5, (0, 255, 0), -1)
    cv2.putText(step9, "9. Finding Seed", (10, 40), font, 1.2, (255, 255, 255), 2)
    cv2.putText(step9, "(brightest point)", (10, 80), font, 0.8, (255, 255, 255), 2)

    # 10. Region Growing
    step10 = cv2.cvtColor(segmented_img, cv2.COLOR_GRAY2BGR)
    cv2.putText(step10, "10. Region Growing", (10, 40), font, 1.2, (255, 255, 255), 2)

    # 11. Closing
    step11 = cv2.cvtColor(segmented_img_closed, cv2.COLOR_GRAY2BGR)
    cv2.putText(step11, "11. Closing", (10, 40), font, 1.2, (255, 255, 255), 2)

    # 12. Tumor on Original
    step12 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # Segmented tumor region is shown in red
    tumor_mask_3ch = cv2.cvtColor(segmented_img_closed, cv2.COLOR_GRAY2BGR)
    step12[segmented_img_closed > 0] = [0, 0, 255]  # Red overlay
    cv2.putText(step12, "12. Tumor on Original", (10, 40), font, 1.2, (255, 255, 255), 2)

    # 13. Ground Truth
    step13 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.circle(step13, (x, y), radius, (0, 0, 255), 4)
    cv2.putText(step13, "13. Ground Truth", (10, 40), font, 1.2, (255, 255, 255), 2)

    # 14. Overlay (Ground truth + Segmentation)
    step14 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # Segmented region in green
    step14[segmented_img_closed > 0] = [0, 255, 0]
    # Ground truth box in red
    cv2.circle(step14, (x, y), radius, (0, 0, 255), 4)
    cv2.putText(step14, "14. Overlay", (10, 40), font, 1.2, (255, 255, 255), 2)

    # combining all steps into one image
    row1 = np.hstack([step1, step2, step3, step4])
    row2 = np.hstack([step5, step6, step7, step8])
    row3 = np.hstack([step9, step10, step11, step12])
    row4 = np.hstack([step13, step14, np.zeros_like(step1), np.zeros_like(step1)])

    combined = np.vstack([row1, row2, row3, row4])

    # saving results
    cv2.imwrite("results/" + name + "_finding_seed.png", combined)

print("\nResults saved in results folder")