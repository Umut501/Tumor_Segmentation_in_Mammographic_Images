import cv2
import os
import numpy as np
import math

# I am extracting largest connected component, because  it is the breast region
def largest_connected_component(binary):
    bin_u8 = (binary > 0).astype('uint8') * 255
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_u8, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(binary, dtype=np.uint8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = 1 + int(np.argmax(areas))
    return (labels == largest_idx).astype(np.uint8)

# I am cutting the muscle parts using two triangles at 30 degree angle
def remove_muscle(img, angle_deg=26):
    h, w = img.shape[:5] # height and width
    ang = math.radians(angle_deg)

    # Sol çizgi
    x0_left, y0_left = 0, h
    x1_left = int(h * math.tan(ang))
    y1_left = 0

    # Sağ çizgi
    x0_right, y0_right = w, h
    x1_right = int(w - h * math.tan(ang))
    y1_right = 0

    # V shaped polygon to keep other areas
    polygon = np.array([
        [x0_left, y0_left],
        [x1_left, y1_left],
        [x1_right, y1_right],
        [x0_right, y0_right]
    ], np.int32)

    # Maske (0 her yer, 255 sadece göğüs bölgesi)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, polygon, 255)

    cleaned = cv2.bitwise_and(img, img, mask=mask)
    return cleaned, mask

# finding the seed algorithm
def compute_seed_pipeline(img, params):
    # In the first step I used otsu thresholding to get largest white region since it is the breast area
    _, m = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = largest_connected_component(m > 0).astype(np.uint8) * 255
    breast_mask_u8 = m.astype(np.uint8)
    breast_only = cv2.bitwise_and(img, img, mask=m)

    # COMMENTED OUT: Pectoral muscle removal using Otsu thresholding
    # (Now using geometric angle-based masking instead)
    # # Second step is taking Otsu threshold inside the breast region again
    # vals = breast_only[breast_only > 0]
    # inner_otsu = int(threshold_otsu(vals)) if vals.size > 0 else 0
    # thresh = inner_otsu + params.get("inner_thresh_offset", 0)
    # muscle_region_mask = largest_connected_component((breast_only > thresh).astype(np.uint8))

    # # Remove it from the breast image
    # breast_no_pectoral = breast_only.copy()
    # breast_no_pectoral[muscle_region_mask > 0] = 0


    # # I removed the muscles by thresholding inside breast and to set that area black I used intensity slicing
    # nonzero = breast_only[breast_only > 0]
    # if nonzero.size > 0:
    #     inner_otsu = int(threshold_otsu(nonzero))
    # else:
    #     inner_otsu = 0
    # inner_offset = params.get('inner_thresh_offset', 0)
    # thresh_val = inner_otsu + inner_offset
    # th2 = (breast_only > thresh_val).astype(np.uint8)
    # biggest_white = largest_connected_component(th2)
    # breast_no_pectoral = breast_only.copy()
    # breast_no_pectoral[biggest_white > 0] = 0

    # Pectoral muscle already removed by geometric angle masking
    breast_no_pectoral = breast_only.copy()

    # erosion with a large structuring element to remove vessels
    selem_radius = params.get('erosion_radius', 81)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (selem_radius*2+1, selem_radius*2+1))
    _, cleaned_bin = cv2.threshold(breast_no_pectoral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cleaned_bin = (cleaned_bin > 0).astype(np.uint8)
    eroded = cv2.erode((cleaned_bin*255).astype(np.uint8), kernel)

    # Step 3.5: mean filter (blur) with a large kernel
    mean_radius = params.get('mean_filter_radius', 81)
    mean_ksize = (mean_radius, mean_radius)
    mean_filtered = cv2.blur(breast_no_pectoral, mean_ksize)

    # Step 4: min filter with a large kernel
    min_radius = params.get('min_filter_radius', 21)
    kernel_min = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (min_radius, min_radius))
    min_filtered = cv2.erode(breast_no_pectoral, kernel_min)

    # Step 5: pick brightest pixel in min_filtered as seed
    # ensure we search only inside breast mask
    masked_min = cv2.bitwise_and(min_filtered, min_filtered, mask=breast_mask_u8)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(masked_min)
    seed_x, seed_y = int(max_loc[0]), int(max_loc[1])

    # Also produce a contrast_stretched masked image to pass to region growing
    return seed_x, seed_y, breast_mask_u8, breast_only, breast_no_pectoral, cleaned_bin, eroded, mean_filtered, min_filtered

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
    # --- 30° açıyla pektoral kasın maskelenmesi ---
    pectoral_removed, angle_mask = remove_muscle(contrast_stretching)
    # Pipeline’a bu görüntü girecek
    img_for_pipeline = pectoral_removed

    # So after I have done the preprocessing
    # I removed the noise with gaussian blur
    # there were some holes in some regions in segmentation before applying the gaussian blur
    # Also, contrast stretching helped to separate regions in a better way
    

    # Otomatik seed bulma - compute via pipeline (steps 1-5)
    params_pipeline = {'inner_thresh_offset': 20, 'erosion_radius': 12, 'min_filter_radius': 21}
    # x_auto, y_auto, breast_mask_u8, breast_only, breast_no_pectoral, cleaned_bin, eroded, mean_filtered, min_filtered = compute_seed_pipeline(img, params_pipeline)

    x_auto, y_auto, breast_mask_u8, breast_only, breast_no_pectoral, cleaned_bin, eroded, mean_filtered, min_filtered = compute_seed_pipeline(img_for_pipeline, params_pipeline)


    # Region growing threshold
    thresh = 12
    # Use contrast_stretching masked by breast_no_pectoral for region growing
    mask_for_rg = (breast_no_pectoral > 0).astype(np.uint8) * 255
    contrast_masked = cv2.bitwise_and(contrast_stretching, contrast_stretching, mask=mask_for_rg)
    segmented_img = region_growing_4_connectivity(contrast_masked, x_auto, y_auto, thresh, img.shape[0] * img.shape[1])

    # Closing
    closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))
    segmented_img_closed = cv2.morphologyEx(segmented_img, cv2.MORPH_CLOSE, closing_kernel)

    # Tumor region
    tumor_region = cv2.bitwise_and(img, img, mask=segmented_img_closed)

    # Ground truth
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(img_color, (x - radius, y - radius), (x + radius, y + radius), (0, 0, 255), 4)

    # Pipeline visualization
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

    # 4. Triangle Mask (Binary)
    step4 = cv2.cvtColor(angle_mask, cv2.COLOR_GRAY2BGR)
    cv2.putText(step4, "4. Triangle Mask (Binary)", (10, 40), font, 1.2, (255, 255, 255), 2)

    # 5. Triangle Filter Applied
    step5 = cv2.cvtColor(pectoral_removed, cv2.COLOR_GRAY2BGR)
    cv2.putText(step5, "5. Triangle Filter Applied", (10, 40), font, 1.2, (255, 255, 255), 2)

    # 6. Breast Region Extracted (Otsu)
    step6 = cv2.cvtColor(breast_only, cv2.COLOR_GRAY2BGR)
    cv2.putText(step6, "6. Breast Region (Otsu)", (10, 40), font, 1.2, (255, 255, 255), 2)

    # 7. Erosion (applied to grayscale breast_no_pectoral for visualization)
    # Create grayscale erosion for display
    selem_radius_vis = params_pipeline.get('erosion_radius', 12)
    kernel_vis = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (selem_radius_vis*2+1, selem_radius_vis*2+1))
    eroded_grayscale = cv2.erode(breast_no_pectoral, kernel_vis)
    step7 = cv2.cvtColor(eroded_grayscale, cv2.COLOR_GRAY2BGR)
    cv2.putText(step7, "7. Erosion", (10, 40), font, 1.2, (255, 255, 255), 2)

    # 8. Min Filtered
    step8 = cv2.cvtColor(min_filtered, cv2.COLOR_GRAY2BGR)
    cv2.putText(step8, "8. Min Filtered", (10, 40), font, 1.2, (255, 255, 255), 2)

    # 9. Finding Seed (the brightest point in image)
    step9 = cv2.cvtColor(min_filtered, cv2.COLOR_GRAY2BGR)
    cv2.circle(step9, (x_auto, y_auto), 5, (0, 255, 0), -1)
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
    # Segmented tumor bölgesini kırmızı renkte overlay
    tumor_mask_3ch = cv2.cvtColor(segmented_img_closed, cv2.COLOR_GRAY2BGR)
    step12[segmented_img_closed > 0] = [0, 0, 255]  # Red overlay
    cv2.putText(step12, "12. Tumor on Original", (10, 40), font, 1.2, (255, 255, 255), 2)

    # 13. Ground Truth
    step13 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(step13, (x - radius, y - radius), (x + radius, y + radius), (0, 0, 255), 4)
    cv2.putText(step13, "13. Ground Truth", (10, 40), font, 1.2, (255, 255, 255), 2)

    # 14. Overlay (Ground truth + Segmentation)
    step14 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # Segmented region in green
    step14[segmented_img_closed > 0] = [0, 255, 0]
    # Ground truth box in red
    cv2.rectangle(step14, (x - radius, y - radius), (x + radius, y + radius), (0, 0, 255), 4)
    cv2.putText(step14, "14. Overlay", (10, 40), font, 1.2, (255, 255, 255), 2)

    # Combine all steps in 4x4 grid (14 images + 2 empty)
    row1 = np.hstack([step1, step2, step3, step4])
    row2 = np.hstack([step5, step6, step7, step8])
    row3 = np.hstack([step9, step10, step11, step12])
    row4 = np.hstack([step13, step14, np.zeros_like(step1), np.zeros_like(step1)])

    combined = np.vstack([row1, row2, row3, row4])

    # saving results
    cv2.imwrite("results/" + name + "_pipeline.png", combined)

print("\nResults saved in 'results' folder")