I present a method and algorithm for automatically detecting tumors in a
set of images from the MIAS (Mini Mammography Database). The report provides a step
by step explanation of the algorithm used and the reasons for my method choices for each
algorithm step. The algorithm first performs various preprocessing and masking operations
to remove unnecessary parts from the images. After removing the irrelevant parts, the next
step leaves only the breast region in the images. In the third step, the algorithm uses erosion and minimum filters to erode and remove vessels outside the tumor and irrelevant parts. In the next step, I select the brightest point in the processed images. This selected pixel is
assigned as the starting point for the region growing algorithm. Therefore, in the next step,
I use this starting point for region growing and apply morphological closing to fill in small
gaps within the segmented tumor area. Finally, I display the tumor region in the original image and plot the ground truth circle on the images for comparison.