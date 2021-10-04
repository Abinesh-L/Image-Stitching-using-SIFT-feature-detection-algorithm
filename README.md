# Image-Stitching-using-SIFT-feature-detection-algorithm
The goal of this task is to experiment with image stitching methods. Two images may have the same background but different foreground. For example, a moving person may be moving in the scene. The two images must be stitched into one image eliminating foreground objects that move in the scene. 

In this project there are 2 tasks performed:
1) Background Stitching: Two images with same background, but different foregrounds are
provided. The task is to stitch the two images given and eliminate the foreground objects
2) Image Panorama: A set of 4 images are given which needs to be stitched together to create one
panoramic image
# TASK 1:
 SIFT (Scale Invariant Feature Transform):
SIFT is a feature detection algorithm used to detect and describe the local features of an image. These
features derived and Scale and Rotation Invariant. In this project the in-built function in Opencv is
used to detect the features of the image.
kp1, des1 = sift.detectAndCompute(img_1,None)
kp1 – stores the orientation and coordinates of the descriptor
des1 – stores the bin values (128 bins) / the feature vector
 Nearest Neighbor:
This algorithm is generally used to solve optimization problems in a given set, of finding the point in a
set to another given point. In this project, once the image features are detected for both the images
this concept is used for matching the key descriptors between the two images.
This algorithm is based on the magnitude of SSD (Sum of Squared Differences) of the feature vectors
of the two descriptors selected in two different images. So comparing one feature vector of the first
image with all the feature vectors of the second image, the least SSD (or the nearest descriptor) is
taken as the match. To further improve the quality matches, the concept of ratio threshold is used,
where the SSD values of the closest neighbor(A) and second closest neighbor(B) is selected and their
ratio(A/B) is calculated. This ratio defines how distinct the match between the descriptors are. If the
ratio is high then it means they are not so unique and vice versa.
Using this algorithm the perfect matches are calculated.
 Homography matrix:
The Homography matrix relates the transformation between two image planes. This matrix defines
the action of image rectification, image registration, or computation of camera motion—rotation and
translation—between two images. This matrix can be calculated using a minimum number of 4 key
descriptors matches between two images. The inbuilt function in Opencv is used.
cv2.findHomography(ptB, ptA, cv2.RANSAC, 5.0)
ptB, ptA – represents the matched descriptors index
