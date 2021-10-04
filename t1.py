#Only add your code inside the function (including newly improted packages)
# You can design a new function and call the new function in the given functions. 
# Not following the project guidelines will result in a 10% reduction in grades

import cv2
import numpy as np
import matplotlib.pyplot as plt
   

def stitch_background(img1, img2, savepath=''):
    "The output image should be saved in the savepath."
    "Do NOT modify the code provided."
#Nearest Neighbor algorithm function
#======================================
    def matchmaking(kp1,kp2,des1,des2):
        good=[]
        for i in range(len(kp1)):
            dif = des2 - des1[i,:]
            dift = np.transpose(dif)
            eucld = np.sum(dif * dift.T, axis=1)
            sorteucld = np.sort(eucld)
            ind1 = np.asarray(np.where(eucld == sorteucld[0]))
            if(sorteucld[0]<0.5*sorteucld[1]):
                m = [i, np.asscalar(ind1)]
                good.append(m)
        ptsA=[]
        ptsB=[]
        for i in range(len(good)):
            imkp1=good[i][0]
            imkp2=good[i][1]
    #matches.append([kp1[imkp1],kp2[imkp2]])
            ptsA.append(kp1[imkp1].pt)
            ptsB.append(kp2[imkp2].pt)
    
        ptA = np.float32(ptsA).reshape(-1,1,2)
        ptB = np.float32(ptsB).reshape(-1,1,2)    
        return ptA,ptB
#============================================    
   
    img_1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    #plt.imshow(img1)
    #plt.show()
    img_2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    #plt.imshow(img2)
    #plt.show()
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img_1,None)
    kp2, des2 = sift.detectAndCompute(img_2,None)
    ptA,ptB = matchmaking(kp1,kp2,des1,des2)
    
    #HOMOGRAPHY MATRIX
    H, masked = cv2.findHomography(ptB, ptA, cv2.RANSAC, 5.0)
    
    #IMAGE STITCHING
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]
    p1 = np.float32([[0,0],[0,rows1],[cols1,rows1],[cols1,0]]).reshape(-1,1,2)
    temp = np.float32([[0,0],[0,rows2],[cols2,rows2],[cols2,0]]).reshape(-1,1,2)
    p2 = cv2.perspectiveTransform(temp,H)    
    p = np.concatenate((p1,p2), axis=0)    
    [x_min, y_min] = np.int32(p.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(p.max(axis=0).ravel() + 0.5)      
    Td = [-x_min,-y_min]      
    H_translation = np.array([[1, 0, Td[0]], [0, 1, Td[1]], [0, 0, 1]])
    result = cv2.warpPerspective(img2, H_translation.dot(H), (x_max-x_min, y_max-y_min))
    
    #FOREGROUND REMOVAL
    imcrop=result[Td[1]:rows1+Td[1], Td[0]:cols1+Td[0]]
    imgnew=img1
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            for k in range(img1.shape[2]):
             if (imcrop[i,j,k])>(img1[i,j,k]):
                imgnew[i,j,k]=imcrop[i,j,k]
      
    #FINAL STITCHED IMAGE          
    result[Td[1]:rows1+Td[1], Td[0]:cols1+Td[0]] = imgnew
    plt.imshow(result)
# =============================================================================
# 
#     dst = cv2.warpPerspective(img1,H1,((img1.shape[1] + img2.shape[1]), img2.shape[0]))#warped image
#     dst[0:img2.shape[0], 0:img2.shape[1]] = img2 #stitched image
#     cv2.imwrite('output1.jpg',dst)
#     result=dst
#     plt.imshow(dst)
#     plt.show()s
# =============================================================================
    cv2.imwrite(savepath,result)
    return 


if __name__ == "__main__":
    img1 = cv2.imread('./images/t1_1.png')
    img2 = cv2.imread('./images/t1_2.png')
    savepath = 'task1.png'
    stitch_background(img1, img2, savepath=savepath)
    #print(final)
    #plt.imshow(final)

