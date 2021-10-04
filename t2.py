# 1. Only add your code inside the function (including newly improted packages). 
#  You can design a new function and call the new function in the given functions. 
# 2. For bonus: Give your own picturs. If you have N pictures, name your pictures such as ["t3_1.png", "t3_2.png", ..., "t3_N.png"], and put them inside the folder "images".
# 3. Not following the project guidelines will result in a 10% reduction in grades
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt



def stitch(imgmark, N=4, savepath=''): #For bonus: change your input(N=*) here as default if the number of your input pictures is not 4.
    "The output image should be saved in the savepath."
    "The intermediate overlap relation should be returned as NxN a one-hot(only contains 0 or 1) array."
    "Do NOT modify the code provided."
    
    imgpath = [f'./images/{imgmark}_{n}.png' for n in range(1,N+1)]
    imgs = []
    for ipath in imgpath:
        img = cv2.imread(ipath)
        imgs.append(img)
#Nearest Neighbor algorithm function
#=================================
    def matchmaking(kp1,kp2,des1,des2):
     good=[]
     for i in range(len(kp1)):
        dif = des2 - des1[i,:]
        dift = np.transpose(dif)
        eucld = np.sum(dif * dift.T, axis=1)
        sorteucld = np.sort(eucld)
        ind1 = np.asarray(np.where(eucld == sorteucld[0]))
        #ind2 = np.asarray(np.where(eucld == sorteucld[1]))
        if(sorteucld[0]<0.5*sorteucld[1]):
            m = [i, np.asscalar(ind1)]
            good.append(m)

    #matches=[]
     ptsA=[]
     ptsB=[]
     for i in range(len(good)):
        imkp1=good[i][0]
        imkp2=good[i][1]
        ptsA.append(kp1[imkp1].pt)
        ptsB.append(kp2[imkp2].pt)
    
     ptA = np.float32(ptsA).reshape(-1,1,2)
     ptB = np.float32(ptsB).reshape(-1,1,2)
    
     return ptA,ptB,len(good)
#==============================================

#Image Stitching
#==================================================
    def stitch_background(img1, img2):
        "The output image should be saved in the savepath."
        "Do NOT modify the code provided."
        img_1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        #plt.imshow(img1)
        #plt.show()
        img_2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
        #plt.imshow(img2)
        #plt.show()
        sift = cv2.xfeatures2d.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img_1,None)
        kp2, des2 = sift.detectAndCompute(img_2,None)
        ptA,ptB,l = matchmaking(kp1,kp2,des1,des2)
    
        H, masked = cv2.findHomography(ptA, ptB, cv2.RANSAC, 5.0)
        
        #IMAGE STITCHING
        rows1, cols1 = img2.shape[:2]
        rows2, cols2 = img1.shape[:2]
        p1 = np.float32([[0,0],[0,rows1],[cols1,rows1],[cols1,0]]).reshape(-1,1,2)
        temp = np.float32([[0,0],[0,rows2],[cols2,rows2],[cols2,0]]).reshape(-1,1,2)
        p2 = cv2.perspectiveTransform(temp, H)        
        p = np.concatenate((p1,p2), axis=0)        
        [x_min, y_min] = np.int32(p.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(p.max(axis=0).ravel() + 0.5)          
        Td = [-x_min,-y_min]          
        H_translation = np.array([[1, 0, Td[0]], [0, 1, Td[1]], [0, 0, 1]])
        result = cv2.warpPerspective(img1, H_translation.dot(H), (x_max-x_min, y_max-y_min))
        result[Td[1]:rows1+Td[1], Td[0]:cols1+Td[0]] = img2
        plt.imshow(result)
        plt.show()
        return result
#========================================================
        
#Calculate the overlap Ratio function 
#==========================================================        
    def match(img1, img2):
        "The output image should be saved in the savepath."
        "Do NOT modify the code provided."
        img_1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        #plt.imshow(img1)
        #plt.show()
        img_2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
        #plt.imshow(img2)
        #plt.show()
        sift = cv2.xfeatures2d.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img_1,None)
        kp2, des2 = sift.detectAndCompute(img_2,None)
        ptA,ptB,l = matchmaking(kp1,kp2,des1,des2)
        result = l/len(des1)
        return l,result
#==========================================================
        
#DETERMINE STITCHING SEQUENCE
#================================================    
    desmatrix=np.zeros((N,N))
    overlap=np.zeros((N,N))
    for i in range(len(imgs)):
        for j in range(len(imgs)):
            l,result = match(imgs[i],imgs[j])
            if result>=0.2:
                desmatrix[i,j]=l
                overlap[i,j]=1
    x=np.arange(len(imgs))
    for i in range(len(imgs)):
        x[i] = np.count_nonzero(desmatrix[i,:])
    max = np.max(x)
    index = np.asarray(np.where(x == max)).flatten()
    sequence=list()
    sum=np.arange(len(index))
    for i in range(len(index)):
        sum[i]=np.sum(desmatrix[index[i],:]) - desmatrix[index[i],index[i]]
    maxs=np.max(sum)
    index1 = np.asarray(np.where(sum == maxs)).flatten()
    sequence.append(np.asscalar(index[index1]))
#===============================================================

#IMAGE STITCHING PROCESS    
#===============================================================   
    sq = desmatrix[index[index1],:].flatten()
    sort=-np.sort(-sq).flatten()
    sort=sort[1:len(sort)]
    for i in range(len(sort)):
      ind = np.asarray(np.where(sq == sort[i])).flatten()
      sequence.append(np.asscalar(ind))
      
    finalimg = imgs[sequence[0]]
    for i in range(1,len(sequence)):
        finalimg = stitch_background(finalimg, imgs[sequence[i]])
#=================================================================    
    print(imgmark+" completed")
    cv2.imwrite(savepath,finalimg)
    
    overlap_arr=overlap
    return overlap_arr
if __name__ == "__main__":
    #task2
    overlap_arr = stitch('t2', N=4, savepath='task2.png')
    with open('t2_overlap.txt', 'w') as outfile:
        json.dump(overlap_arr.tolist(), outfile)
    #bonus
    overlap_arr2 = stitch('t3',N=4, savepath='task3.png')
    with open('t3_overlap.txt', 'w') as outfile:
        json.dump(overlap_arr2.tolist(), outfile)

