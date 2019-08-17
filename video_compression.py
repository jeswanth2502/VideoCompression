import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 
import cv2
import os
import numpy as np
import glob


def FrameCapture(path): 	 
	vidObj = cv2.VideoCapture(path)  
	count = 0 
	success = 1
	while success:         
         success, image = vidObj.read()
         #path = 'D:/python_package/image'
         #cv2.imwrite(os.path.join(path , 'frame%d.jpg' % count), image)
         cv2.imwrite("frame%d.jpg" % count, image) 
         count += 1

  
if __name__ == '__main__': 
	FrameCapture("aa.mp4") 

 
'''
for img in glob.glob("D:/python_package/image/*.jpg"):
    cv_img = cv2.imread(img)
    count=int(0)
    img=cv_img
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
    x,y=gray.shape[:2]
    #plt.imshow(gray,cmap=plt.get_cmap('gray'))
    #plt.show()
    new=np.fft.fft2(gray)
    shift=np.fft.fftshift(new)
    n=np.abs(shift)
    f=np.log(n+1)
    #plt.imshow(f,cmap=plt.get_cmap('gray'))
    #plt.show()
    count=2
    m=np.max(np.abs(new[:]))
    arr=np.array([0.001])
    for thresh in 0.1*arr*m:
        ind=abs(new)>thresh
        newfilt=np.multiply(new,ind)
        count=x*y-np.sum(ind[:])
        percent=100-count/x*y*100
        nfilt=np.fft.ifft2(newfilt)
        #filt=np.abs(nfilt)
#        nfilt=cv2.cvtColor(nfilt,cv2.COLOR_GRAY2BGR)
        #cv2.imshow('nfilt',nfilt)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        nfilt=np.abs(nfilt)
        plt.imshow(nfilt,cmap=plt.get_cmap('gray'))
        path = 'D:/python_package/image1'
        cv2.imwrite('frame%d.jpg' % count, nfilt)
        plt.show()
        count=count+1       
'''
img_array=[]
for image in glob.glob("D:/python_package/image1/*.jpg"):
    img=cv2.imread(image)
    plt.imshow(img)
    plt.show()
    height,width,layers=img.shape[:3]
    size=(width,height)
    img_array.append(img)
    
    
    
out=cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'),15,size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
