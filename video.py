import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 
import cv2
import os
import numpy as np
import glob


def FrameCapture(path):
    global count
    vidObj = cv2.VideoCapture(path)  
    success = 1
    while success:         
         success, image = vidObj.read()
         path = 'D:/python_package/image'
         cv2.imwrite(os.path.join(path , 'frame%d.jpg' % count), image)
         count += 1

         
count = int(0) 
FrameCapture("D:/python_package/naren.mp4") 
print("NUMBER OF FRAMES: ",count)



cnt=0
for i in range(count-1):
    img="D:/python_package/image/frame"+str(i)+".jpg"
    cv_img = cv2.imread(img)
    #count=int(0)
    img=cv_img
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
    x,y=gray.shape[:2]
    new=np.fft.fft2(gray)
    shift=np.fft.fftshift(new)
    n=np.abs(shift)
    f=np.log(n+1)
    #count=2
    m=np.max(np.abs(new[:]))
    arr=np.array([0.001])
    for thresh in 0.1*arr*m:
        ind=abs(new)>thresh
        newfilt=np.multiply(new,ind)
        nfilt=np.fft.ifft2(newfilt)
        nfilt=np.abs(nfilt)
        path = 'D:/python_package/image1/'
        cv2.imwrite(path+str(i)+".jpg", nfilt)
    cnt+=1

    
img_array=[]
for i in range(0,cnt-1) :
    image='D:/python_package/image1/'+str(i)+'.jpg'
    img=cv2.imread(image)
    height,width,layers=img.shape[:3]
    size=(width,height)
    img_array.append(img)
out=cv2.VideoWriter('project.mp4',cv2.VideoWriter_fourcc(*'XVID'),20,size)


for i in range(int(len(img_array))):
    out.write(img_array[i])
out.release()
