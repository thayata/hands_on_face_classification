import numpy as np
import cv2
import os
import glob

def main(in_dir,out_dir):

    ###in_dir="./data/*" # 入力用のディレクトリ
    ###out_dir="./face_data/" # 出力用(顔だけ画像)のディレクトリ
    in_jpg=glob.glob(in_dir) # [./data/各人物]のリスト
    in_jpg_member=[]

    for i in range(len(in_jpg)):
        in_jpg_member.append(glob.glob(in_jpg[i]+"/*.jpg")) # [./data/各人物/XXX.jpg]のリスト

    #in_fileName=os.listdir("./data") # 各人物のリスト
    in_fileName= [filename for filename in os.listdir("./data") if not filename.startswith('.')]

    for member_num in range(len(in_fileName)):
        OutputPath=out_dir+in_fileName[member_num]
        if not os.path.exists(OutputPath):
            os.makedirs(OutputPath)
        for num in range(len(in_jpg_member[member_num])):
            print(num)
            image=cv2.imread(in_jpg_member[member_num][num])
            if image is None:
                print("Not open:")
                continue
            image_gs=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            #cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml") 
            cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml") 
            face_list=cascade.detectMultiScale(image_gs, scaleFactor=1.1, minNeighbors=2,minSize=(64,64))

            count=0
            if len(face_list)>0:
                for rect in face_list:
                    count+=1
                    x,y,width,height=rect
                    print(x,y,width,height)
                    image_face=image[y:y+height,x:x+width]
                    if image_face.shape[0]<64:
                        continue
                    image_face = cv2.resize(image_face,(64,64))
                    fileName=os.path.join(out_dir+str(in_jpg_member[member_num][num][7:-4])+"_"+str(count)+".jpg")
                    print(fileName)
                    cv2.imwrite(str(fileName),image_face)
            else:
                print("no face")
                continue

