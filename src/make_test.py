
import numpy as np
import cv2
import os
import glob
import shutil
import random


def augmentation_image(img,flip=True,thr=True,filt=True): 
    methods=[flip,thr,filt]

    augmentation = np.array([
        lambda x: cv2.flip(x,1),
        lambda x: cv2.threshold(x,100,255,cv2.THRESH_TOZERO)[1],
        lambda x: cv2.GaussianBlur(x,(5,5),0)
        ])#上から、反転処理、閾値処理、ぼかし処理

    images=[]
    for func in augmentation[methods]:
        images.append(func(img))

    return images

def main(in_dir,out_dir):

    if not os.path.exists("./train_data"):
        shutil.copytree("./face_data","./train_data")#./face_dataを./train_dataにコピー

    #in_dir="./train_data/*"
    #out_dir="./test_data/"
    member_dir=glob.glob(in_dir)


    #test datasetを抜き出し
    for num1 in range(len(member_dir)):
        temp_image_list=os.listdir(member_dir[num1])

        random.shuffle(temp_image_list)#listのshuffle
        temp_member_name=member_dir[num1].split("/")[-1]

        TEST_PATH=out_dir+temp_member_name
        if not os.path.exists(TEST_PATH):
            os.makedirs(TEST_PATH)

        #train:10→train:8,test:2
        for i in range(int((len(temp_image_list))/5)):
            shutil.move(str(member_dir[num1]+"/"+temp_image_list[i]),TEST_PATH)

    for num2 in range(len(member_dir)):   
        temp_image_list2=os.listdir(member_dir[num2])

        for p in range(len(temp_image_list2)):
            TRAIN_IMAGE_PATH=os.path.join(member_dir[num2]+"/"+temp_image_list2[p])
            img=cv2.imread(TRAIN_IMAGE_PATH)
            #print(img.shape)
            augmentation_face_image=augmentation_image(img)

            for method in range(len(augmentation_face_image)):
                AUGMENTATION_PATH=os.path.join(TRAIN_IMAGE_PATH[:-4]+"_"+str(method)+".jpg")
                cv2.imwrite(AUGMENTATION_PATH,augmentation_face_image[method]) 

