
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.network import Net
import cv2
import os
import glob
from torchsummary import summary
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import streamlit as st

#画像中の人物を四角で囲んで、名前を追記して返す関数
def detect(image, model):

    #顔検出器の準備
    classifier = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
    #cascade = cv2.CascadeClassifier("lbpcascade_animeface.xml")
    #画像をグレースケール化
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #画像の中から顔を検出
    faces = classifier.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=2,minSize=(64,64))

    #1人以上の顔を検出した場合
    if len(faces)>0:
        for face in faces:
            x, y, width, height = face
            detect_face = image[y:y+height, x:x+width]
            if detect_face.shape[0] < 64:
                continue
            detect_face = cv2.resize(detect_face, (64,64))
            transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            detect_face = transform(detect_face)
            detect_face = detect_face.view(1,3,64,64)

            output = model(detect_face)

            name_label = output.argmax(dim=1, keepdim=True)
            name = label_to_name(name_label)

            cv2.rectangle(image, (x,y), (x+width,y+height), (255, 0, 0), thickness=3) #四角形描画
            cv2.putText(image, name,(x,y+height+20), cv2.FONT_HERSHEY_DUPLEX, 1, (255,0,0),2) #人物名記述

    return image


<<<<<<< HEAD
=======
#ラベルから対応する名前を返す関数
>>>>>>> a070e5e (update_for_2026)
def label_to_name(name_label):

    if name_label == 0:
        name = "Nibu Akari"
    elif name_label == 1:
        name = "Tomita Suzuka"
    elif name_label == 2:
        name = "Matuda Konoka"
    elif name_label == 3:
        name = "Kanemura Miku"
    #elif name_label == 4:
    #    name = "KAWADA Hina"
    
    return name

def main(num):

    st.set_page_config(layout="wide")
    #タイトルの表示
    st.title("日向坂46 顔認識アプリ")
    #制作者の表示
    #st.text("Created by your name")
    #アプリの説明の表示
    st.markdown("This is test. Change faces to your favorite persons.")

    #サイドバーの表示
    image = st.sidebar.file_uploader("画像をアップロードしてください", type=['jpg','jpeg', 'png'])
    #サンプル画像を使用する場合
    use_sample = st.sidebar.checkbox("サンプル画像を使用する")
    if use_sample:
        image = "./face_test/sample.jpeg"

    #保存済みのmodelのロードは次の3行で行うことができる
    model = Net(num=num)
    model.load_state_dict(torch.load('hinatazaka_cnn_cpu.pt'))
    model.eval()

    summary(model,(3,64,64))

    if image != None:
        
        #画像の読み込み
        image = np.array(Image.open(image))
        detect_image = detect(image, model)
        #顔検出を行った結果を表示
        st.image(detect_image, use_container_width=True)

if __name__ == "__main__":
    main()
