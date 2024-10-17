
from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
import glob
from PIL import Image
import numpy as np
import streamlit as st

#### MTCNN ResNet のモデル読み込み
resnet = InceptionResnetV1(pretrained='vggface2').eval()

#### 画像ファイルから画像の特徴ベクトルを取得(ndarray 512次元)
def feature_vector(image):
    img = Image.open(image)
    mtcnn = MTCNN()
    face= mtcnn(img)
    feature_vector = resnet(face.unsqueeze(0)).squeeze().detach().numpy().copy()
    return feature_vector

#### 2つのベクトル間のコサイン類似度を取得(cosine_similarity(a, b) = a・b / |a||b|)
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

#画像中の人物を四角で囲んで、名前を追記して返す関数
def detect(image,dict):
    
    mtcnn = MTCNN(keep_all=True)
    img = Image.open(image)
    faces = mtcnn.detect(img)[0] # 顔領域を検出．画像中に複数の顔が検出されることも想定する
    faces_img = mtcnn(img)
    img_cv2 = np.array(img)

    #img_cv2 = cv2.imread(image)
    #img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    #faces = mtcnn.detect(img_cv2)[0] # 顔領域を検出．画像中に複数の顔が検出されることも想定する
    #faces_img = mtcnn(img_cv2)
    feature_vectors = [resnet(face.unsqueeze(0)).squeeze().detach().numpy().copy() for face in faces_img]

    #1人以上の顔を検出した場合
    if len(faces)>0:
        for i in range(len(faces)):
            tmp=faces[i]
            tmp= [int(i) for i in tmp]
            x, y, width, height = tmp
            feature_vector = feature_vectors[i]
            name = label_to_name(feature_vector,dict)
            cv2.rectangle(img_cv2, (x,y), (x+width,y+height), (255, 0, 0), thickness=3) #四角形描画
            cv2.putText(img_cv2, name,(x,y+height+20), cv2.FONT_HERSHEY_DUPLEX, 1, (255,0,0),2) #人物名記述

    return img_cv2

def label_to_name(feature_vector,dict):

    #### 2枚の画像間の類似度を取得
    name_label= 100
    for i in range(len(dict)):
        similarity = cosine_similarity(feature_vector, dict[i])
        if similarity > 0.7:
            name_label = i
            break

    if name_label == 0:
        name = "Party People"
    elif name_label == 1:
        name = "Kanemura Miku"
    elif name_label == 2:
        name = "Matuda Konoka"
    elif name_label == 3:
        name = "Nibu Akari"
    elif name_label == 4:
        name = "KAWADA Hina"
    else: 
        name = "Unknown"
    
    return name

def main(in_dir):


    in_jpg=glob.glob(in_dir)
    features=np.zeros((len(in_jpg),512))

    for i in range(len(in_jpg)):
        features[i,:]=feature_vector(in_jpg[i])

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
        image = "sample.jpeg"

    if image != None:
        
        #画像の読み込み
        detect_image = detect(image, features)
        #顔検出を行った結果を表示
        st.image(detect_image, use_column_width=True)

if __name__ == "__main__":
    main('./face_dict/*.jpg')
