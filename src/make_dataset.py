
import torch
import os
import glob
import cv2
from PIL import Image

class Mydataset(torch.utils.data.Dataset):
    def __init__(self,root,transform=None):
        self.root=root#./train_data or ./test_data
        self.transform=transform
        self.data=[]
        self.targets=[]
        member_dir=sorted(glob.glob(self.root+"/*"))
        print(len(member_dir))
        print(member_dir)
        for member in range(len(member_dir)):
            #image_list=os.listdir(member_dir[member])
            image_list= [filename for filename in os.listdir(member_dir[member]) if not filename.startswith('.')]
            print(member_dir[member])
            #for i in range(len(image_list)):
            #    if 'DS' in image_list[i]:
            #        print(image_list[I])#.DS_Storeを検知できるように
            for num in range(len(image_list)):
                IMAGE_PATH=os.path.join(member_dir[member]+"/"+image_list[num])
                #print(IMAGE_PATH)
                self.data.append(cv2.imread(IMAGE_PATH))
                #ラベルは、富田:0,金村:1,松田:2,丹生:3,河田:4
                #print(member_dir)の順番をチェック
                self.targets.append(member)
                if type(self.data[num]) == type(None):
                    print(IMAGE_PATH)

    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):
        img,target=self.data[index],self.targets[index]
        img=Image.fromarray(img)#numpyからPILに
        if self.transform is not None:
            img=self.transform(img)
        return img,target

