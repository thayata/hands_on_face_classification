
import torch
import torchvision
import torchvision.transforms as transforms
import os
import argparse
from src.make_dataset import Mydataset
from src.network import Net
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
import matplotlib.pyplot as plt

def parser(epo,rate,num):
    parser =argparse.ArgumentParser(description='Pytorch_Hinatazaka')
    parser.add_argument('--epochs','--e',type=int,default=2,help='number of epochs to train (default: 2)')
    parser.add_argument('--label','--lb',type=int,default=4,help='number of faces (default: 4)')    
    parser.add_argument('--lr','--l',type=float,default=0.001,help='learning rate (default: 0.001)')
    parser.add_argument('--save-model', action='store_true', default=True,help='For Saving the current Model')
    args = parser.parse_args(args=['--epochs',epo,'--label',num,'--lr',rate])
    return args

#training用の関数
def train(args, model, device, train_loader,criterion,optimizer,epoch):
    model.train()
    train_loss=0.0
    for batch_idx,(data,target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()#Adam初期化
        output=model(data)#model出力
        loss=criterion(output,target)#交差エントロピー誤差
        loss.backward()#逆誤差伝搬
        optimizer.step()#Adam利用
        train_loss+=loss.item()
        
    train_loss = train_loss / len(train_loader.dataset)
        
    return train_loss 

#test用の関数
def test(args, model, device,test_loader,criterion):
    model.eval()
    test_loss=0
    correct=0
    with torch.no_grad():
        for data,target in test_loader:
            data, target = data.to(device), target.to(device)
            output=model(data)
            test_loss += criterion(output,target).item()
            pred=output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')
    return test_loss,accuracy


def main(epo,rate,num,train_dir,test_dir):
    args = parser(epo,rate,num)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])#transforms

    trainset=Mydataset(root=train_dir,transform=transform)#Dataset
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,shuffle=True, num_workers=1, drop_last=True)#Dataloader
    testset=Mydataset(root=test_dir,transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=10,shuffle=True, num_workers=1, drop_last=True)

    model=Net(num=args.label)#modelの定義
    summary(model,(3,64,64))#modelを出力する便利ツール
    criterion=nn.CrossEntropyLoss()#lossの定義
    optimizer=optim.Adam(model.parameters(),lr=args.lr)#optimizerの定義

    #load cnn model to gpu
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Using device: {device}')
    model.to(device)

    #以下でtrain loss、test loss、test accuracyをグラフ化できるように設定
    x_epoch_data=[]
    y_train_loss_data=[]
    y_test_loss_data=[]
    y_test_accuracy_data=[]
    for epoch in range(1,args.epochs+1):
        train_loss_perEpoch = train(args,model,device,trainloader,criterion,optimizer,epoch)
        test_loss_perEpoch,test_accuracy_perEpoch = test(args,model,device,testloader,criterion)

        x_epoch_data.append(epoch)
        y_train_loss_data.append(train_loss_perEpoch)
        y_test_loss_data.append(test_loss_perEpoch)
        y_test_accuracy_data.append(test_accuracy_perEpoch)

    plt.plot(x_epoch_data,y_train_loss_data,label='train_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='upper right')
    plt.show()

    plt.plot(x_epoch_data,y_test_loss_data,label='test_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='upper right')
    plt.show()

    plt.plot(x_epoch_data,y_test_accuracy_data,label='test_accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(loc='lower right')
    plt.show()

    if (args.save_model):
        #gpuで学習したモデルをgpuで使う
        torch.save(model.state_dict(),"hinatazaka_cnn.pt")#args.save_modelがTrueなら最適化されたモデルを保存
        #gpuで学習したモデルをcpuで使う
        torch.save(model.to('cpu').state_dict(),"hinatazaka_cnn_cpu.pt")

if __name__ == '__main__':
    main()
