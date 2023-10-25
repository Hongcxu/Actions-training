import time
import math
from matplotlib.pyplot import title
import numpy as np
import torch
import os
import torch.nn as nn
import torch.utils.data
import torch.utils.data as Data
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler 
import tqdm

from sklearn.manifold import TSNE
from torch.nn.modules.container import T
from action_model import My_Net
from losses import OriTripletLoss
from read_action_data_10 import read_data
from plot_function import plot_embedding, confusion_matrix


torch.backends.cudnn.enable =True
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# 1.load train_data
train_x, train_y, test_x, test_y = read_data()
# 2.生成训练集与测试集数据集(train_dataset和test_dataset）和数据加载器(train_loader,test_loader)
BATCH_SIZE = 64
# TensorDataset可以将训练中的输入和输出值作为一对数据,输入输出都是元组
train_dataset = Data.TensorDataset(train_x, train_y) 
# 再通过DataLoader对象加载数据的方式进行设置，随机训练某批数据
train_loader = Data.DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

# 2.生成训练集与测试集数据集(train_dataset和test_dataset）和数据加载器(train_loader,test_loader)
BATCH_SIZE = 64
# TensorDataset可以将训练中的输入和输出值作为一对数据,输入输出都是元组
test_dataset = Data.TensorDataset(test_x, test_y) 
# 再通过DataLoader对象加载数据的方式进行设置，随机训练某批数据
test_loader = Data.DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=1,
)
val_num = len(test_dataset)

"-----------------------------3.开始训练模型-------------------"
# 1.实例化网络
net = My_Net(num_classes=10, init_weight = True)
# model_weight_path = '/data/XRD/rudar_cluster/action_recognization/save_model/Net_13_0.990.pth'
# assert os.path.exists(model_weight_path), "weights file: '{}' not exist.".format(model_weight_path)
# weights_dict = torch.load(model_weight_path, map_location=device)
# print(net.load_state_dict(weights_dict, strict=False))
net.to(device)


# 2.初始化部分参数
save_path= 'save_model/Net_13_{:.3f}.pth'    # 训练好的网络保存地址与文件名
#if not os.path.isdir(save_path):
   # os.makedirs(save_path)

best_acc = 0.90               # 记录最优准确率
epoches = 50                # 总训练次数
lrf = 0.0001
train_acc_list, test_acc_list, ce_loss_list, tri_loss_list = [], [], [], []
# 定义交叉熵损失函数
loss_function = nn.CrossEntropyLoss()
TripletLoss = OriTripletLoss()
# 定义Adam优化器(学习率lr=0.0001)
optimizer = optim.Adam(params = net.parameters(), lr = 0.001)
# scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
lf = lambda x: ((1 + math.cos(x * math.pi / epoches)) / 2) * (1 - lrf) + lrf  # cosine
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
# 3.for循环训练过程
if __name__ == '__main__':
    for epoch in range(epoches):
        running_loss1 = 0.0  # 记录每个epoch的总损失值
        running_loss2 = 0.0
        train_batch_acc = 0.0
        "-----------------------------2.1 train过程-------------------"
        net.train()         # 与Dropout层有关
        time_start = time.perf_counter()  # 记录开始训练的时间
        for step, data in enumerate(train_loader, start = 0):
            # 1.将数据与标签分开
            images, labels = data[0].to(device), data[1].to(device)
            # 2.梯度清零，防止积累
            optimizer.zero_grad()
            # 3.模型输入，得到输出
            feature, category = net(images)
            # 4.计算误差
            loss1 = loss_function(category, labels)
            loss2 = TripletLoss(feature, labels)
            loss = 1*loss1 + 0.1*loss2
            # 5.误差反向传播
            loss.backward()
            # 6.更新参数
            optimizer.step()
            # 7.记录一个batch的损失值
            running_loss1 += loss.item()  # tensor变量转变为data
            running_loss2 += loss2.item()
            # 8. 计算训练过程的准确率
            predictions = torch.max(category, dim = 1)[1]  # tensor变量在dim = 2维度取最大值，并返回最大值的索引
            train_batch_acc += (predictions == labels).sum().item()
            # 8.打印训练过程的进度
            rate = (step+1)/len(train_loader)
            a = "*" * int(rate*50)
            b = "." * int((1-rate)*50)
            print('\rtraining: {:^3.0f}% [{}->{}] batch_loss:{:.3f}'
                .format(int(rate*100), a, b, loss), end = "")
        print()
        scheduler.step()
        train_acc = train_batch_acc / len(train_x)  # len(train_x)为训练集总数量


        # 训练完一个epoch后, 打印单位batch的train_loss, 打印测试集上的总准确率test_acc
        print("[epoch:%d]----CE_loss:%.3f----Tri_loss:%.3f--train_acc:%.3f"
                %(epoch+1, running_loss1 / step, running_loss2 / step, train_acc))

        "-----------------------------2.2 test过程-------------------"
        # 1.count validate acc
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            # val_bar = tqdm(test_loader)
            for val_data in test_loader:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))  # eval model only have last output layer
                predict_y = torch.max(outputs[1], dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
        val_accurate = acc / val_num
        print('[epoch %d]  val_accuracy: %.3f' %(epoch + 1, val_accurate))

        # 2.save model(acc>0.95)
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path.format(best_acc))

        # 3.acc list
        train_acc_list.append(train_acc)
        test_acc_list.append(val_accurate)
        ce_loss_list.append(running_loss1)
        tri_loss_list.append(running_loss2)


        if (epoch+1) % 1 ==0:
            # 3. plot t-SNE feature distribute
            probs = []
            pred = []
            targets = []
            net.eval() # 预测过程中Dropout部分神经元不失活
            with torch.no_grad():
                for test_data in test_loader:
                    test_images, test_labels = test_data[0].to(device), test_data[1].to(device)
                    feature, category = net(test_images)
                    # probs.append(F.softmax(feature, dim=1))
                    probs.append(feature)
                    pred.append(torch.max(category, dim=1)[1])
                    targets.append(test_labels)
                for test_data in train_loader:
                    test_images, test_labels = test_data[0].to(device), test_data[1].to(device)
                    feature, category = net(test_images)
                    probs.append(feature)
                    pred.append(torch.max(category, dim=1)[1])
                    targets.append(test_labels)

            # for prob in probs:
            probs   = torch.cat(probs, dim = 0)
            pred    = torch.cat(pred, dim = 0)
            targets = torch.cat(targets, dim = 0)

            np.savetxt("feature_10.txt", probs.cpu().numpy(), fmt="%f", delimiter=",")
            np.savetxt("predict_10.txt", pred.cpu().numpy(), fmt="%d", delimiter=",")

            tsne = TSNE(n_components=2, init='pca', random_state=None)
            result = tsne.fit_transform(probs.detach().cpu().numpy())
            title_tsne = 'T-SNE Embedding of the Feature Vector(initial distribution)' #% (int(epoch+1))
            fig = plot_embedding(result, targets.detach().cpu().numpy(), title_tsne, epoch)

            # 4.plot confusion_matrix
            class_names_dict = {0: 'ā', 1: 'ō', 2: 'yī', 3: 'bā', 4: 'hēshuĭ', 5: 'A', 6: 'E', 7: 'I', 8: 'O', 9: 'U'}
            class_names = []
            confusion_matrix_file = 'visual_image\\confusion_matrix_epoch_%d.png' % (int(epoch+1))
            title_cm = 'Overall accuracy: %.3f'% (val_accurate)
            for i in range(len(class_names_dict)):
                class_names.append(class_names_dict[i])
            # predictions = [torch.cat(pred_, dim = 0).cpu() for pred_ in pred]
            confusion_matrix(pred.cpu().numpy(), targets.cpu().numpy(), class_names, confusion_matrix_file, title_cm)

    File_train_acc = open('train_acc10.txt', 'w')
    for ip in train_acc_list:
        File_train_acc.write(str(ip))
        File_train_acc.write('\n')
    File_train_acc.close()

    File_train_acc = open('test_acc10.txt', 'w')
    for ip in test_acc_list:
        File_train_acc.write(str(ip))
        File_train_acc.write('\n')
    File_train_acc.close()

    File_train_acc = open('ce_loss10.txt', 'w')
    for ip in ce_loss_list:
        File_train_acc.write(str(ip))
        File_train_acc.write('\n')
    File_train_acc.close()

    File_train_acc = open('tri_loss10.txt', 'w')
    for ip in tri_loss_list:
        File_train_acc.write(str(ip))
        File_train_acc.write('\n')
    File_train_acc.close()

    print("Finished Training!")