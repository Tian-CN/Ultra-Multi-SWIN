import os
import sys
import json
import pickle
import random

import torch
from tqdm import tqdm

import matplotlib.pyplot as plt

from thop import profile

# 写独立受试者，四分类，数据集切分，程序
def split_data_cross_subjects_multi_SD(root: str = "./data/Ultrasound_photos_Pre-processed/Pleural_effusion", val_subj: int = [3], label_img: int =[0], val_rate: float = 0.2):   #多级别分类
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root) #是否存在文件夹
    # supported = [".avi", ".mp4"]  # 支持的文件后缀类型
    supported = ["S_1", "S_2", "S_3", "S_4", "S_5", "S_6", "S_7", "S_8", "S_9", "S_10", "S_11", "S_12", "S_13", "S_14",
                 "S_15", "S_16", "S_17", "S_18", "S_19", "S_20", "S_21", "S_22", "S_23", "S_24", "S_25"]  # 支持的文件前缀类型

    cla_path = root
    # 遍历文件夹，找出不同subject的数据
    images_S = {}
    for subject_nb in range(25):  # 共17组
        images_S[subject_nb] = [os.path.join(root, i) for i in os.listdir(cla_path) if
                                i.split('-', 1)[0] == supported[subject_nb]]  ## 防止歧义
    # 排序，保证各平台顺序一致
    # images_S.sort()
    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息

    # 按需求将受试者数据分为训练集与测试集
    for subj_nb in images_S:
        if subj_nb in val_subj:  # 如果该路径在采样的验证集样本中则存入验证集
            if not (images_S[subj_nb] is None):
                val_path = random.sample(images_S[subj_nb], k=int(len(images_S[subj_nb]) * val_rate))
                for img_path in images_S[subj_nb]:
                    if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                        val_images_path.append(img_path)
                        val_images_label.append(label_img)
                    else:  # 否则存入训练集
                        train_images_path.append(img_path)
                        train_images_label.append(label_img)

    print("{} images from class {} for training.".format(len(train_images_path), label_img))
    print("{} images from class {} for validation.".format(len(val_images_path), label_img))
    # assert len(train_images_path) > 0, "number of training images must greater than 0."
    # assert len(val_images_path) > 0, "number of validation images must greater than 0."
    return train_images_path, train_images_label, val_images_path, val_images_label


def read_split_data_cross_subjects_SD(root: str, val_subj: int = [3]): # 独立跨受试
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证各平台顺序一致
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)
    train_images_path = []
    train_images_label = []
    val_images_path = []
    val_images_label = []
    every_class_num = []  # 存储每个类别的样本总数
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        train_images_path_, train_images_label_, val_images_path_, val_images_label_ = split_data_cross_subjects_multi_SD(root = cla_path, val_subj=val_subj, label_img = image_class)

        val_images_path.extend(val_images_path_)
        val_images_label.extend(val_images_label_)
        train_images_path.extend(train_images_path_)
        train_images_label.extend(train_images_label_)
        every_class_num.append(len(train_images_path_))
        every_class_num.append(len(val_images_path_))
    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(val_images_path) > 0, "number of validation images must greater than 0."
    return train_images_path, train_images_label, val_images_path, val_images_label

# 写独立受试者，四分类，数据集切分，程序
def split_data_cross_subjects_multi(root: str = "./data/Ultrasound_photos_Pre-processed/Pleural_effusion", val_subj: int = [3], label_img: int =[0]):   #多级别分类
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root) #是否存在文件夹
    # supported = [".avi", ".mp4"]  # 支持的文件后缀类型
    supported = ["S_1", "S_2", "S_3", "S_4", "S_5", "S_6", "S_7", "S_8", "S_9", "S_10", "S_11", "S_12", "S_13", "S_14",
                 "S_15", "S_16", "S_17", "S_18", "S_19", "S_20", "S_21", "S_22", "S_23", "S_24", "S_25"]  # 支持的文件前缀类型

    cla_path = root
    # 遍历文件夹，找出不同subject的数据
    images_S = {}
    for subject_nb in range(25):  # 共17组
        images_S[subject_nb] = [os.path.join(root, i) for i in os.listdir(cla_path) if
                                i.split('-', 1)[0] == supported[subject_nb]]  ## 防止歧义
    # 排序，保证各平台顺序一致
    # images_S.sort()
    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息

    # 按需求将受试者数据分为训练集与测试集
    for subj_nb in images_S:
        if subj_nb in val_subj:  # 如果该路径在采样的验证集样本中则存入验证集
            if not (images_S[subj_nb] is None):
                for imgs_path in images_S[subj_nb]:
                    val_images_path.append(imgs_path)
                    val_images_label.append(label_img)
        else:  # 否则存入训练集
            if not (images_S[subj_nb] is None):
                for imgs_path in images_S[subj_nb]:
                    train_images_path.append(imgs_path)
                    train_images_label.append(label_img)
    print("{} images from class {} for training.".format(len(train_images_path), label_img))
    print("{} images from class {} for validation.".format(len(val_images_path), label_img))
    # assert len(train_images_path) > 0, "number of training images must greater than 0."
    # assert len(val_images_path) > 0, "number of validation images must greater than 0."
    return train_images_path, train_images_label, val_images_path, val_images_label

def split_data_cross_subjects(root: str = "./data/Ultrasound_photos_Pre-processed/Pleural_effusion", val_subj: int = [3], label_img: int =[0]):
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root) #是否存在文件夹

    # supported = [".avi", ".mp4"]  # 支持的文件后缀类型
    supported = ["S_1", "S_2", "S_3", "S_4", "S_5", "S_6", "S_7", "S_8", "S_9", "S_10", "S_11", "S_12", "S_13", "S_14", "S_15", "S_16", "S_17", "S_18", "S_19", "S_20", "S_21", "S_22", "S_23", "S_24", "S_25"]  # 支持的文件前缀类型

    cla_path = root
    # 遍历文件夹，找出不同subject的数据
    images_S = {}
    for subject_nb in range(25): # 共17组
        images_S[subject_nb] = [os.path.join(root, i) for i in os.listdir(cla_path) if i.split('-', 1)[0] in supported[subject_nb]]
    # 排序，保证各平台顺序一致
    # images_S.sort()


    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息

    # 按需求将受试者数据分为训练集与测试集
    for subj_nb in images_S:
        if subj_nb in val_subj:  # 如果该路径在采样的验证集样本中则存入验证集
            for imgs_path in images_S[subj_nb]:
                val_images_path.append(imgs_path)
                val_images_label.append(label_img)
        else:  # 否则存入训练集
            for imgs_path in images_S[subj_nb]:
                train_images_path.append(imgs_path)
                train_images_label.append(label_img)

    print("{} images from class {} for training.".format(len(train_images_path), label_img))
    print("{} images from class {} for validation.".format(len(val_images_path), label_img))
    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(val_images_path) > 0, "number of validation images must greater than 0."
    return train_images_path, train_images_label, val_images_path, val_images_label

def read_split_data_cross_subjects(root: str, val_subj: int = [3]): # 独立跨受试
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证各平台顺序一致
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)
    train_images_path = []
    train_images_label = []
    val_images_path = []
    val_images_label = []
    every_class_num = []  # 存储每个类别的样本总数
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        train_images_path_, train_images_label_, val_images_path_, val_images_label_ = split_data_cross_subjects_multi(root = cla_path, val_subj=val_subj, label_img = image_class)

        val_images_path.extend(val_images_path_)
        val_images_label.extend(val_images_label_)
        train_images_path.extend(train_images_path_)
        train_images_label.extend(train_images_label_)
        every_class_num.append(len(train_images_path_))
        every_class_num.append(len(val_images_path_))
    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(val_images_path) > 0, "number of validation images must greater than 0."
    return train_images_path, train_images_label, val_images_path, val_images_label

def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证各平台顺序一致
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 排序，保证各平台顺序一致
        images.sort()
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(val_images_path) > 0, "number of validation images must greater than 0."

    plot_image = False
    # plot_image = True
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(flower_class)), every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(flower_class)), flower_class)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    # plot_num = min(batch_size, 4)
    plot_num = min(batch_size, 1) #TZK

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


def train_one_epoch_count_params(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        flops, params = profile(model, inputs=(images.to(device),))
        print('flops:{}'.format(flops))
        print('params:{}'.format(params))


        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


# read_split_data_cross_subjects(root = "./data/Ultrasound_photos_Pre-processed/Pleural_effusion")