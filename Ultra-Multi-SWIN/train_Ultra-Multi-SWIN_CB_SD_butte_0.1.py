import os
import argparse

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from my_dataset import MyDataSet
# from model import swin_tiny_patch4_window7_224 as create_model
# from model_T_2Dmask_shift_filer import swin_tiny_patch4_window7_224 as create_model
from model_Tmask_shift_filer_StepMask_SoftMax1_test0505_res_combine import swin_tiny_patch4_window7_224 as create_model
# from model_Tmask_shift_filer_deform import swin_tiny_patch4_window7_224 as create_model  # TZK_deform
# from utils_T import read_split_data, train_one_epoch, evaluate, plot_data_loader_image, split_data_cross_subjects
from utils_T_butte import read_split_data, train_one_epoch, evaluate, plot_data_loader_image, read_split_data_cross_subjects, read_split_data_cross_subjects_SD
#TZK
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import random
from matplotlib import pyplot as plt
from scipy.io import loadmat, savemat
import numpy as np

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    tb_writer = SummaryWriter()
    # 历史数据保存
    history = np.zeros([4, args.epochs])
    nb_save = 0
    # Ultrasound_photos 独立跨受试实现（True），受试者依赖实验（False）
    if args.Subject_independent_Experiment:
        train_images_path, train_images_label, val_images_path, val_images_label = read_split_data_cross_subjects(
            root=args.data_path, val_subj=args.val_subj)
    else:
        train_images_path, train_images_label, val_images_path, val_images_label = read_split_data_cross_subjects_SD(
            root=args.data_path, val_subj=args.val_subj, val_rate_=args.val_rate)

    # train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)
    # val_subj = [3]
    # 独立跨受试
    # train_images_path, train_images_label, val_images_path, val_images_label = read_split_data_cross_subjects(root=args.data_path, val_subj=val_subj)

    img_size = 224

    # gaussiansize = [1, 3, 5, 7, 9] # TZK_random
    gaussiansize = [1, 3]  # TZK_random
    sharpness_factor = []

    data_transform = {
        # "train": transforms.Compose([transforms.RandomResizedCrop(img_size),
        #                              transforms.RandomHorizontalFlip(),
        #                              transforms.ToTensor(),
        #                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "train": transforms.Compose([transforms.Resize(int(img_size)),
                                     transforms.GaussianBlur(kernel_size=int(random.choice(gaussiansize)), sigma=(0.1, 2.0)),  #  TZK_GaussianBlur
                                     # transforms.RandomEqualize(p=0.5),
                                     # transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        # "val": transforms.Compose([transforms.Resize(int(img_size * 1.143)),
        #                            transforms.CenterCrop(img_size),
        #                            transforms.ToTensor(),
        #                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
        "val": transforms.Compose([transforms.Resize(int(img_size)),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    # nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers

    nw = 0 # 服务器4
    pin_memory = True # 服务器True
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=pin_memory,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=pin_memory,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    # plot_data_loader_image(train_loader)  #tzk
    # plot_data_loader_image(val_loader)    #tzk

    model = create_model(num_classes=args.num_classes).to(device)

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)["model"]
        # 删除有关分类类别的权重
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head外，其他权重全部冻结
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=5E-2)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs, eta_min=1E-10)  # TZK_scheduler

    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)

        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)
        # scheduler.step()    # TZK_scheduler
        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        # 保存历史数据
        history[0][nb_save] = train_loss
        history[1][nb_save] = train_acc
        history[2][nb_save] = val_loss
        history[3][nb_save] = val_acc
        nb_save = nb_save + 1

        torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))

    # 画历史图
    val_loss_ = history[2][:nb_save]
    val_acc_ = history[3][:nb_save]
    train_loss_ = history[0][:nb_save]
    train_acc_ = history[1][:nb_save]
    plt.subplot(1, 2, 1)
    plt.plot(train_acc_, label='train Accuracy')
    plt.plot(val_acc_, label='val Accuracy')
    plt.title('train and val Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_loss_, label='train Loss')
    plt.plot(val_loss_, label='val Loss')
    plt.title('train and val Loss')
    plt.legend()
    #保存历史图
    plt.savefig(
        './data/save_data/tswin_butte_SD_20230924_0.1/plot/butte_scheduler_0.00003_20230924_30_Subject-dependent_dorp_mask_layer_3_Ultra-SWIN-TANSFORMER_all_s' + str(args.all_subj[-1]) + '_val_s' + str(args.val_subj) + '_epochs_' + str(args.epochs) + '.png',
        bbox_inches='tight')
    # plt.show()
    plt.clf()
    # 保存历史数据
    savemat(
        './data/save_data/tswin_butte_SD_20230924_0.1/data/butte_scheduler_0.00003_20230924_30_Subject-dependent_dorp_mask_layer_3_Ultra-SWIN-TANSFORMER_all_s' + str(args.all_subj[-1]) + '_val_s' + str(args.val_subj) + '_epochs_' + str(args.epochs) + '.mat',
        {'history': history, 'subject_set': args.all_subj, 'subject_set_text': args.val_subj})
    print('done')

if __name__ == '__main__':
    # for val_subj_ in range(25):
    # for val_subj_ in range(7, 25):
    parser = argparse.ArgumentParser()

    parser.add_argument('--val_rate', type=float, default=0.1)

    # parser.add_argument('--num_classes', type=int, default=5)   #flowers
    parser.add_argument('--num_classes', type=int, default=2)   #Ultrasound_photos
    # parser.add_argument('--epochs', type=int, default=10)     #flowers
    parser.add_argument('--epochs', type=int, default=10)       #Ultrasound_photos
    parser.add_argument('--batch-size', type=int, default=64)   #服务器可 64 44 36
    # parser.add_argument('--lr', type=float, default=0.0000001)
    # parser.add_argument('--lr', type=float, default=0.00000001)     # TZK_deform
    parser.add_argument('--lr', type=float, default=0.00003)     # TZK_deform

    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    # parser.add_argument('--data-path', type=str,
    #                     default="/data/flower_photos")
    # parser.add_argument('--data-path', type=str,
    #                     default="./data/flower_photos")  #flowers
    # parser.add_argument('--data-path', type=str,
    #                     default="./data/Ultrasound_photos")  #Ultrasound_photos
    parser.add_argument('--Subject_independent_Experiment', type=bool,
                        default=False)  # Ultrasound_photos 独立跨受试实现（True），受试者依赖实验（False）
    # parser.add_argument('--val_subj', default=[0])  # Ultrasound_photos 独立跨受试实现中，作为验证集的受试者
    parser.add_argument('--val_subj', default=[0])  # Ultrasound_photos 独立跨受试实现中，作为验证集的受试者
    # parser.add_argument('--all_subj', default=[0, 1, 2, 3])  # Ultrasound_photos 独立跨受试实现中，所有的受试者
    parser.add_argument('--all_subj', default=range(25))  # Ultrasound_photos 独立跨受试实现中，所有的受试者
    # parser.add_argument('--data-path', type=str,
    #                     default="./data/Ultrasound_photos_Pre-processed_selected")  # Ultrasound_photos text
    parser.add_argument('--data-path', type=str,
                        default="./data/butte")  # Ultrasound_photos text
    # parser.add_argument('--data-path', type=str,
    #                     default="./data/Ultrasound_photos_un-Pre-processed")  # Ultrasound_photos text
    # parser.add_argument('--data-path', type=str,
    #                     default="./data/Ultrasound_photos_un-Pre-processed")  # Ultrasound_photos text


    # 预训练权重路径，如果不想载入就设置为空字符
    # parser.add_argument('--weights', type=str, default='./swin_tiny_patch4_window7_224.pth',
    #                     help='initial weights path')
    parser.add_argument('--weights', type=str, default='')  #   TZK，不用预训练参数
    # 是否冻结权重`
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    # torch.autograd.set_detect_anomaly(True)
    opt = parser.parse_args()

    main(opt)
