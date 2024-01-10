import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from albumentations import HorizontalFlip, VerticalFlip, Rotate
from typing_extensions import Concatenate
# def create_dir(path):
#     """ Create a directory. """
#     if not os.path.exists(path):
#         os.makedirs(path)

def load_data(path, split=0.2):
    """ Load the images and masks """
    images = sorted(glob(f"{path}\\images\\*.png"))      # 在指定路径path下，查找所有包含名为image子文件夹的文件夹，并在其中查找所有以.png结尾的文件名，将它们按字典序排序后存储在images列表中。
    masks = sorted(glob(f"{path}\\masks\\*.png"))

    """ Split the data """
    split_size = int(len(images) * split)               # 联系下一行指定了验证集的比例
    train_x, valid_x= train_test_split(images, test_size=split, random_state=42)
    train_y, valid_y = train_test_split(masks, test_size=split, random_state=42)

    return (train_x, train_y), (valid_x, valid_y)

def augment_data(images, masks, save_path, augment=True):
    """ Performing data augmentation. """
    H = 256
    W = 256
    # idx表示当前元组所对应的下标/顺序；zip将images和masks中互相对应的元素打包成一个个元组，返回元组组成的对象；使用len来指定预计迭代次数total
    for idx, (x, y) in tqdm(enumerate(zip(images, masks)), total=len(images)):      # 利用enumerate函数同时获取对象的索引和值
        """ Extracting the dir name and image name """
        print(x)
        # dir_name = x.split("\\")[-3]                                  # 从当前图像文件x的路径中，以/为分隔符，获取分割后的倒数第三个目录名
        name = x.split("\\")[-1].split(".")[0]      #  新的名称：图片文件名（没有后缀名）

        """ Read the image and mask """
        x = cv2.imread(x, cv2.IMREAD_COLOR)         # 以彩色格式读取图像
        y = cv2.imread(y, cv2.IMREAD_COLOR)

        if augment == True:
            aug = HorizontalFlip(p=1.0)             # 水平翻转，p表示翻转图像的概率，集对每个图像都进行翻转
            augmented = aug(image=x, mask=y)        # 对输入的图像和掩码进行增强
            x1 = augmented["image"]                 # 得到翻转后的图像
            y1 = augmented["mask"]
            #
            # aug = VerticalFlip(p=1)                 #垂直翻转图像
            # augmented = aug(image=x, mask=y)
            # x2 = augmented['image']
            # y2 = augmented['mask']

            # aug = Rotate(limit=45, p=1.0)           # 旋转增强，限制旋转角度在 -45 到 45 度之间，p=1.0 表示每个图像都进行旋转
            # augmented = aug(image=x, mask=y)        # 对输入的原始图像和掩码进行增强
            # x3 = augmented["image"]
            # y3 = augmented["mask"]

            X = [x, x1]
            Y = [y, y1]

        else:
            X = [x]
            Y = [y]

        idx = 0                                     # 初始化索引
        for i, m in zip(X, Y):
            i = cv2.resize(i, (W, H))               # 将图像调整为指定的256大小
            m = cv2.resize(m, (W, H))
            m = m/255.0                             # 对掩膜图像进行归一化
            m = (m > 0.5) * 255                     # 将掩膜图像二值化

            if len(X) == 1:                         # 如果没有进行数据增强
                tmp_image_name = f"{name}.jpg"
                tmp_mask_name  = f"{name}.jpg"
            else:
                tmp_image_name = f"{name}_{idx}.jpg"        # 将变量name的值插入到字符串中
                tmp_mask_name  = f"{name}_{idx}.jpg"

            image_path = os.path.join(save_path, "images\\", tmp_image_name)
            mask_path  = os.path.join(save_path, "masks\\", tmp_mask_name)

            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)

            idx += 1



if __name__ == "__main__":
    """ Load the dataset """
    dataset_path = "D:/shujuji/human/train"
    (train_x, train_y), (valid_x, valid_y) = load_data(dataset_path, split=0.2)

    print("Train: ", len(train_x))
    print("Valid: ", len(valid_x))

    # create_dir("new_data/train/image/")
    # create_dir("new_data/train/mask/")
    # create_dir("new_data/valid/image/")
    # create_dir("new_data/valid/mask/")

    augment_data(train_x, train_y, "human\\new_data\\train", augment=True)
    augment_data(valid_x, valid_y, "human\\new_data\\valid", augment=False)
