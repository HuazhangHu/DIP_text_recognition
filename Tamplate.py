from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import skimage
from skimage import morphology, draw, transform
from glob import glob

tamplate_list = []
image_dict = {}
tamplate_paths = glob('Data/model/*.jpg')

for l in tamplate_paths:
    tamplate_list.append(l.split('字符模板')[1].split('.jpg')[0])

# 模板匹配


def feature_extract(image, vision=False):
    feature = {}
    winSize = (12, 12)
    blockSize = (8, 8)
    blockStride = (2, 2)
    cellSize = (4, 4)
    nbins = 9

    winStride = (2, 2)
    padding = (2, 2)
    # 方向梯度直方图   HOG特征+特征匹配
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)

    if image.shape != (40, 20):
        image = cv2.resize(image, (20, 40))
    if np.max(image) != 1:  # 归一化
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        image = image.astype(np.uint8)

    image_sket = morphology.skeletonize(image).astype(np.uint8) * 255  # 提取字符图像的骨架
    # picture = Image.fromarray(image_sket)
    # picture.show()

    # feature radon
    theta = np.linspace(-30., 30., 60, endpoint=False)
    sinogram = skimage.transform.radon(image_sket, theta=theta, circle=True)
    feature['radon'] = sinogram
    # feature hog
    feature['hog'] = hog.compute(image_sket, winStride, padding).reshape((-1,))
    # feature pooling
    feature['pooling'] = np.zeros((8, 4))
    for h in range(0, 40, 5):
        for w in range(0, 20, 5):
            feature['pooling'][int(h / 5), int(w / 5)] = np.mean(image_sket[h: h + 5, w: w + 5])

    # if vision:
    #     plt.figure(2)
    #     plt.subplot(2, 1, 1)
    #     plt.imshow(image, cmap='gray')
    #     plt.subplot(2, 1, 2)
    #     plt.imshow(image_sket, cmap='gray')
    #     plt.show()

    return feature, image_sket


def tamplate_data(args=None):
    feature_dict = {}

    for char in tamplate_list:
        feature_dict[char] = []

        image_paths = [os.path.join('Data/model/字符模板{}.jpg'.format(char))]
        if (char == '1') or (char == '2') or (char == '3') or (char == '4') or ('char' == '9'):
            image_paths.append(os.path.join('Data/model/more/字符模板{}.jpg'.format(char)))
        for path in image_paths:
            image = (np.array(Image.open(path).convert('L')) > 128).astype(np.uint8)
            # image = morphology.skeletonize(image).astype(np.uint8) * 255

            feature, patch_sket = feature_extract(image, args)
            feature_dict[char].append(feature)

            Image.fromarray(patch_sket.astype(np.uint8)).save(os.path.join('character', 'char {}.jpg'.
                                                              format(char)))

    return feature_dict
