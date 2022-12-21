import os
import cv2
import glob
import warnings
warnings.filterwarnings('ignore')
import shutil
import torch
import PIL.Image
from PIL import Image
import numpy as np
import skimage.io as io
from matplotlib import cm
import matplotlib.pyplot as plt
from torchvision import transforms
from skimage.util import random_noise
from skimage.filters import gaussian
from skimage.transform import rotate, AffineTransform, warp


DEBUG_LOG = True

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

augmentation_type = [
    "_randomHorizontalFlip", 
    "_Pad", 
    "_RandomRotation", 
    "_RandomAffine", 
    "_brightness", 
    "_contrast",
    "_saturation",
    "_hue",
    "_flipLR",
    "_flipUD"
]

aug_img = []

def imshow(img_org, img_trans):
    """helper function to show data augmentation
    :param img: path of the image
    :param transform: data augmentation technique to apply"""
    fig, ax = plt.subplots(3, 3, figsize=(15, 15))
    cnt = 0
    for i in range(3):
        for j in range(3):
            ax[i,j].set_title(f'{img_trans[cnt]}')
            ax[i,j].imshow(img_org[cnt])
            cnt+=1
    augmentation_path = os.path.join(LOG_DIR, 'Augmentation_sample.png')
    fig.savefig(augmentation_path)
    print(augmentation_path)

def _randomHorizontalFlip(img):
    hlip_transform = transforms.RandomHorizontalFlip(p=1)
    return hlip_transform(img)
def _Pad(img):
    pad_transform = transforms.Pad((2, 5, 0, 5))
    return pad_transform(img)
def _RandomRotation(img):
    rotation_transform = transforms.RandomRotation(30)
    return rotation_transform(img)
def _RandomAffine(img):
    randon_affine_transform = transforms.RandomAffine(0, translate=(0.4, 0.5))
    return randon_affine_transform(img)
def _brightness(img):
    trans_brightness = transforms.ColorJitter(brightness=2)
    return trans_brightness(img)
def _contrast(img):
    trans_contrast = transforms.ColorJitter(contrast=2)
    return trans_contrast(img)
def _saturation(img):
    transform_saturation = transforms.ColorJitter(saturation=2)
    return transform_saturation(img)
def _hue(img):
    transform_hue = transforms.ColorJitter(hue=0.2)
    return transform_hue(img)

def _shift_operation(img):
    img = np.array(img)
    #apply shift operation
    transform = AffineTransform(translation=(25,25))
    wrapShift = warp(img,transform,mode='wrap')
    return wrapShift

def _add_random_noise(img):
    img = np.array(img)
#     img = io.imread(img)

    sigma=0.155
    #add random noise to the image
    noisyRandom = random_noise(img,var=sigma**2)
    return noisyRandom

def _blurred_gaussian(img):
#     img = io.imread(img)
    img = np.array(img)
    blurred = gaussian(img,sigma=1,multichannel=True)
    return blurred

#flip image left-to-right
def _flipLR(img):
    img = np.array(img)
    flipLR = np.fliplr(img)
    return flipLR
def _flipUD(img):
    img = np.array(img)
    #flip image up-to-down
    flipUD = np.flipud(img)
    return flipUD


def get_process_image(images:list=[], output_path:str=""):
    for img_path in images:
        file_name = os.path.basename(img_path)
        shutil.copy(img_path, output_path)
        img_org = PIL.Image.open(img_path)
        
        for fun_ in augmentation_type:
            def_fun = eval(f'{fun_}')
            tran_img = def_fun(img_org)
            output_img_name = os.path.join(output_path, file_name[:-4]+"_"+fun_+".jpg")
            if isinstance(tran_img, np.ndarray):
                tran_img = Image.fromarray(tran_img.astype('uint8'), 'RGB')
            tran_img.save(output_img_name)
            if len(aug_img) != 9:
                aug_img.append(tran_img)
    



def get_data_augmentation(dataset_path:str= "", dataset_aug_path:str=""):

    print("Start Data Augmentation Process : ", end="", flush=True)
    folders= os.listdir(dataset_path)
    for folder in folders:
        images_path = os.listdir(os.path.join(dataset_path, folder))

        for class_folder in images_path:
            output_path = os.path.join(dataset_aug_path, folder, class_folder)
            input_path = os.path.join(dataset_path, folder, class_folder)
            os.makedirs(output_path, exist_ok = True)
            images = glob.glob(input_path+"/*")
            get_process_image(
                images = images, 
                output_path = output_path
                )

    if DEBUG_LOG:
        imshow(aug_img, augmentation_type)
    print("Done")

if __name__ == "__main__":
    
    dataset_path= "./dataset_256X256/dataset/" 
    dataset_aug_path = "./dataset_256X256/dataset_aug"

    get_data_augmentation(
        dataset_path = dataset_path, 
        dataset_aug_path=dataset_aug_path
        )

