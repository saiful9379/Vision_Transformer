import warnings
warnings.filterwarnings("ignore")
import os
import glob
import json
import torchvision
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import ViTFeatureExtractor
from config import config as cfg
from tqdm import tqdm
import pandas as pd
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

EVAL_BATCH = 1
MODEL_PATH = os.path.join(cfg.MODEL_PATH, cfg.MODEL_NAME)
CLASS_PATH = os.path.join(cfg.MODEL_PATH, cfg.CLASS_DICT)

with open(CLASS_PATH) as f:
    class_dict = json.load(f)

model = torch.load(MODEL_PATH)
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
model.eval()

def get_model_prediction(input_img ,target = 0):
    newsize = (256, 256)
    img = input_img.resize(newsize)
    transform_list = [torchvision.transforms.ToTensor()]
    transform = torchvision.transforms.Compose(transform_list)
    img  = transform(img)
    inputs = img.permute(1, 2, 0)
    target = torch.as_tensor([target])
    with torch.no_grad():
        originalInput = inputs
        for index, array in enumerate(inputs):
            inputs[index] = np.squeeze(array)
        inputs = torch.tensor(np.stack(feature_extractor(inputs)['pixel_values'], axis=0))
        inputs = inputs.to(device)
        target = target.to(device)
        prediction, loss = model(inputs, target)
        predicted_class = np.argmax(prediction.cpu())
        cls_index = predicted_class.cpu().detach().numpy()
        return cls_index

def prediction(input_test):
    img = Image.open(input_test)
    p_cls_idx = int(get_model_prediction(img))
    predicted_cls= class_dict[f"{p_cls_idx}"]
    print("Class : ", predicted_cls)
    plt.title(f'Prediction : {predicted_cls}')
    plt.imshow(img)
    plt.show()



    
if __name__ == "__main__":
    input_test = "./dataset_256X256/dataset/test/flower/5_256.jpg"
    prediction(input_test)
