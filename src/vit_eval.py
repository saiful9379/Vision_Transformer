import warnings
warnings.filterwarnings("ignore")
import os
import glob
import json
import torchvision
import torch
import itertools
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import pandas as pd
from config import config as cfg
from transformers import ViTFeatureExtractor
from sklearn.metrics import f1_score, accuracy_score , precision_score, recall_score


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

EVAL_BATCH = 1
MODEL_PATH = os.path.join(cfg.MODEL_PATH, cfg.MODEL_NAME)
CLASS_PATH = os.path.join(cfg.MODEL_PATH, cfg.CLASS_DICT)

with open(CLASS_PATH) as f:
    class_dict = json.load(f)

model = torch.load(MODEL_PATH)
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
model.eval()



def accuracy_calculation(y_true, y_pred):
    f1_s = f1_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    accuracy = accuracy_score(y_true, y_pred)
    return precision, recall, f1_s, accuracy


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot
    """
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    c_path = os.path.join(cfg.LOG_DIR, 'confusion_matrix.png')
    plt.savefig(c_path)
    plt.clf() #clear buffer
    print("confusion matrix path  :", c_path)


def model_evaluation(input_img ,target = 0):
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

def evaluation(input_test):
    classes = [value for key, value in class_dict.items()]
    class_wised_prediction = []
    folders = sorted(os.listdir(input_test))
    cls_prediction_map = {}
    for _class in folders:
        print(_class)
        cls_dict  = {key : 0 for key, value in class_dict.items()}
        cls_folder_dir = os.path.join(input_test, _class)
        images = glob.glob(cls_folder_dir+"/*")
        target = classes.index(_class)
        target_list = [target]*len(images)
        predict_list = []
        for i in tqdm(range(len(images))):
            img = Image.open(images[i])
            p_cls_idx = int(model_evaluation(img, target=target))
            if str(p_cls_idx) in cls_dict:
                cls_dict[str(p_cls_idx)] +=1
            predict_list.append(p_cls_idx)
            
        cls_p_list = {
            "gt" : target_list, 
            "predict" : predict_list
        }
        # print("Done")
        cls_prediction_map[_class] = cls_p_list
        p_summary = [value for key, value in cls_dict.items()]
        class_wised_prediction.append(p_summary)

    plot_confusion_matrix(
        cm = np.array(class_wised_prediction), 
        normalize    = True,
        target_names = ['berry', 'bird', 'dog', "flower"],
        title        = "Confusion Matrix, Normalized"
        )
    p_list = []
    for i in cls_prediction_map:
        y_true = cls_prediction_map[i]["gt"]
        y_pred = cls_prediction_map[i]["predict"]

        p, r, f1, a = accuracy_calculation(y_true, y_pred)
        performance = [i, p, r, f1, a]
        p_list.append(performance)
    
    df = pd.DataFrame(p_list, columns = ['Class_Name', 'Precision', "Recall", "F1 Score", "Accuracy"])
    print(df)
    
if __name__ == "__main__":
    input_test = "./dataset_256X256/dataset/test"
    evaluation(input_test)
