
import os
import json
import torchvision
import torch
import torch.nn as nn
import numpy as np
from numpy import arange
from config import config as cfg
from torchvision import transforms
import torch.nn.functional as F
from pickle import dump, load
from matplotlib.pylab import plt
import torch.utils.data as data
from torch.autograd import Variable
from torchvision.transforms import ToTensor
from model import ViTForImageClassification
from transformers import ViTFeatureExtractor
from transformers.modeling_outputs import SequenceClassifierOutput
from data_augmentation import get_data_augmentation

EPOCHS = cfg.EPOCHS
BATCH_SIZE = cfg.BATCH_SIZE
LEARNING_RATE = cfg.LEARNING_RATE
model_folder = cfg.MODEL_PATH

if cfg.DEBUG_FOR_ORGINAL_DATA:
    get_data_augmentation(cfg.DATASET_PATH, cfg.DATASET_AUG_PATH)
    training_data_path = os.path.join(cfg.DATASET_AUG_PATH, "train")
    validation_data_path = os.path.join(cfg.DATASET_AUG_PATH, "test")
else:
    training_data_path = os.path.join(cfg.DATASET_PATH, "train")
    validation_data_path = os.path.join(cfg.DATASET_PATH, "test")


# print(training_data_path, validation_data_path)

os.makedirs(model_folder, exist_ok = True)
model_path = os.path.join(model_folder, cfg.MODEL_NAME)
class_path = os.path.join(model_folder, cfg.CLASS_DICT)

def plot_loss_accuracy(train_loss_path, val_loss_path):
    # # Load the training and validation loss dictionaries
    train_loss = load(open(train_loss_path, 'rb'))
    val_loss = load(open(val_loss_path, 'rb'))
    #  
    # Retrieve each dictionary's values
    # Retrieve each dictionary's values
    # print(train_loss)
    # print(val_loss)
    train_values = train_loss.values()
    val_values = val_loss.values()
    epochs = range(0, cfg.EPOCHS)
    # Plot and label the training and validation loss values
    plt.plot(epochs, train_values, label='Training Loss')
    plt.plot(epochs, val_values, label='Validation Loss')

    # Add in a title and axes labels
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    # Set the tick locations
    plt.xticks(arange(0, cfg.EPOCHS, 2))

    # Display the plot
    # plt.legend(loc='best')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(os.path.join(cfg.LOG_DIR, 'model_training_loss.png'))
    plt.clf() #clear buffer
    # plt.show()

def data_loader():
    TRANSFORM_IMG = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
        ])

    train_ds = torchvision.datasets.ImageFolder(
        training_data_path, 
        transform=TRANSFORM_IMG
        )
    valid_ds = torchvision.datasets.ImageFolder(
        validation_data_path, 
        transform=TRANSFORM_IMG
        )
    return train_ds, valid_ds


def train():

    train_ds, valid_ds = data_loader()
    print("Number of Class : ", len(train_ds.classes))
    model = ViTForImageClassification(len(train_ds.classes)) 
    # model = ViTForImageClassification(len(train_ds.classes))    
    # Feature Extractor
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
    # Adam Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Cross Entropy Loss
    loss_func = nn.CrossEntropyLoss()
    # Use GPU if available  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    if torch.cuda.is_available():
        model.cuda()
        
    print("Number of train samples: ", len(train_ds))
    print("Number of test samples: ", len(valid_ds))

    print("Detected Classes are: ", train_ds.class_to_idx)
    class_dict = {value: key for key, value in train_ds.class_to_idx.items()}

    with open(class_path, 'w') as f:
        json.dump(class_dict, f)


    train_loader = data.DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=True,  
        num_workers=4
    )
    test_loader  = data.DataLoader(
        valid_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4
    ) 
    print("Data loader Finished")
    # Train the model
    train_loss_dict, val_loss_dict  = {}, {}
    for epoch in range(EPOCHS):
        t_loss, v_loss = 0, 0
        v_idx = 0        
        for step, (x, y) in enumerate(train_loader):
            # Change input array into list with each batch being one element
            x = np.split(np.squeeze(np.array(x)), BATCH_SIZE)
            # Remove unecessary dimension
            for index, array in enumerate(x):
                x[index] = np.squeeze(array)
            # Apply feature extractor, stack back into 1 tensor and then convert to tensor
            x = torch.tensor(np.stack(feature_extractor(x)['pixel_values'], axis=0))
            # Send to GPU if available
            x, y  = x.to(device), y.to(device)
            b_x = Variable(x)   # batch x (image)
            b_y = Variable(y)   # batch y (target)
            # Feed through model
            output, loss = model(b_x, None)
            # Calculate loss
            e_step = f'epoch_step'
            # train_loss_dict[e_step] = loss
            
            if loss is None: 
                loss = loss_func(output, b_y)   
                optimizer.zero_grad()           
                loss.backward()                 
                optimizer.step()
                t_loss += float(loss)
                # print("loss", float(loss),type(loss))
            else:
                # print("loss", float(loss), type(loss))
                t_loss += float(loss)

            if step % 50 == 0:
                # Get the next batch for testing purposes
    #             test = next(iter(train_loader))
                test = next(iter(test_loader))
                test_x = test[0]
                # Reshape and get feature matrices as needed
                test_x = np.split(np.squeeze(np.array(test_x)), BATCH_SIZE)
                for index, array in enumerate(test_x):
                    test_x[index] = np.squeeze(array)
                test_x = torch.tensor(np.stack(feature_extractor(test_x)['pixel_values'], axis=0))
                # Send to appropirate computing device
                test_x = test_x.to(device)
                test_y = test[1].to(device)
                # Get output (+ respective class) and compare to target
                test_output, loss = model(test_x, test_y)
                test_output = test_output.argmax(1)
                # Calculate Accuracy
                v_idx+=1
                v_loss += float(loss)
                accuracy = (test_output == test_y).sum().item() / BATCH_SIZE
                print('Epoch: ', epoch, "step : ", step,"/",len(train_loader), '| train loss: %.4f' % loss, '| test accuracy: %.2f' % accuracy)
            
        train_loss_dict[epoch] = (t_loss/len(train_loader))
        val_loss_dict[epoch] = (v_loss/v_idx)
        # print(train_loss_dict, val_loss_dict)     
    torch.save(model, model_path)
    print("Model Save Done")
        # Save the training loss values
    with open(cfg.TRAINING_LOSS_PICKL, 'wb') as file:
        dump(train_loss_dict, file)

    # Save the validation loss values
    with open(cfg.VALIDATION_LOSS_PICKL, 'wb') as file:
        dump(val_loss_dict, file)
    plot_loss_accuracy(cfg.TRAINING_LOSS_PICKL, cfg.VALIDATION_LOSS_PICKL)
    print("Training Plot Save")


if __name__ ==   "__main__":
    train()
    # plot_loss_accuracy(cfg.TRAINING_LOSS_PICKL, cfg.VALIDATION_LOSS_PICKL)
