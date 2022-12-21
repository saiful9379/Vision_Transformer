import os
class config:
    EPOCHS = 20
    BATCH_SIZE = 20
    LEARNING_RATE = 2e-5
    DATASET_PATH = "./dataset_256X256/dataset"
    # DATASET_PATH = "/home/user/Downloads/dataset_256X256/dataset"
    DATASET_AUG_PATH = "./dataset_256X256/dataset_aug"
    MODEL_PATH = "./model"
    MODEL_NAME = "vit_model.pt"
    CLASS_DICT = "vit_class.json"

    DEBUG_FOR_ORGINAL_DATA = False
    LOG_DIR = "logs"
    os.makedirs(LOG_DIR, exist_ok=True)
    TRAINING_LOSS_PICKL = os.path.join(LOG_DIR, "train_loss.pkl")
    VALIDATION_LOSS_PICKL = os.path.join(LOG_DIR, "val_loss.pkl")
    

    # if DEBUG_FOR_ORGINAL_DATA:
    #     training_data_path = os.path.join(dataset, "train")
    #     validation_data_path = os.path.join(dataset, "test")
    # else:
    #     training_data_path = os.path.join(dataset_aug, "train")
    #     validation_data_path = os.path.join(dataset_aug, "test")


    # model_folder = "models"
    # # os.makedirs(model_path)
    # os.makedirs(model_folder, exist_ok = True)

    # model_path = os.path.join(model_folder, "vit_model.pt")
    # class_path = os.path.join(model_folder, "vit_class.json")

