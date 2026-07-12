import os

# General parameters
# -------------------
PROJECT_NAME_YOLO = os.path.join(os.getcwd(), "outputs", "yolo_results")  # in model.train(); weights and results are saved in PROJECT_NAME_YOLO/FOLDER_NAME_YOLO
FOLDER_NAME_YOLO = "tag_yolo26s_baseline"
DATA_PATH = os.path.join(os.getcwd(), "data", "yolo") # name of the folder containing yolo export
DATA_YAML = os.path.join(DATA_PATH, 'data.yaml')


# Hyperparameters (sequential)
# -----------------------------
DEVICE = 0
MODEL = "yolo26s.pt"
HYPERPARAMS = {
    "epochs": 100,
    # Ultralytics keeps best.pt from the epoch with the highest validation
    # fitness. For detection, fitness is box mAP50-95 on the validation set.
    # Early stopping triggers after this many epochs without improvement.
    "patience": 25,
    "batch": -1,
    "cos_lr": False,
    "workers": 0,
    "imgsz": 1024,
    "mosaic": 0.5,
    "crop_fraction" : 0.0,
    "scale" : 0.0,
    "translate" : 0.0,
    #"degrees" : 180,
}


# Wandb
# ------
#LOCAL_RUN = False  # if True, skip wandb logging
PROJECT_WNDB = "Test"
ENTITY_WNDB = "emsml"
GROUP_WNDB = "yolo26"
RUN_WNDB = "tag_yolo26s_baseline"
