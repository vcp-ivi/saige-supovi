import multiprocessing
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / "outputs" / "ultralytics_config"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "outputs" / "matplotlib_config"))

from ultralytics import YOLO

from config import *


if __name__ == "__main__":
    
    # train custom dataset from pretrained yolo of specific size
    model = YOLO(MODEL)

    wandb_run = None
    load_dotenv()
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key:
        import wandb
        from ultralytics.utils.callbacks.wb import callbacks as wb_callbacks

        os.environ["WANDB_API_KEY"] = wandb_api_key
        wandb_run = wandb.init(
            project=PROJECT_WNDB,
            entity=ENTITY_WNDB,
            group=GROUP_WNDB,
            name=RUN_WNDB,
        )

        # Add W&B callbacks, including model artifact upload.
        for event, func in wb_callbacks.items():
            model.add_callback(event, func)
    else:
        os.environ["WANDB_MODE"] = "disabled"

    model.train(
        data=DATA_YAML,
        device=DEVICE,
        project=PROJECT_NAME_YOLO,
        name=FOLDER_NAME_YOLO,
        **HYPERPARAMS,
        #cfg=os.path.join(os.getcwd(),'custom_args.yaml'),
    )
    
    if wandb_run is not None:
        wandb.finish()
