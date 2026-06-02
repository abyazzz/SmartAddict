from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
INSTANCE_DIR = BASE_DIR / "instance"
MODEL_ROOT_DIR = BASE_DIR / "model"
STATUS_DIR = INSTANCE_DIR / "retrain_statuses"
CONFIG_PATH = INSTANCE_DIR / "model_config.json"
DATASET_PATH = BASE_DIR / "dataset-notebook" / "Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv"
NOTEBOOK_PATH = BASE_DIR / "dataset-notebook" / "Tubes_FIX.ipynb"


class Config:
    SECRET_KEY = "smartaddict-ml-secret-2025"
    SQLALCHEMY_DATABASE_URI = "sqlite:///smartaddict.db"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    TEMPLATES_AUTO_RELOAD = True