import pickle
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from network import utils

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "network" / "models"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# Nạp toàn bộ 5 mô hình (PyTorch DL + Scikit-Learn ML)
models = {
    "ReLU": {
        "type": "pytorch",
        "model": utils.LoadModel(MODEL_PATH / "ReLU" / "ReLU.pth", "ReLU"),
    },
    "SiLU": {
        "type": "pytorch",
        "model": utils.LoadModel(MODEL_PATH / "SiLU" / "SiLU.pth", "SiLU"),
    },
}

ml_files = {
    "Multinomial NB": PROCESSED_DIR / "nb_model.pkl",
    "Linear SVM": PROCESSED_DIR / "svm_model.pkl",
    "Logistic Regression": PROCESSED_DIR / "lr_model.pkl",
}

for name, path in ml_files.items():
    if path.exists():
        with open(path, "rb") as f:
            models[name] = {"type": "sklearn", "model": pickle.load(f)}


class GenerateRequest(BaseModel):
    model_type: str
    prompt: str


# API lấy danh sách các model khả dụng
@app.get("/models")
def list_model():
    return {"models": list(models.keys())}


# API dự đoán intent cho câu prompt
@app.post("/generate")
def generate(request: GenerateRequest):
    entry = models.get(request.model_type)
    if entry is None:
        raise HTTPException(status_code=404, detail="Unknown model type")

    if entry["type"] == "pytorch":
        vector = utils.Text_to_vector(request.prompt, for_pytorch=True)
        prediction = utils.Predict(entry["model"], vector)
        res = utils.Output_to_text(prediction, is_pytorch_logits=True)
    else:
        vector = utils.Text_to_vector(request.prompt, for_pytorch=False)
        prediction = entry["model"].predict(vector)
        res = utils.Output_to_text(prediction, is_pytorch_logits=False)

    return {"model_type": request.model_type, "result": res}
