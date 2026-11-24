from fastapi import FastAPI, HTTPException
from pathlib import Path
from network import utils
from pydantic import BaseModel

app = FastAPI()
modelPath = Path(__file__).resolve().parent.parent / "network" / "models"
print(modelPath)

#put model here
models = {
    "ReLU": utils.LoadModel(modelPath / "ReLU" / "ReLU.pth", "ReLU"),
    "SiLU": utils.LoadModel(modelPath / "SiLU" / "SiLU.pth", "SiLU")
}

class GenerateRequest(BaseModel):
    model_type: str
    prompt: str

#use this api to get model the user want to use
@app.get("/models")
def list_model():
    return {"models": list(models.keys())}

#use this api to get output
@app.post("/generate")
def generate(request: GenerateRequest):
    model = models.get(request.model_type)
    if model is None:
        raise HTTPException(status_code=404, detail="Unknown model type")

    vector = utils.Text_to_vector(request.prompt)
    prediction =  utils.Predict(model, vector)
    res = utils.Output_to_text(prediction)
    return {"model_type": request.model_type, "result": res}
