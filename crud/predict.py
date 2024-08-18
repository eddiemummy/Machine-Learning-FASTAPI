import os.path

import pandas as pd
from fastapi import APIRouter, HTTPException, status, UploadFile, File, Form
from starlette.responses import JSONResponse
from crud.prepare import prepare_and_train
import shutil
from schemas import Algorithms, AlgorithmType

model_router = APIRouter(
    prefix="/model",
    tags=["model"],
)

uploaded_file_info = {}
@model_router.post("/upload", response_description="Upload File")
async def upload_file(
        algorithm_type: AlgorithmType,
        algorithm: Algorithms|None = None,
        target: str = Form(...),
        file: UploadFile = File(...)):
    file_path = f"file/{file.filename}"
    try:
        with open(file_path, 'wb') as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="File could not be saved.")

    try:
        uploaded_file_info["file_path"] = file_path
        uploaded_file_info["algorithm_type"] = algorithm_type.value
        uploaded_file_info["target"] = target
        uploaded_file_info["algorithm"] = algorithm.value if algorithm else None
        return {"message": "File uploaded"}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@model_router.get('/predict', response_description='Predict')
async def predict():
    try:
        file_path = uploaded_file_info.get("file_path")
        algorithm_type = uploaded_file_info.get("algorithm_type")
        target = uploaded_file_info.get("target")
        algorithm = uploaded_file_info.get("algorithm")

        if not file_path:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File does not exist.")

        df = pd.read_csv(file_path)
        result = prepare_and_train(
            df=df,
            algorithm_type=algorithm_type,
            target=target,
            algorithm=algorithm
        )
        return JSONResponse(content=result, status_code=status.HTTP_200_OK)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
