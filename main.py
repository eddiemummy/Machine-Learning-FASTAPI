from fastapi import FastAPI
from crud.predict import model_router
import uvicorn

app = FastAPI()
app.include_router(model_router)

if __name__ == "__main__":
    uvicorn.run('main:app', reload=True, port=5009)
