from fastapi import FastAPI
import train

app = FastAPI()
@app.get("/")
def get_root():
    return {"message": "Welcome to YOLOv8n!"}

if __name__ == "__main__":
    train.model_training()