
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from fastapi.staticfiles import StaticFiles
from pathlib import Path

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your trained model
MODEL = tf.keras.models.load_model("../saved_models/ODIR1.keras")
CLASS_NAMES = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']

@app.get("/ping")
async def ping():
    return {"message": "Hello, I am alive"}

def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert("RGB")  # Ensures 3 channels
    image = image.resize((224, 224))  # Resize if your model expects specific input size
    return np.array(image)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, axis=0)

    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))

    return {
        'class': predicted_class,
        'confidence': confidence
    }

# Serve static files from the "frontend" directory
app.mount("/", StaticFiles(directory=Path(__file__).parent / "frontend", html=True), name="frontend")

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
