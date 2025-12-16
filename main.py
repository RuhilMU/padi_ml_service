from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import io
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Config
MODEL_PATH = "tuned_rice_disease_densenet121.pth"

# DenseNet121
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class_names = [
    'bacterial_leaf_blight', 
    'brown_spot',
    'healthy', 
    'narrow_brown_spot', 
    'leaf_scald', 
    'leaf_blast', 
]

logger.info("Loading PyTorch model...")
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(MODEL_PATH, map_location=device)
    model.eval()
    logger.info(f"Model loaded successfully on {device}!")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

def preprocess_image(image_bytes):
    """Convert image bytes to tensor ready for prediction"""
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert("RGB")
    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor

@app.get("/")
def read_root():
    return {
        "status": "ML Service is running",
        "model": "DenseNet121",
        "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        "classes": class_names
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # 1. Read file
        contents = await file.read()
        logger.info(f"Received image: {file.filename}, size: {len(contents)} bytes")
        
        # 2. Preprocess
        processed_image = preprocess_image(contents)
        
        # 3. Predict
        with torch.no_grad():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            processed_image = processed_image.to(device)
            outputs = model(processed_image)
            probabilities = F.softmax(outputs, dim=1)
        
        # 4. Results
        confidence, class_idx = torch.max(probabilities, 1)
        predicted_class = class_names[class_idx.item()]
        confidence_value = confidence.item()
        
        logger.info(f"Prediction: {predicted_class} with confidence {confidence_value:.4f}")
        
        return {
            "predictedClass": predicted_class,
            "confidence": float(confidence_value)
        }

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")