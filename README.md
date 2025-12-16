# Rice Disease Detection - ML Service

FastAPI service for rice disease detection using PyTorch DenseNet121 model.

## Model Information

- **Model**: DenseNet121 (fine-tuned)
- **Model File**: `tuned_rice_disease_densenet121.pth`
- **Input Size**: 224x224 RGB images
- **Classes**: 6 rice disease categories
  - `bacterial_leaf_blight`
  - `brown_spot`
  - `healthy`
  - `narrow_brown_spot`
  - `leaf_scald`
  - `leaf_blast`

## Setup

### 1. Create Virtual Environment

```bash
# Virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Service

```bash
# Development mode
uvicorn main:app --reload --port 5000

# Production mode
uvicorn main:app --host 0.0.0.0 --port 5000
```

## API Endpoints

### GET `/`
Health check endpoint

**Response:**
```json
{
  "status": "ML Service is running",
  "model": "DenseNet121",
  "device": "cpu",
  "classes": ["leaf_scald", "bacterial_leaf_blight", ...]
}
```

### POST `/predict`
Predict rice disease from image

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: `file` (image file - jpg, jpeg, png)

**Response:**
```json
{
  "predictedClass": "leaf_blast",
  "confidence": 0.9234
}
```

**Error Response:**
```json
{
  "detail": "Prediction failed: <error message>"
}
```

## Docker Deployment

```bash
# Build image
docker build -t rice-disease-ml .

# Run container
docker run -p 5000:5000 rice-disease-ml
```

## Testing

### Using cURL

```bash
curl -X POST "http://localhost:5000/predict" \
  -F "file=@path/to/rice_leaf.jpg"
```

### Using Python

```python
import requests

url = "http://localhost:5000/predict"
files = {"file": open("rice_leaf.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

## Integration with Backend

The NestJS backend (`padi-check`) calls this service via HTTP:

1. Backend receives image upload from user
2. Backend sends image to ML service `/predict` endpoint
3. ML service returns prediction and confidence
4. Backend saves result to database and returns to user

**Environment Variable Required:**
```
ML_API_URL=http://your-ml-service-url:5000
```

## Model Details

The model uses standard ImageNet preprocessing:
- Resize to 256x256
- Center crop to 224x224
- Normalize with mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

## Troubleshooting

**Model not loading:**
- Ensure `tuned_rice_disease_densenet121.pth` is in the same directory as `main.py`
- Check PyTorch version compatibility

**Prediction errors:**
- Verify image format (jpg, jpeg, png)
- Check image is not corrupted
- Ensure sufficient memory for model inference

**Connection refused:**
- Verify service is running on correct port
- Check firewall settings
- Ensure `ML_API_URL` is correctly configured in backend
