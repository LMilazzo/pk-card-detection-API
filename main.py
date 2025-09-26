from fastapi import FastAPI, UploadFile, File, Request
from PIL import Image, UnidentifiedImageError
from sentence_transformers import SentenceTransformer
import pandas as pd
from ultralytics import YOLO
from pydantic import BaseModel
from typing import Optional
import numpy as np
import base64
import boto3
import cv2
import io

#-------------------------------------------------------------------------

#Setup
app = FastAPI()

det_model = YOLO("card_detection.pt")
encode_model = SentenceTransformer("clip-ViT-L-14-LOCAL-CLONE-9-25-2025")
library = pd.read_parquet("ex2_card_data.parquet")

#-------------------------------------------------------------------------

#Validation Model
class B64String(BaseModel):
    b64_string: str

#Helper for image encoding
def base64_encode(img: np.ndarray) -> str:
    _, buffer = cv2.imencode('.jpg', img)
    b = base64.b64encode(buffer).decode('utf-8')
    return b #the encoded string

#Helper for image decoding to PIL Image
def base64_decode_toImage(b64: str) -> Image:
    image_data = base64.b64decode(b64)
    img = Image.open(io.BytesIO(image_data))
    return img #The PIL Image

#-------------------------------------------------------------------------
# END POINT TO DETECT THE CARDS RETURNS THE CROPPED CARD 
# AND ORIGINAL IMAGE WITH BOUNDING BOX
#-------------------------------------------------------------------------

#Validation model for output
class CardDetection(BaseModel):
    Frame: B64String
    Cropped_Card: Optional[B64String]
    Card_Detected: bool

#Detection function
@app.post("/detect-cards")
async def detectCards(frame: UploadFile = File(...)) -> dict:

    content = await frame.read()

    try:
        img = Image.open(io.BytesIO(content)).convert("RGB")
    except UnidentifiedImageError:
        return {"Error" : "Invalid Image Error"}

    frame_cv = np.array(img.copy())[:, :, ::-1].copy()   # RGB -> BGR and read / write privledges

    results = det_model.predict(source=frame_cv, conf=0.5, verbose=False)

    crop_resized = None

    #Consider only the first card detected
    result = results[0]

    for box in result.boxes:

        # Bounding box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        # Confidence and class
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = f"{det_model.names[cls]} {conf:.2f}"

        # Draw box and label
        cv2.rectangle(frame_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame_cv, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        #Crop the image to the card and resize
        cropped = frame_cv[y1:y2, x1:x2]
        crop_resized = cv2.resize(cropped, (245, 340), interpolation=cv2.INTER_CUBIC)


    return CardDetection(
        Frame=B64String(b64_string=base64_encode(frame_cv)),
        Cropped_Card= B64String(b64_string=base64_encode(crop_resized)) if crop_resized is not None else None,
        Card_Detected= crop_resized is not None
    )

#-------------------------------------------------------------------------
# END POINT TO GENERATE THE EMBEDDING FOR AN IMAGE
#-------------------------------------------------------------------------

#Validation model for output
class ImageEmbedding(BaseModel):
    Embedding: list[float]

#Encoding Function
@app.post("/encode-card-from-b64")
async def encodeCard(b64_string: B64String) -> ImageEmbedding:

    img = base64_decode_toImage(b64_string.b64_string)

    emb = encode_model.encode(img)

    return  ImageEmbedding(Embedding=emb.tolist())

#-------------------------------------------------------------------------
# END POINT TO FIND THE MATCHES
#-------------------------------------------------------------------------

#Validation model for a match
class Match(BaseModel):
	id: str #Card ID
	score: float #Cosine similarity

#Matching Function
@app.post("/match1")
async def match1(work: ImageEmbedding) -> dict:

    #Find Match from library


    return Match(id="test", score=0)




