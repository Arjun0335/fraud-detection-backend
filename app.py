from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import boto3, tempfile, os

app = FastAPI()

bucket = "fraudet"
prefix = "fraud_detection_model/"

def load_model():
    s3 = boto3.client("s3")
    tmpdir = tempfile.mkdtemp()

    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    for obj in response.get("Contents", []):
        key = obj["Key"]
        path = os.path.join(tmpdir, key.replace(prefix, ""))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        s3.download_file(bucket, key, path)

    return pipeline("text-classification", model=tmpdir, tokenizer=tmpdir)

classifier = load_model()

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(data: TextInput):
    result = classifier(data.text)[0]
    return {
        "label": result["label"],
        "confidence": result["score"]
    }
