from pathlib import Path

from fastapi import FastAPI
from google.cloud import storage
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse
from starlette.staticfiles import StaticFiles

client = storage.Client.from_service_account_json(
    '/Users/u254428/PycharmProjects/hotels-job-providers-properties-modified-since/auth.json')
bucket_name = 'images-cdn'
bucket = client.bucket(bucket_name)

generated_image_path = str(Path('../generated_image').resolve())
upscaled_image_path = str(Path('../upscaled_image').resolve())
model_path = str(Path('../models').resolve()) + '/RRDB_ESRGAN_x4.pth'
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to a specific list of allowed origins if desired
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/upscaled_image", StaticFiles(directory="upscaled_image"), name="upscaled_image")
from txt2img import generateImage


class RequestArgs(BaseModel):
    prompt: str


@app.post("/txt2img")
async def txt2img(args: RequestArgs):
    print(args.prompt)
    img_path, image_name = generateImage(args.prompt, generated_image_path, upscaled_image_path, model_path)
    print("uploading image to the cloud as ", image_name)
    blob = bucket.blob(image_name)
    blob.upload_from_filename(img_path + "/" + image_name)
    print(f"https://storage.googleapis.com/{bucket_name}/{image_name}")
    return f"https://storage.googleapis.com/{bucket_name}/{image_name}"

@app.post("/txt2imglocal")
async def txt2imglocal(args:RequestArgs):
    img_path, image_name = generateImage(args.prompt, generated_image_path, upscaled_image_path, model_path)
    return FileResponse(img_path+"/"+image_name,media_type="png")


