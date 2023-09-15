from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from starlette.staticfiles import StaticFiles
from pathlib import Path
from controllers import generateImage


app = FastAPI()
generated_image_path = str(Path('generated_image').resolve())
upscaled_image_path = str(Path('upscaled_image').resolve())
model_path = str(Path('models').resolve()) + '/RRDB_ESRGAN_x4.pth'
templates = Jinja2Templates(directory="frontend/templates")
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")
app.mount("/upscaled_image", StaticFiles(directory="upscaled_image"), name="static")


@app.get("/home")
async def read_root(request: Request):
    img_path, image_name = generateImage("Kunnamkulam", generated_image_path, upscaled_image_path, model_path)
    return templates.TemplateResponse("index.html", {"request": request,
                                                     "image_url": "upscaled_image/"+image_name})
