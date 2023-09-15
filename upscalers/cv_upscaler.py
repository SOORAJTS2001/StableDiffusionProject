import cv2
from cv2 import dnn_superres


class UpscaleImage:
    def __init__(self, sample_image_np_array, model_path, upscaled_image_path):
        self.model_path = model_path
        self.sample_image_np_array = sample_image_np_array
        self.upscaled_image_path = upscaled_image_path

    def upscale(self):
        print((self.model_path))
        # Create an SR object
        sr = dnn_superres.DnnSuperResImpl_create()

        # Read image
        image = self.sample_image_np_array

        # Read the desired model
        # path = "/Users/u254428/PycharmProjects/diffusionbee-stable-diffusion-ui/EDSR_x4.pb"
        sr.readModel(self.model_path)

        print("Read model")

        # Set the desired model and scale to get correct pre- and post-processing
        sr.setModel("edsr", 4)
        print("Set model")

        # Upscale the image
        result = sr.upsample(image)
        print("upscaled")

        # Save the image
        cv2.imwrite(self.upscaled_image_path, result)
