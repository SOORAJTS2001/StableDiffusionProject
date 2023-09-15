import datetime
import io

import cv2
import numpy as np
import torch
import upscalers.RRDBNet_arch as arch


def upscale(model_path, generated_image_array, upscaled_image_path):
    # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
    device = torch.device('mps')  # if you want to run on CPU, change 'cuda' -> cpu
    # device = torch.device('cpu')

    model = arch.RRDBNet(3, 3, 64, 23, gc=32)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)

    print('Model path {:s}. \nTesting...'.format(model_path))
    img = generated_image_array
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()

    timestamp_name = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ".png"
    cv2.imwrite(upscaled_image_path+"/"+timestamp_name, output)
    return upscaled_image_path,timestamp_name
