# from PIL import Image, ImageEnhance
# import numpy as np
# import cv2
# import torch
# import os
# from skimage.util import random_noise
# import matplotlib.pyplot as plt
# from torchvision import transforms

# class DestructiveAttackers():
#     def __init__(self, kernel_size=5, sigma=1):
#         self.kernel_size = kernel_size
#         self.sigma = sigma

#     # def GaussianBlurAttack(self, image_paths, out_paths):
#     #     for (img_path, out_path) in tqdm(zip(image_paths, out_paths)):
#     #         img = cv2.imread(img_path)
#     #         img = cv2.GaussianBlur(img, (self.kernel_size, self.kernel_size), self.sigma)
#     #         cv2.imwrite(out_path, img)
#     def GaussianBlurAttack(self, image_paths, out_paths):
#         for (img_path, out_path) in tqdm(zip(image_paths, out_paths)):
#             # Open the image using PIL
#             img = Image.open(img_path)
#             # Apply Gaussian Blur filter
#             blurred_img = img.filter(ImageFilter.GaussianBlur(self.blur_radius))
#             # Save the blurred image to the output path
#             blurred_img.save(out_path)

#     def RotationAttack

from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from bm3d import bm3d_rgb

class DestructiveAttackers:
    def __init__(self, blur_radius=5, rotation_angle=45, brightness_factor=1.5, exposure_factor=2.0, contrast_factor=2.0):
        self.blur_radius = blur_radius
        self.rotation_angle = rotation_angle
        self.brightness_factor = brightness_factor
        self.exposure_factor = exposure_factor
        self.contrast_factor = contrast_factor

    def GaussianBlurAttack(self, image_paths, out_paths):
        for (img_path, out_path) in tqdm(zip(image_paths, out_paths)):
            img = Image.open(img_path)
            blurred_img = img.filter(ImageFilter.GaussianBlur(self.blur_radius))
            blurred_img.save(out_path)

    def RotateImageAttack(self, image_paths, out_paths):
        for (img_path, out_path) in tqdm(zip(image_paths, out_paths)):
            img = Image.open(img_path)
            rotated_img = img.rotate(self.rotation_angle)
            rotated_img.save(out_path)

    def IncreaseExposure(self, image_path, out_path):
        img = Image.open(image_path)
        enhancer = ImageEnhance.Brightness(img)
        high_exposure_img = enhancer.enhance(self.exposure_factor)
        high_exposure_img.save(out_path)

    def IncreaseContrast(self, image_path, out_path):
        img = Image.open(image_path)
        enhancer = ImageEnhance.Contrast(img)
        high_contrast_img = enhancer.enhance(self.contrast_factor)
        high_contrast_img.save(out_path)

    def BrightenImage(self, image_path, out_path):
        img = Image.open(image_path)
        enhancer = ImageEnhance.Brightness(img)
        bright_img = enhancer.enhance(self.brightness_factor)
        bright_img.save(out_path)

    def CropImage(self, image_path, out_path):
        img = Image.open(image_path)
        left = img.width // 4
        upper = img.height // 4
        right = img.width * 3 // 4
        lower = img.height * 3 // 4
        cropped_img = img.crop((left, upper, right, lower))
        cropped_img.save(out_path)

    def SaveAsJPEG(self, image_path, out_path, quality=80):
        img = Image.open(image_path)
        img.save(out_path, "JPEG", quality=quality)

    def AddGaussianNoise(self, image_path, out_path, mean=0, std_dev=25):
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        gaussian_noise = np.random.normal(mean, std_dev, img.shape).astype(np.float32)
        noisy_img = img + gaussian_noise
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
        cv2.imwrite(out_path, noisy_img)

    def ApplyBM3D(self, image_path, out_path, std_dev=0.1):
        img = Image.open(image_path).convert('RGB')
        y_est = bm3d_rgb(np.array(img) / 255, std_dev)
        plt.imsave(out_path, np.clip(y_est, 0, 1), cmap='gray', vmin=0, vmax=1)


    


            