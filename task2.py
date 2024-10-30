import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import math
def reconstruct_image_with_target_image(original_img, target_img, ratio):
    f_original = np.fft.fft2(original_img)
    f_original_shift = np.fft.fftshift(f_original)
    f_target = np.fft.fft2(target_img)
    f_target_shift = np.fft.fftshift(f_target)

    amplitude_original = np.abs(f_original_shift)
    phase_original = np.angle(f_original_shift)
    amplitude_target = np.abs(f_target_shift)

    rows, cols = original_img.shape
    mask = np.zeros((rows, cols), np.uint8)
    mask[rows//4:3*rows//4, cols//4:3*cols//4] = 1
    amplitude_new = ((1-ratio)*amplitude_original + ratio*amplitude_target)*mask + amplitude_original*(1-mask)

    f_combined_shift = amplitude_new * np.exp(phase_original * 1j)
    f_combined = np.fft.ifftshift(f_combined_shift)

    # combined_img = np.fft.ifft2(f_combined)
    combined_img = np.clip(np.abs(combined_img), 0, 255).astype(np.uint8)

    combined_img = np.abs(combined_img).astype(np.uint8)

    return combined_img
def process_images_in_batch(data_name, ratio, domain1_path, target_domain_path, output_path):
    domain1_images = os.listdir(domain1_path)
    target_domain_images = os.listdir(target_domain_path)
    if data_name == 'ODOC':
        target_img_name = np.random.choice(target_domain_images)
        target_img = Image.open(os.path.join(target_domain_path, target_img_name))
        target_img_array = np.array(target_img)

        for img1_name in domain1_images:
            img1 = Image.open(os.path.join(domain1_path, img1_name))
            img1_array = np.array(img1)

            combined_channels = []
            for i in range(3):
                combined_img = reconstruct_image_with_target_image(img1_array[:, :, i], target_img_array[:, :, i], ratio)
                combined_channels.append(combined_img)
                
            rgb_image = np.stack(combined_channels, axis = -1)
            Image.fromarray(rgb_image).save(os.path.join(output_path, f"{img1_name}"))

    else:
        target_img_name = np.random.choice(target_domain_images)
        target_img = Image.open(os.path.join(target_domain_path, target_img_name)).convert('L')
        target_img_array = np.array(target_img)
        

        for img1_name in domain1_images:
            img1 = Image.open(os.path.join(domain1_path, img1_name)).convert('L')
            img1_array = np.array(img1)

            combined_img = reconstruct_image_with_target_image(img1_array, target_img_array, ratio)
            rgb_image = np.stack((combined_img, ) * 3, axis = -1)

            Image.fromarray(rgb_image).save(os.path.join(output_path, f"{img1_name}"))
data_name = 'ODOC'
target_domains = ['Domain2', 'Domain3', 'Domain4', 'Domain5']
ratio = 0.7
type = 'test'

for target_domain in target_domains:
    domain1_path = f'/home/lyf/FedICRA/data/{data_name}/Domain1/{type}/imgs'
    target_domain_path = f'/home/lyf/FedICRA/data/{data_name}/{target_domain}/{type}/imgs'
    output_path = f'/home/lyf/FedICRA/data/{data_name}/Domain1/{type}/origin_imgs_ratio_{ratio}/to{target_domain}'

    os.makedirs(output_path, exist_ok=True)

    process_images_in_batch(data_name, ratio, domain1_path, target_domain_path, output_path)