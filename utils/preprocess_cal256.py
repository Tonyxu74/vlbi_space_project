import os
from myargs import args
import numpy as np
from tqdm import tqdm
from PIL import Image
import random


def find_file(root_dir, endswith):
    all_files = []
    for path, subdirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(endswith):
                all_files.append(os.path.join(path, file))

    return all_files


def process_cal256(path):
    image_data = find_file(path, '.jpg')
    random.shuffle(image_data)
    image_data = image_data[:args.trainNum]

    if not os.path.exists('../data/arrays/cal256'):
        os.mkdir('../data/arrays/cal256')

    image_num = 0

    for image_pth in tqdm(image_data):
        # open greyscale image
        image = Image.open(image_pth).convert('L')
        square_size = min(image.size[0], image.size[1])
        image = image.crop(box=(0, 0, square_size, square_size))
        image = image.resize(args.imageDims, resample=Image.NEAREST)
        image = np.asarray(image).astype(np.float32) / 255

        fft_image = np.fft.fft2(image)
        fft_image = np.fft.fftshift(fft_image)

        amp_image = np.abs(fft_image)
        phase_image = np.angle(fft_image, deg=False)

        np.save('../data/arrays/cal256/{}_amp_img.npy'.format(image_num), amp_image)
        np.save('../data/arrays/cal256/{}_phase_img.npy'.format(image_num), phase_image)

        image_num += 1


if __name__ == "__main__":
    process_cal256('../raw_data/256_ObjectCategories')
