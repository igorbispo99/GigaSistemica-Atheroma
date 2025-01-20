import os
import cv2 as cv
import numpy as np
from tqdm import tqdm

def generate_crops(input_path, output_path, label):
    # generate crops from images
    # we have to divide the imagem in 2 parts: 2x1
    # we will use just bottom part of the image

    # create output path
    if not os.path.exists(os.path.join(output_path, label)):
        os.makedirs(os.path.join(output_path, label))

    # list all images in input path
    images = os.listdir(input_path)

    # iterate over images
    for image in tqdm(images):
        # read image
        img = cv.imread(os.path.join(input_path, image))
        # get image shape
        h, w, _ = img.shape
        # select bottom part of the image
        img = img[h//2:, :]

        cv.imwrite(os.path.join(output_path, f'{label}/{image}'), img)

INPUT_PATH_POSITIVE = '/home/igorbispo/Downloads/giga_new/images'
INPUT_PATH_NEGATIVE = '/home/igorbispo/Downloads/sem_ateroma/images'

OUTPUT_PATH = '/mnt/data/giga_classifier/crops'

# save equal number of positive and negative crops
generate_crops(INPUT_PATH_POSITIVE, OUTPUT_PATH, '1')
generate_crops(INPUT_PATH_NEGATIVE, OUTPUT_PATH, '0')

