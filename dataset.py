import pandas as pd
import random
from PIL import Image
import os
import pickle
import shutil
import numpy as np
import os
import tensorflow as tf
import gc
import math

# __all__ = ["generate_dataset", "load_metadata", "load_dataset"]

directory = "./data/"
metadata_file_dir = "./metadata/image_ids.csv"
metadata_file = []
dataset_dir = "./dataset/"
image_default_size = (128, 128)
# image_default_size = (256, 256)


def __clear_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def __random_mask_cubes(shape):
    points = random.randint(8, 32)
    size_min = shape[0]//64
    size_max = shape[0]//8
    mask = np.zeros(shape)
    n = 0
    while n < points:
        size = random.randint(size_min, size_max)
        i = random.randint(size // 2, shape[0] - size // 2)
        j = random.randint(size // 2, shape[1] - size // 2)
        x = -size // 2
        while x < size // 2:
            y = -size // 2
            while y < size // 2:
                mask[i + x, j + y] = 255
                y += 1
            x += 1
        n += 1
    return mask


def __apply_mask(image, mask):

    output = image.copy()

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if mask[i, j] > 128:
                output[i, j] = (1, 1, 1)
    return output


def load_metadata():
    global dataset_dir, metadata_file
    with open(dataset_dir + "metadata.dat", 'rb') as fp:
        metadata_file = pickle.load(fp)


def generate_metadata(num_img):
    global metadata_file_dir, metadata_file, dataset_dir
    ds = pd.read_csv(metadata_file_dir)
    id_list = random.sample(range(len(ds["ImageID"])), num_img)
    for i in id_list:
        metadata_file.append(ds["ImageID"][i])
    with open(dataset_dir + "metadata.dat", 'wb') as fp:
        pickle.dump(metadata_file, fp)


def generate_paths_from_folder(folder, num_img=1000, train_percentage=0.9):
    global metadata_file
    paths = []
    for filename in os.listdir(folder):
        paths.append(os.path.join(folder, filename))
    random.shuffle(paths)
    return paths[0:math.ceil(num_img * train_percentage)], paths[math.ceil(num_img * train_percentage):num_img]


def normalize_image(image):
    return (image / 255) * 0.999


def restore_normalized_image(image):
    img = image * 255 / 0.999
    np.place(img, img > 255, 255)
    return img


def load_dataset_map(image_paths, batch_size):
    print("Loading dataset...")

    def _parser(image_path):
        def _image_load(path):
            # img = Image.open(path.numpy()).resize(image_default_size).convert("YCbCr")
            img = Image.open(path.numpy()).resize(image_default_size).convert("RGB")
            file = np.asarray(img)

            # if file.shape[2] > 3:
            #     file = file[:, :, 0:3]

            # img_new = Image.new("YCbCr", img.size)
            # img_new.paste(img)
            # file = np.asarray(img_new)
            # file = np.load(path.numpy(), allow_pickle=True)/255
            output = np.empty([3, image_default_size[0], image_default_size[1], 3])
            mask_3 = np.empty([image_default_size[0], image_default_size[1], 3])
            mask = __random_mask_cubes(image_default_size)
            file = normalize_image(file)
            masked_image = __apply_mask(file, mask)
            mask_3[..., 0] = mask / 255
            mask_3[..., 1] = mask / 255
            mask_3[..., 2] = mask / 255
            # mask = mask / 255
            output[0] = file
            # output[1] = file
            output[1] = masked_image
            output[2] = mask_3
            # output[1] = masked_image
            return output

        image = tf.py_function(_image_load, [image_path], tf.float32)
        return image

    dataset = tf.data.Dataset.from_tensor_slices(image_paths)

    dataset_size = len(image_paths)
    dataset = dataset.shuffle(buffer_size=dataset_size, reshuffle_each_iteration=True)

    dataset = dataset.map(_parser)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    gc.collect()
    print("Load complete.")
    return dataset


def generate_image_paths():
    global metadata_file, dataset_dir
    paths = []
    for i in metadata_file:
        paths.append(dataset_dir + "images/" + str(i) + ".jpg")
    return paths


def load_dataset_image():
    data_dir = "./dataset/images/"

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.1,
        subset="training",
        seed=123,
        image_size=(128, 128),
        batch_size=32)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.1,
        subset="validation",
        seed=123,
        image_size=(128, 128),
        batch_size=32)

    return train_ds, val_ds
