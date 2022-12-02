import dataset as ds
import os
import model_context as md
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import random


# paths = ds.generate_image_paths()
paths_train, paths_validation = ds.generate_paths_from_folder("./dataset/img_celeba/", 200000, 0.9)

train_ds = ds.load_dataset_map(paths_train, 16)
val_ds = ds.load_dataset_map(paths_validation, 16)

#######################################################################################################################
generator = md.make_generator_model()
discriminator = md.make_discriminator_model()

md.train(train_ds, val_ds, 50, generator, discriminator)

generator.save("generator_model")
discriminator.save("discriminator_model")


########################################################################################################################

# model_context = md.load_model("C:/Users/emanu/Desktop/openimages/inpainting/200kCeleb/epochs/checkpoints_generator/", generator_context)
# model_completion = md.load_model("C:/Users/emanu/Desktop/openimages/inpainting/200kCelebNew/epochs/checkpoints_generator/", generator_completion)
#
# img = Image.open("C:/Users/emanu/Desktop/openimages/inpainting/dataset/img_celeba/000063.jpg").resize((128, 128))
# img_np = np.asarray(img)
# img_np = ds.normalize_image(img_np)
# mask = ds.__random_mask_cubes((128, 128))
# img_mask = ds.__apply_mask(np.asarray(img_np), mask) / 255
# print(img_mask.shape)
#
#
#
# img_npp = np.empty((1, 128, 128, 3))
# img_npp[0] = img_np
# img_context = generator_context.predict(img_npp)
# img_completion = generator_completion.predict(img_npp)
#
# img_context = ds.restore_normalized_image(img_context[0])
# img_completion = ds.restore_normalized_image(img_completion[0])
# img_mask = img_mask*255
# img_mask = ds.restore_normalized_image(img_mask)
#
# Image.fromarray(img_context.astype(np.uint8)).save("context.jpg")
# Image.fromarray(img_completion.astype(np.uint8)).save("completion.jpg")
# Image.fromarray(img_mask.astype(np.uint8)).save("mask.jpg")


