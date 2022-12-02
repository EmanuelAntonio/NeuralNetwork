import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import os, sys, warnings
from tensorflow.keras import layers
import tensorflow_addons as tfa
import time
import dataset as ds

generator_optimizer = tf.keras.optimizers.Adam(8e-5, 0.5, 0.999)
discriminator_optimizer = tf.keras.optimizers.Adam(8e-5, 0.5, 0.999)
# generator_optimizer = tf.keras.optimizers.Adam(1e-3, 0.6, 0.8)
# discriminator_optimizer = tf.keras.optimizers.Adam(1e-3, 0.6, 0.8)

MSE = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

checkpoint_path = "epochs/checkpoints"


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    # real_loss = tf.reduce_sum((tf.ones_like(real_output) - real_output) ** 2)
    # fake_loss = tf.reduce_sum((tf.zeros_like(fake_output) - fake_output) ** 2)
    return real_loss + fake_loss

    # fake_loss = MSE(tf.zeros_like(fake_output), fake_output)
    # real_loss = MSE(tf.ones_like(real_output), real_output)

    # fake_loss = tf.nn.l2_loss(tf.zeros_like(fake_output) - fake_output) * 2
    # real_loss = tf.nn.l2_loss(tf.ones_like(real_output) - real_output) * 2
    # return real_loss + fake_loss


def pixel_wise_loss(generated_images, original_images, masks):
    # masked_original_images = tf.math.multiply(original_images, masks)
    # masked_generated_imagens = tf.math.multiply(generated_images, masks)
    # mse = MSE(generated_images, original_images)
    # return tf.math.reduce_mean(mse)
    # l1 = tf.math.abs(generated_images - original_images)
    # l1 = tf.math.reduce_sum(l1) / (original_images.shape[0] * ds.image_default_size[0] * ds.image_default_size[1])
    # return l1
    return 100 - tf.reduce_mean(tf.image.psnr(generated_images[..., 0], original_images[..., 0], max_val=0.999))
    # return tf.reduce_mean((generated_images - original_images)**2)


def generator_loss(fake_output, generated_images, original_images, masks):
    pixel_wise = pixel_wise_loss(generated_images, original_images, masks)
    adv_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
    # adv_loss = tf.reduce_mean((tf.ones_like(fake_output) - fake_output) ** 2)
    return pixel_wise + adv_loss


@tf.function
def train_step(images, masked_images, masks, generator, discriminator):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        generated_images = generator(masked_images[..., 0], training=True)
        real_output = discriminator(images[..., 0], training=True)
        fake_output = discriminator(generated_images[..., 0], training=True)

        gen_loss = generator_loss(fake_output, generated_images, images, masks)
        disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


@tf.function
def validation_tf_function(generator, discriminator, images, masked_images):
    generated_images = generator(masked_images[:, :, :, 0], training=False)
    real_output = discriminator(images[:, :, :, 0], training=False)
    fake_output = discriminator(generated_images, training=False)
    return generated_images, real_output, fake_output


def validation_step(images, masked_images, masks, generator, discriminator, save_image, epoch):
    generated_images, real_output, fake_output = validation_tf_function(generator, discriminator, images, masked_images)
    gen_loss = generator_loss(fake_output, generated_images, images, masks)
    disc_loss = discriminator_loss(real_output, fake_output)
    if save_image:
        os.mkdir("epochs/epoch_" + str(epoch + 1))
        img = ds.restore_normalized_image(generated_images[0].numpy())
        img_np = np.empty((ds.image_default_size[0], ds.image_default_size[1], 3))
        img_np[..., 0] = np.reshape(img, ds.image_default_size)
        img_np[..., 1] = img_np[..., 0]
        img_np[..., 2] = img_np[..., 0]
        Image.fromarray(img_np.astype(np.uint8)).save("epochs/epoch_" + str(epoch + 1) + "/epoch" + "_" + str(epoch + 1) + ".jpg")
        img = ds.restore_normalized_image(images[0].numpy())
        img_np[..., 0] = np.reshape(img[:, :, 0], ds.image_default_size)
        img_np[..., 1] = img_np[..., 0]
        img_np[..., 2] = img_np[..., 0]
        Image.fromarray(img_np.astype(np.uint8)).save("epochs/epoch_" + str(epoch + 1) + "/epoch_original" + "_" + str(epoch + 1) + ".jpg")
        img = ds.restore_normalized_image(masked_images[0].numpy())
        # img_np[:, :, 0] = np.reshape(img, ds.image_default_size)
        # img_np[:, :, 1] = np.reshape(img, ds.image_default_size)
        # img_np[:, :, 2] = np.reshape(img, ds.image_default_size)
        Image.fromarray(img.astype(np.uint8), "YCbCr").save(
            "epochs/epoch_" + str(epoch + 1) + "/epoch_masked" + "_" + str(epoch + 1) + ".jpg")
    # return np.sum(tf.math.reduce_mean(gen_loss).numpy()), np.sum(tf.math.reduce_mean(disc_loss).numpy())
    return gen_loss, disc_loss, tf.reduce_mean(tf.image.psnr(generated_images[..., 0], images[..., 0], max_val=0.999))


def __mask_images(images):
    imgs = images.numpy()
    masked_images = np.ndarray(imgs.shape)
    for i in range(masked_images.shape[0]):
        mask = ds.__random_mask_cubes(masked_images.shape[1:3])
        masked_images[i] = ds.__apply_mask(imgs[i], mask)
    return tf.constant(masked_images)


def train(train_ds, val_ds, epochs, generator, discriminator):
    global checkpoint_path
    gen_loss_list = []
    disc_loss_list = []
    psnr_list = []
    checkpoint_gen = tf.train.Checkpoint(generator)
    checkpoint_disc = tf.train.Checkpoint(discriminator)
    manager_gen = tf.train.CheckpointManager(checkpoint_gen, checkpoint_path + "_generator", epochs, checkpoint_name="checkpoint")
    manager_disc = tf.train.CheckpointManager(checkpoint_disc, checkpoint_path + "_discriminator", epochs, checkpoint_name="checkpoint")
    ds.__clear_folder("./epochs/")
    os.mkdir("epochs/checkpoints_generator/")
    os.mkdir("epochs/checkpoints_discriminator")
    for epoch in range(epochs):
        start = time.time()
        save = True
        for image_batch in train_ds:
            # masked_images = __mask_images(image_batch)
            train_step(image_batch[:, 0, ...], image_batch[:, 1, ...], image_batch[:, 2, ...], generator, discriminator)
            # img = tf.math.multiply(image_batch[:, 0, ...], image_batch[:, 2, ...])
            # img = img.numpy()[0]
            # img = ds.restore_normalized_image(img)
            # img = Image.fromarray(img.astype(np.uint8))
            # img.show()

        gen_loss = None
        disc_loss = None
        psnr_epoch = 0
        cont = 0
        for image_batch in val_ds:
            # masked_images = __mask_images(image_batch)
            gen_loss1, disc_loss1, psnr = validation_step(image_batch[:, 0, ...], image_batch[:, 1, ...], image_batch[:, 2, ...], generator, discriminator, save, epoch)
            psnr_epoch += psnr.numpy()
            if save:
                save = False
            if gen_loss is None:
                gen_loss = gen_loss1.numpy()
                disc_loss = disc_loss1.numpy()
            else:
                gen_loss += gen_loss1.numpy()
                disc_loss += disc_loss1.numpy()
            cont += 1

        manager_gen.save()
        manager_disc.save()
        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
        gen_loss_list.append(gen_loss/cont)
        disc_loss_list.append(disc_loss / cont)
        psnr_list.append(psnr_epoch / cont)
        print("Gen_Loss: ", gen_loss/cont)
        print("Disc_Loss: ", disc_loss / cont)
        print("Psnr: ", psnr_epoch / cont)


        plt.clf()
        plt.plot(gen_loss_list)
        plt.title("Generator Loss")
        # plt.show()
        plt.savefig('gen_loss.png')
        plt.clf()
        plt.plot(disc_loss_list)
        plt.title("Discriminator Loss")
        # plt.show()
        plt.savefig('disc_loss.png')


def load_model(model_path, model):
    checkpoint = tf.train.Checkpoint(model)
    return checkpoint.restore(tf.train.latest_checkpoint(model_path))


def _down_sample(out_feat, model, normalize=True):
    model.add(layers.Conv2D(out_feat, 4, padding="same", strides=2))
    if normalize:
        model.add(layers.BatchNormalization(epsilon=0.8))
    model.add(layers.LeakyReLU(0.2))


def _up_sample(out_feat, model, normalize=True):
    model.add(layers.Conv2DTranspose(out_feat, 4, padding="same", strides=2))
    if normalize:
        model.add(layers.BatchNormalization(epsilon=0.8))
    model.add(layers.ReLU())


def make_generator_model():
    model = tf.keras.Sequential()

    model.add(layers.InputLayer((ds.image_default_size[0], ds.image_default_size[1], 1)))
    model.add(layers.Conv2D(64, 5, padding="same"))
    model.add(layers.Activation("relu"))
    model.add(layers.Conv2D(128, 3, strides=2, padding="same"))
    model.add(layers.Activation("relu"))
    model.add(layers.Conv2D(128, 3, padding="same"))
    model.add(layers.Activation("relu"))

    model.add(layers.Conv2D(256, 3, strides=2, padding="same"))
    model.add(layers.Activation("relu"))
    model.add(layers.Conv2D(256, 3, padding="same"))
    model.add(layers.Activation("relu"))
    model.add(layers.Conv2D(256, 3, padding="same"))
    model.add(layers.Activation("relu"))
    model.add(layers.Conv2D(256, 3, dilation_rate=2, padding="same"))
    model.add(layers.Activation("relu"))
    model.add(layers.Conv2D(256, 3, dilation_rate=4, padding="same"))
    model.add(layers.Activation("relu"))
    model.add(layers.Conv2D(256, 3, dilation_rate=8, padding="same"))
    model.add(layers.Activation("relu"))
    model.add(layers.Conv2D(256, 3, dilation_rate=16, padding="same"))
    model.add(layers.Activation("relu"))
    model.add(layers.Conv2D(256, 3, padding="same"))
    model.add(layers.Activation("relu"))
    model.add(layers.Conv2D(256, 3, padding="same"))
    model.add(layers.Activation("relu"))

    model.add(layers.Conv2DTranspose(128, 4, strides=2, padding="same"))
    model.add(layers.Activation("relu"))
    model.add(layers.Conv2D(128, 3, padding="same"))
    model.add(layers.Activation("relu"))

    model.add(layers.Conv2DTranspose(64, 4, strides=2, padding="same"))
    model.add(layers.Activation("relu"))
    model.add(layers.Conv2D(32, 3, padding="same"))
    model.add(layers.Activation("relu"))
    model.add(layers.Conv2D(1, 3, activation="sigmoid", padding="same"))

    # model.summary()
    # os.system("pause")
    return model


def make_discriminator_model():
    model = tf.keras.Sequential()

    model.add(layers.InputLayer((ds.image_default_size[0], ds.image_default_size[1], 1)))
    # model.add(layers.InputLayer((256, 256, 3)))
    model.add(layers.Conv2D(64, 5, activation="relu", strides=2, padding="same"))
    model.add(layers.Conv2D(128, 5, activation="relu", strides=2, padding="same"))
    model.add(layers.Conv2D(256, 5, activation="relu", strides=2, padding="same"))
    model.add(layers.Conv2D(512, 5, activation="relu", strides=2, padding="same"))
    model.add(layers.Conv2D(512, 5, activation="relu", strides=2, padding="same"))
    model.add(layers.Conv2D(512, 5, activation="relu", strides=2, padding="same"))
    model.add(layers.Dense(1))

    # model.summary()
    # os.system("pause")
    return model
