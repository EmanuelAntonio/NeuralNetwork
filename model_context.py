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

# generator_optimizer = tf.keras.optimizers.SGD(1e-3, 0.9)
# discriminator_optimizer = tf.keras.optimizers.SGD(1e-3, 0.9)
# generator_optimizer = tf.keras.optimizers.Adam(1e-3, 0.9, 0.999)
# discriminator_optimizer = tf.keras.optimizers.Adam(1e-3, 0.9, 0.999)
# generator_optimizer = tf.keras.optimizers.Adam(8e-5, 0.5, 0.999)
# discriminator_optimizer = tf.keras.optimizers.Adam(8e-5, 0.5, 0.999)
generator_optimizer = tf.keras.optimizers.Adam(1e-3, 0.6, 0.8)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-3, 0.6, 0.8)

MSE = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

checkpoint_path = "epochs/checkpoints"


def discriminator_loss(real_output, fake_output):
    # real_loss = cross_entropy(tf.zeros_like(real_output), real_output)
    # fake_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
    # return real_loss + fake_loss

    fake_loss = MSE(tf.zeros_like(fake_output), fake_output)
    real_loss = MSE(tf.ones_like(real_output), real_output)

    # fake_loss = tf.nn.l2_loss(tf.zeros_like(fake_output) - fake_output) * 2
    # real_loss = tf.nn.l2_loss(tf.ones_like(real_output) - real_output) * 2
    return real_loss + fake_loss


def pixel_wise_loss(generated_images, original_images, masks):
    # masked_original_images = tf.math.multiply(original_images, masks)
    # masked_generated_imagens = tf.math.multiply(generated_images, masks)
    # mse = MSE(generated_images, original_images)
    # return tf.math.reduce_mean(mse)
    l1 = tf.math.abs(generated_images - original_images)
    l1 = tf.math.reduce_sum(l1) / (original_images.shape[0] * ds.image_default_size[0] * ds.image_default_size[1])
    return l1

    # return 100 - tf.reduce_mean(tf.image.psnr(generated_images, original_images, max_val=0.999))


def generator_loss(fake_output, generated_images, original_images, masks):

    pixel_wise = pixel_wise_loss(generated_images, original_images, masks)
    adv_loss = MSE(tf.ones_like(fake_output), fake_output)
    # adv_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)

    return 0.999 * pixel_wise + 0.001 * adv_loss


@tf.function
def train_step(images, masked_images, masks, generator, discriminator):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        generated_images = generator(masked_images, training=True)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output, generated_images, images, masks)
        disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

        return gen_loss, disc_loss


@tf.function
def validation_tf_function(generator, discriminator, images, masked_images):
    generated_images = generator(masked_images, training=False)
    real_output = discriminator(images, training=False)
    fake_output = discriminator(generated_images, training=False)
    return generated_images, real_output, fake_output


def validation_step(images, masked_images, masks, generator, discriminator, save_image, epoch):
    generated_images, real_output, fake_output = validation_tf_function(generator, discriminator, images, masked_images)
    gen_loss = generator_loss(fake_output, generated_images, images, masks)
    disc_loss = discriminator_loss(real_output, fake_output)
    if save_image:
        os.mkdir("epochs/epoch_" + str(epoch + 1))
        img = ds.restore_normalized_image(generated_images.numpy()[0])
        Image.fromarray(img.astype(np.uint8)).save("epochs/epoch_" + str(epoch + 1) + "/epoch" + "_" + str(epoch + 1) + ".jpg")
        img = ds.restore_normalized_image(images[0].numpy())
        Image.fromarray(img.astype(np.uint8)).save("epochs/epoch_" + str(epoch + 1) + "/epoch_original" + "_" + str(epoch + 1) + ".jpg")
        img = ds.restore_normalized_image(masked_images[0].numpy())
        Image.fromarray(img.astype(np.uint8)).save(
            "epochs/epoch_" + str(epoch + 1) + "/epoch_masked" + "_" + str(epoch + 1) + ".jpg")
    # return np.sum(tf.math.reduce_mean(gen_loss).numpy()), np.sum(tf.math.reduce_mean(disc_loss).numpy())
    return gen_loss, disc_loss, tf.reduce_mean(tf.image.psnr(generated_images, images, max_val=0.999))


def __mask_images(images):
    imgs = images.numpy()
    masked_images = np.ndarray(imgs.shape)
    for i in range(masked_images.shape[0]):
        mask = ds.__random_mask_cubes(masked_images.shape[1:3])
        masked_images[i] = ds.__apply_mask(imgs[i], mask)
    return tf.constant(masked_images)


def train(train_ds, val_ds, epochs, generator, discriminator):
    global checkpoint_path
    gen_loss_list_train = []
    disc_loss_list_train = []
    gen_loss_list_validation = []
    disc_loss_list_validation = []
    psnr_list_validation = []
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
        gen_loss = None
        disc_loss = None
        cont = 0
        for image_batch in train_ds:
            gen_loss1, disc_loss1 = train_step(image_batch[:, 0, ...], image_batch[:, 1, ...], image_batch[:, 2, ...], generator, discriminator)
            if gen_loss is None:
                gen_loss = gen_loss1.numpy()
                disc_loss = disc_loss1.numpy()
            else:
                gen_loss += gen_loss1.numpy()
                disc_loss += disc_loss1.numpy()
            cont += 1

        if gen_loss / cont > 2:
            gen_loss_list_train.append(2)
        else:
            gen_loss_list_train.append(gen_loss / cont)
        if disc_loss / cont > 2:
            disc_loss_list_train.append(2)
        else:
            disc_loss_list_train.append(disc_loss / cont)
        print("Gen_Loss_train: ", gen_loss / cont)
        print("Disc_Loss_train: ", disc_loss / cont)

        plt.clf()
        plt.plot(gen_loss_list_train)
        plt.title("Train Generator Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        # plt.show()
        plt.savefig('train_gen_loss.png')
        plt.clf()
        plt.plot(disc_loss_list_train)
        plt.title("Train Discriminator Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        # plt.show()
        plt.savefig('train_disc_loss.png')

        gen_loss = None
        disc_loss = None
        psnr_epoch = 0
        cont = 0

        for image_batch in val_ds:
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
        gen_loss_list_validation.append(gen_loss/cont)
        disc_loss_list_validation.append(disc_loss / cont)
        psnr_list_validation.append(psnr_epoch / cont)
        print("Gen_Loss: ", gen_loss/cont)
        print("Disc_Loss: ", disc_loss / cont)
        print("Psnr: ", psnr_epoch / cont)


        plt.clf()
        plt.plot(gen_loss_list_validation)
        plt.title("Validation Generator Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        # plt.show()
        plt.savefig('validation_gen_loss.png')
        plt.clf()
        plt.plot(disc_loss_list_validation)
        plt.title("Validation Discriminator Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        # plt.show()
        plt.savefig('validation_disc_loss.png')


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

    # Encoder
    model.add(layers.InputLayer((128, 128, 3)))
    _down_sample(64, model, False)
    _down_sample(128, model)
    _down_sample(256, model)
    _down_sample(512, model)
    # Backbone
    model.add(layers.Conv2D(4000, 1, activation=None, padding="valid"))
    # Decoder
    _up_sample(512, model)
    _up_sample(256, model)
    _up_sample(128, model)
    _up_sample(64, model)
    model.add(layers.Conv2D(3, 3, activation=None, padding="same"))
    model.add(layers.Activation("tanh"))

    # model.summary()
    # os.system("pause")
    return model


def _discriminator_block(outFilters, model, stride = 2, normalize = True):
    model.add(layers.Conv2D(outFilters, 3, padding="same", strides=stride))
    if normalize:
        model.add(tfa.layers.InstanceNormalization())
    model.add(layers.LeakyReLU(0.2))


def make_discriminator_model():
    model = tf.keras.Sequential()

    model.add(layers.InputLayer((128, 128, 3)))
    # model.add(layers.InputLayer((256, 256, 3)))
    _discriminator_block(32, model, 2, False)
    _discriminator_block(64, model)
    _discriminator_block(128, model)
    _discriminator_block(256, model)
    _discriminator_block(512, model)

    model.add(layers.Conv2D(1, 3, strides=1, activation=None, padding="same"))

    # model.summary()
    # os.system("pause")
    return model
