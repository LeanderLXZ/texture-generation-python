import os
import numpy as np
from glob import glob
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, ReLU, \
    LeakyReLU, Dropout, ZeroPadding2D, Concatenate
from tensorflow.keras.losses import BinaryCrossentropy, Reduction
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint


class CycleGAN(Model):
    def __init__(
        self,
        style_generator,
        photo_generator,
        style_discriminator,
        photo_discriminator,
        lambda_cycle=10,
    ):
        super(CycleGAN, self).__init__()
        self.s_gen = style_generator
        self.p_gen = photo_generator
        self.s_dis= style_discriminator
        self.p_dis = photo_discriminator
        self.lambda_cycle = lambda_cycle
        
    def compile(
        self,
        s_gen_optimizer,
        p_gen_optimizer,
        s_dis_optimizer,
        p_dis_optimizer,
        gen_loss_fn,
        dis_loss_fn,
        cycle_loss_fn,
        identity_loss_fn
    ):
        super(CycleGAN, self).compile()
        self.s_gen_opt = s_gen_optimizer
        self.p_gen_opt = p_gen_optimizer
        self.s_dis_opt = s_dis_optimizer
        self.p_dis_opt = p_dis_optimizer
        self.gen_loss_fn = gen_loss_fn
        self.dis_loss_fn = dis_loss_fn
        self.cycle_loss_fn = cycle_loss_fn
        self.identity_loss_fn = identity_loss_fn
        
    def train_step(self, batch_data):
        s_real, p_real = batch_data
        
        with tf.GradientTape(persistent=True) as tape:
            # photo to style back to photo
            s_fake = self.s_gen(p_real, training=True)
            p_cycled = self.p_gen(s_fake, training=True)

            # style to photo back to monet
            p_fake = self.p_gen(s_real, training=True)
            s_cycled = self.s_gen(p_fake, training=True)

            # generating itself
            s_same = self.s_gen(s_real, training=True)
            p_same = self.p_gen(p_real, training=True)

            # discriminator used to check, inputing real images
            dis_s_real = self.s_dis(s_real, training=True)
            dis_p_real = self.p_dis(p_real, training=True)

            # discriminator used to check, inputing fake images
            dis_s_fake = self.s_dis(s_fake, training=True)
            dis_p_fake = self.p_dis(p_fake, training=True)

            # Generator loss
            gen_loss_s = self.gen_loss_fn(dis_s_fake)
            gen_loss_p = self.gen_loss_fn(dis_p_fake)

            # Total cycle consistency loss
            total_cycle_loss = self.cycle_loss_fn(
                s_real, s_cycled, self.lambda_cycle) + self.cycle_loss_fn(
                    p_real, p_cycled, self.lambda_cycle)

            # Total generator loss
            total_gen_loss_s = gen_loss_s + total_cycle_loss + \
                self.identity_loss_fn(s_real, s_same, self.lambda_cycle)
            total_gen_loss_p = gen_loss_p + total_cycle_loss + \
                self.identity_loss_fn(p_real, p_same, self.lambda_cycle)

            # Discriminator loss
            dis_loss_s = self.dis_loss_fn(dis_s_real, dis_s_fake)
            dis_loss_p = self.dis_loss_fn(dis_p_real, dis_p_fake)

        # Gradients for generator and discriminator
        s_gen_gradients = tape.gradient(total_gen_loss_s,
                                        self.s_gen.trainable_variables)
        p_gen_gradients = tape.gradient(total_gen_loss_p,
                                        self.p_gen.trainable_variables)
        s_dis_gradients = tape.gradient(dis_loss_s,
                                        self.s_dis.trainable_variables)
        p_dis_gradients = tape.gradient(dis_loss_p,
                                        self.p_dis.trainable_variables)

        # Apply the gradients to the optimizer
        self.s_gen_opt.apply_gradients(zip(s_gen_gradients,
                                           self.s_gen.trainable_variables))
        self.p_gen_opt.apply_gradients(zip(p_gen_gradients,
                                           self.p_gen.trainable_variables))
        self.s_dis_opt.apply_gradients(zip(s_dis_gradients,
                                           self.s_dis.trainable_variables))
        self.p_dis_opt.apply_gradients(zip(p_dis_gradients,
                                           self.p_dis.trainable_variables))
        
        return {
            'gen_loss_s': total_gen_loss_s,
            'gen_loss_p': total_gen_loss_p,
            'dis_loss_s': dis_loss_s,
            'dis_loss_p': dis_loss_p
        }


class StyleStransfer(object):
    
    def __init__(self, output_channels, image_size, n_epochs):
        self.output_channels = output_channels
        self.image_size = image_size
        self.n_epochs = n_epochs
        self.strategy = tf.distribute.get_strategy()
        self.autotune = tf.data.experimental.AUTOTUNE

        self.style_generator = None
        self.photo_generator = None
        self.style_discriminator = None
        self.photo_discriminator = None
        
    def down_sample(self, n_filters, kernel_size, add_norm=True):
        x = Sequential()
        x.add(Conv2D(
            n_filters,
            kernel_size,
            strides=2,
            padding='same',
            kernel_initializer=tf.random_normal_initializer(0., 0.02), 
            use_bias=False))
        if add_norm:
            x.add(InstanceNormalization(
                gamma_initializer=RandomNormal(mean=0.0, stddev=0.02)))
        x.add(LeakyReLU())
        return x
    
    def up_sample(self, n_filters, kernel_size, add_dropout=False):
        x = Sequential()
        x.add(Conv2DTranspose(
            n_filters,
            kernel_size,
            strides=2,
            padding='same',
            kernel_initializer=tf.random_normal_initializer(0., 0.02),
            use_bias=False))
        x.add(InstanceNormalization(
            gamma_initializer=RandomNormal(mean=0.0, stddev=0.02)))
        if add_dropout:
            x.add(Dropout(0.5))
        x.add(ReLU())
        return x
    
    def build_generator(self):
        
        inputs = Input(shape=[*self.image_size, 3])
        
        down_layers = [
            self.down_sample(64, 4, add_norm=False),
            self.down_sample(128, 4),
            self.down_sample(256, 4),
            self.down_sample(512, 4),
            self.down_sample(512, 4),
            self.down_sample(512, 4),
            self.down_sample(512, 4),
            self.down_sample(512, 4),
        ]

        up_layers = [
            self.up_sample(512, 4, add_dropout=True),
            self.up_sample(512, 4, add_dropout=True),
            self.up_sample(512, 4, add_dropout=True),
            self.up_sample(512, 4),
            self.up_sample(256, 4),
            self.up_sample(128, 4),
            self.up_sample(64, 4),
        ]

        # Downsampling
        x = inputs
        skip_layers = []
        for d in down_layers:
            x = d(x)
            skip_layers.append(x)

        skip_layers = reversed(skip_layers[:-1])
        # Upsampling
        for u, s in zip(up_layers, skip_layers):
            x = u(x)
            x = Concatenate()([x, s])
            
        outputs = Conv2DTranspose(
            self.output_channels,
            4,
            strides=2,
            padding='same',
            kernel_initializer=tf.random_normal_initializer(0., 0.02),
            activation='tanh')(x)

        return Model(inputs=inputs, outputs=outputs)
    
    def build_discriminator(self):
        inputs = Input(shape=[*self.image_size, 3], name='input_image')
        x = self.down_sample(64, 4, False)(inputs)
        x = self.down_sample(128, 4)(x)
        x = self.down_sample(256, 4)(x)
        x = ZeroPadding2D()(x)
        x = Conv2D(
            512,
            4,
            strides=1,
            kernel_initializer=tf.random_normal_initializer(0., 0.02),
            use_bias=False)(x)
        x = InstanceNormalization(
            gamma_initializer=RandomNormal(mean=0.0, stddev=0.02))(x)
        x = LeakyReLU()(x)
        x = ZeroPadding2D()(x)
        outputs = Conv2D(
            1,
            4,
            strides=1,
            kernel_initializer=tf.random_normal_initializer(0., 0.02))(x)

        return tf.keras.Model(inputs=inputs, outputs=outputs)
    
    def get_dis_loss(self, real_img, generated_img):
        with self.strategy.scope():
            real_loss = BinaryCrossentropy(
                from_logits=True, reduction=Reduction.NONE)(
                    tf.ones_like(real_img), real_img)
            generated_loss = BinaryCrossentropy(
                from_logits=True, reduction=Reduction.NONE)(
                    tf.zeros_like(generated_img), generated_img)
            return (real_loss + generated_loss) * 0.5
    
    def get_gen_loss(self, generated_img):
        with self.strategy.scope():
            generated_loss = BinaryCrossentropy(
                from_logits=True, reduction=Reduction.NONE)(
                    tf.ones_like(generated_img), generated_img)
            return generated_loss
    
    def get_cycle_loss(self, real_img, cycled_img, LAMBDA):
        with self.strategy.scope():
            return LAMBDA * tf.reduce_mean(tf.abs(real_img - cycled_img))
        
    def get_identity_loss(self, real_img, same_img, LAMBDA):
        with self.strategy.scope():
            return LAMBDA * 0.5 * tf.reduce_mean(tf.abs(real_img - same_img))
    
    def build_and_compile_model(self):
        with self.strategy.scope():

            print('Building generators...')
            self.style_generator = self.build_generator()
            self.photo_generator = self.build_generator()
            print('Building discriminators...')
            self.style_discriminator = self.build_discriminator()
            self.photo_discriminator = self.build_discriminator()
            
            print('Building CycleGAN model...')
            model = CycleGAN(
                self.style_generator,
                self.photo_generator,
                self.style_discriminator,
                self.photo_discriminator)
            
            print('Compiling CycleGAN model...')
            model.compile(
                s_gen_optimizer=Adam(2e-4, beta_1=0.5),
                p_gen_optimizer=Adam(2e-4, beta_1=0.5),
                s_dis_optimizer=Adam(2e-4, beta_1=0.5),
                p_dis_optimizer=Adam(2e-4, beta_1=0.5),
                gen_loss_fn=self.get_gen_loss,
                dis_loss_fn=self.get_dis_loss,
                cycle_loss_fn=self.get_cycle_loss,
                identity_loss_fn=self.get_identity_loss)
            
            return model
    
    def train_batch(self, batch_generator, train_path, results_path):
        
        model = self.build_and_compile_model()
        os.makedirs(results_path, exist_ok=True)
        
        print('Training...')
        for e in range(self.n_epochs):
            batch_gen = batch_generator(train_path, batch_size=1)
            for i, (batch_X, batch_y) in enumerate(batch_gen):
                
                model.train_on_batch(batch_X, batch_y)
                
                if i % 100 == 0:
                    print ('-' * 70)
                    print ('Epoch {} | Batch {}'.format(e, i))
                    to_monet = self.style_generator(np.array(batch_y))
                    plt.subplot(1, 2, 1)
                    plt.title('Original Photo')
                    plt.xticks([])
                    plt.yticks([])
                    plt.imshow(batch_y[0] * 0.5 + 0.5)
                    plt.subplot(1, 2, 2)
                    plt.title('Style-transferred Photo')
                    plt.xticks([])
                    plt.yticks([])
                    plt.imshow(to_monet[0] * 0.5 + 0.5)
                    plt.savefig(
                        results_path + '/epoch_{}_batch_{}.png'.format(e, i))
    
    def train(self, data_loader, train_path, model_name):
        
        model = self.build_and_compile_model()

        imgs_style, imgs_photo = data_loader(train_path)
        
        model_path = 'weights.' + model_name + '.hdf5'
        
        checkpointer = ModelCheckpoint(
            filepath=model_path,
            verbose=1)
        
        print('Training...')
        model.fit(imgs_style,
                  imgs_photo,
                  batch_size=1,
                  epochs=self.n_epochs,
                  callbacks=[checkpointer],
                  verbose=1)
        
        history = model.history

        plt.figure()
        plt.plot([np.mean(i) for i in history.history['gen_loss_s']])
        plt.plot([np.mean(i) for i in history.history['gen_loss_p']])
        plt.plot([np.mean(i) for i in history.history['dis_loss_s']])
        plt.plot([np.mean(i) for i in history.history['dis_loss_p']])
        plt.title('Model Loss')
        plt.ylabel('loss')
        plt.xlabel('step')
        plt.legend(['gen_loss_s', 'gen_loss_p', 'dis_loss_s', 'dis_loss_p'], 
                   loc='upper left')
        plt.savefig('loss.png')
        
        # model.load_weights(model_path)

        return model
