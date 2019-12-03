env=("../")
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import numpy as np
from keras.models import Model
from keras.layers import Input,Conv2D, LeakyReLU, UpSampling2D, Conv2DTranspose, Add, ReLU
from keras import backend as K
import tensorflow as tf

class LapSRN:
    def __init__(self,optimizer):
        self.optimizer = optimizer
        
        
        
    def feature_embedding(self):
        _in = Input((None,None,64))
        # Conv1
        x = Conv2D(64, (3, 3),
                   activation='relu',
                   padding='same',
                   kernel_initializer = 'he_normal',
                   name='feat_emb1')(_in)
        x = LeakyReLU(alpha=0.2)(x)
        # Conv2
        x = Conv2D(64, (3, 3),
                   padding='same',
                   kernel_initializer = 'he_normal',
                   name='feat_emb2')(_in)
        x = LeakyReLU(alpha=0.2)(x)
        # Conv3
        x = Conv2D(64, (3, 3),
                   padding='same',
                   kernel_initializer = 'he_normal',
                   name='feat_emb3')(_in)
        x = LeakyReLU(alpha=0.2)(x)
        # Conv4
        x = Conv2D(64, (3, 3),
                   padding='same',
                   kernel_initializer = 'he_normal',
                   name='feat_emb4')(_in)
        x = LeakyReLU(alpha=0.2)(x)
        # Conv5
        x = Conv2D(64, (3, 3),
                   padding='same',
                   kernel_initializer = 'he_normal',
                   name='feat_emb5')(_in)
        x = LeakyReLU(alpha=0.2)(x)
        return Model(_in,x)

    def feature_upsampling(self):
        def bilinear_init(shape):
            # https://github.com/tensorlayer/tensorlayer/issues/53
            num_channels = shape[3]
            bilinear_kernel = np.zeros([shape[1], shape[2]], dtype=np.float32)
            scale_factor = (shape[1] + 1) // 2
            if shape[1] % 2 == 1:
                center = scale_factor - 1
            else:
                center = scale_factor - 0.5
            for x in range(shape[1]):
                for y in range(shape[1]):
                    bilinear_kernel[x,y] = (1 - abs(x - center) / scale_factor) * \
                                           (1 - abs(y - center) / scale_factor)
            weights = np.zeros((shape[1], shape[2], num_channels, num_channels))
            for i in range(num_channels):
                weights[:, :, i, i] = bilinear_kernel
            return K.variable(weights)
        
        # Start
        _in = Input((None,None,64))
        x = Conv2DTranspose(64,(3,3),
                            strides=2,
                            padding='same',
                            kernel_initializer = bilinear_init,
                            name='feat_up')(_in)
        return Model(_in,x)

    def image_upsampling(self):
        def bilinear_init(shape):
            # https://github.com/tensorlayer/tensorlayer/issues/53
            num_channels = shape[3]
            bilinear_kernel = np.zeros([shape[1], shape[2]], dtype=np.float32)
            scale_factor = (shape[1] + 1) // 2
            if shape[1] % 2 == 1:
                center = scale_factor - 1
            else:
                center = scale_factor - 0.5
            for x in range(shape[1]):
                for y in range(shape[1]):
                    bilinear_kernel[x,y] = (1 - abs(x - center) / scale_factor) * \
                                           (1 - abs(y - center) / scale_factor)
            weights = np.zeros((shape[1], shape[2], num_channels, num_channels))
            for i in range(num_channels):
                weights[:, :, i, i] = bilinear_kernel
            return K.variable(weights)
        
        # Start
        _in = Input((None,None,3))
        x = Conv2DTranspose(3,(3,3),
                            strides=2,
                            padding='same',
                            kernel_initializer = bilinear_init,
                            name='img_up')(_in)
        return Model(_in,x)
    
    def conv_res(self):
        def residual_block(_input,skip_conn,num):
            # Conv 1
            a = ReLU()(_input)
            a = Conv2D(64, (3, 3),
                       padding='same',
                       kernel_initializer = 'he_normal',
                       name='conv_res1'+str(num))(a)
            # Conv 2
            a = ReLU()(a)
            a = Conv2D(64, (3, 3),
                       padding='same',
                       kernel_initializer = 'he_normal',
                       name='conv_res2'+str(num))(a)
            # Conv 3
            a = ReLU()(a)
            a = Conv2D(64, (3, 3),
                       padding='same',
                       kernel_initializer = 'he_normal',
                       name='conv_res3'+str(num))(a)
            # Conv 4
            a = ReLU()(a)
            a = Conv2D(64, (3, 3),
                       padding='same',
                       kernel_initializer = 'he_normal',
                       name='conv_res4'+str(num))(a)
            # Conv 5
            a = ReLU()(a)
            a = Conv2D(64, (3, 3),
                       padding='same',
                       kernel_initializer = 'he_normal',
                       name='conv_res5'+str(num))(a)
            a = Add(name='res_add'+str(num))([a,skip_conn])
            return a

        residual_blocks = 8

        _in = Input((None,None,64))
        x = residual_block(_in,skip_conn = _in,num = 1)
        for idx in range(residual_blocks-1):
            x = residual_block(x,skip_conn = _in,num = idx+2)

        x = Conv2D(3,(3,3),
                   strides=1,
                   padding='same',
                   activation='sigmoid',
                   kernel_initializer = 'he_normal',
                   name='sub_band1')(x)

        return Model(_in,x)

    def build_LapSRN(self):
        # Parameter Sharing
        shared_feat_emb = self.feature_embedding()
        shared_feat_up = self.feature_upsampling()
        shared_conv_res = self.conv_res()
        shared_img_up = self.image_upsampling()
        
        # Network
        _in = Input((64,64,3))
        x = Conv2D(64,(3,3),strides=1,padding='same',name='conv_in',kernel_initializer = 'he_normal')(_in)
        x = LeakyReLU(alpha=0.2)(x)

        feat_emb_2x = shared_feat_emb(x)
        feat_up_2x = shared_feat_up(feat_emb_2x)
        sub_band_res_2x = shared_conv_res(feat_up_2x)
        img_up_2x = shared_img_up(_in)
        hr_2x_out = Add()([sub_band_res_2x,img_up_2x])


        feat_emb_4x = shared_feat_emb(feat_up_2x)
        feat_up_4x = shared_feat_up(feat_emb_4x)
        sub_band_res_4x = shared_conv_res(feat_up_4x)
        img_up_4x = shared_img_up(hr_2x_out)
        hr_4x_out = Add()([sub_band_res_4x,img_up_4x])

        return Model(_in,[hr_2x_out,hr_4x_out])
    
    def get_LapSRN(self):
        def loss_function(y_true, y_pred):
            '''
            y_true -> Is the array of bicubic downsampled HR images at each level
            y_pred -> Is the addition of sub band residuals from conv_res and upsampling ie img_up
            '''
            r = y_true - y_pred
            # Charbonnier penalty
            epsilon = 1e-3
            p = K.sqrt(K.square(r) + K.square(epsilon))
            return K.mean(p)
        
        def psnr(y_true,y_pred):
            return tf.math.reduce_mean(tf.image.psnr(y_true,y_pred,max_val=1.0))
        
        def ssim(y_true,y_pred):
            return tf.math.reduce_mean(tf.image.ssim(y_true,y_pred,max_val=1.0))
        
        LapSRN = self.build_LapSRN()
        LapSRN.compile(loss=loss_function,
                       metrics= [psnr,ssim],
                       optimizer=self.optimizer)
        
        return LapSRN
    
        