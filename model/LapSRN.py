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
        self.filters = 64
        self.D = 5
        self.R = 8
        self.kernel = (3,3)
        
    def build_conv(self):
        _in = Input((None,None,64))
        x = Conv2D(self.filters, self.kernel,
                   padding='same',
                   kernel_initializer = 'he_normal',
                   name='conv1')(_in)
        x = LeakyReLU(alpha=0.2)(x)
        for idx in range(self.D-1):
            x = Conv2D(self.filters, self.kernel,
                       padding='same',
                       kernel_initializer = 'he_normal',
                       name='conv'+str(idx+2))(x)
            x = LeakyReLU(alpha=0.2)(x)
        return Model(_in,x)
        
    def feature_embedding(self,shared_conv):
        _in = Input((None,None,64))
        x = shared_conv(_in)
        return Model(_in,x)

    def bilinear_init(self,shape):
        # https://github.com/tensorlayer/tensorlayer/issues/53
        num_channels = shape[3]
        bilinear_kernel = np.zeros([shape[0], shape[1]], dtype=np.float32)
        scale_factor = (shape[0] + 1) // 2
        if shape[1] % 2 == 1:
            center = scale_factor - 1
        else:
            center = scale_factor - 0.5
        for x in range(shape[0]):
            for y in range(shape[1]):
                bilinear_kernel[x,y] = (1 - abs(x - center) / scale_factor) * (1 - abs(y - center) / scale_factor)
        weights = np.zeros((shape[0], shape[1], num_channels, num_channels))
        for i in range(num_channels):
            weights[:, :, i, i] = bilinear_kernel
        return K.variable(weights)

    def feature_upsampling(self): 
        _in = Input((None,None,64))
        x = Conv2DTranspose(64,(4,4),
                            strides=2,
                            padding='same',
                            kernel_initializer = self.bilinear_init,
                            use_bias=False,
                            name='feat_up')(_in)
        return Model(_in,x)

    def image_upsampling(self):
        _in = Input((None,None,3))
        x = Conv2DTranspose(3,(4,4),
                            strides=2,
                            padding='same',
                            kernel_initializer = self.bilinear_init,
                            use_bias=False,
                            name='img_up')(_in)
        return Model(_in,x)
    
    def conv_res(self,shared_conv):

        _in = Input((None,None,64))

        x = shared_conv(_in)
        for idx in range(self.R-1):
            x = shared_conv(x)

        x = Conv2D(3,(3,3),
                   strides=1,
                   padding='same',
                   activation='sigmoid',
                   kernel_initializer = 'he_normal',
                   name='sub_band1')(x)

        return Model(_in,x)


    def build_LapSRN(self):
        # Parameter Sharing
        shared_conv = self.build_conv()
        shared_feat_emb = self.feature_embedding(shared_conv)
        shared_feat_up = self.feature_upsampling()
        shared_conv_res = self.conv_res(shared_conv)
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
    
        