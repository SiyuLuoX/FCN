import tensorflow as tf 
from tensorflow.keras.layers import (Conv2D,Conv2DTranspose,Input)
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16


def FCN(input_shape=(512, 512, 3), classes=5):
    #weights='imagenet'表示使用在imagenet上训练好的权重
    #include_top = False表示只使用卷积基，而不使用全连接部分
    covn_base = VGG16(weights='imagenet', 
                        input_shape=input_shape,
                        include_top=False)
    
    layer_names = [
        'block5_conv3',   # 14x14×512
        'block4_conv3',   # 28x28*512
        'block3_conv3',   # 56x56*256
        'block5_pool',    # 7x7*512
    ]
    layers = [covn_base.get_layer(name).output for name in layer_names]
    # 创建特征提取模型
    down_stack = Model(inputs=covn_base.input, outputs=layers)
    down_stack.trainable = False

    inputs = Input(shape=input_shape)
    o1, o2, o3, x = down_stack(inputs)
    x1 = Conv2DTranspose(512, 3, padding='same', 
                        strides=2, activation='relu')(x)  # 14*14*512
    x1 = Conv2D(512, 3, padding='same', activation='relu')(x1)  # 14*14*512
    c1 = tf.add(o1, x1)    # 14*14*512
    x2 = Conv2DTranspose(512, 3, padding='same', 
                        strides=2, activation='relu')(c1)  # 28*28*512
    x2 = Conv2D(512, 3, padding='same', activation='relu')(x2)  # 28*28*512
    c2 = tf.add(o2, x2)
    x3 = Conv2DTranspose(256, 3, padding='same', 
                        strides=2, activation='relu')(c2)  # 256*256*256
    x3 = Conv2D(256, 3, padding='same', activation='relu')(x3)  # 256*256*256
    c3 = tf.add(o3, x3)
    
    x4 = Conv2DTranspose(128, 3, padding='same', 
                        strides=2, activation='relu')(c3)  # 112*112*128
    x4 = Conv2D(128, 3, padding='same', activation='relu')(x4)  # 112*112*128
    
    predictions = Conv2DTranspose(classes, 3, padding='same', 
                                strides=2, activation='softmax')(x4)   # 224*224*3
    
    model = Model(inputs=inputs, outputs=predictions)
    return model
