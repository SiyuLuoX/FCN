import time
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from nets.FCN import FCN

'''自定义灰度图调色板'''
from matplotlib.colors import LinearSegmentedColormap
# clist=[(1,0,0),(0,1,0),(0,0,1),(1,0.5,0),(1,0,0.5),(0.5,1,0)]
clist=['#F1C40F','#FF3800','#FF3800','#FF3800','#FF3800','#FF3800']
newcmp = LinearSegmentedColormap.from_list('chaos',clist)


model =FCN(input_shape=(640,640,3), classes=5)
model.load_weights(r"weights/ep24-val_loss0.32.h5")

image = tf.io.read_file(r'dataset/JPEGImages/IMG0001_0_0.jpg')
image = tf.image.decode_jpeg(image, channels=3)

def predict(img):
    feature = tf.cast(img, tf.float32) / 127.5 - 1 
    # feature = tf.divide(tf.subtract(feature, rgb_mean), rgb_std)
    x = tf.expand_dims(feature, axis=0)
    return model.predict(x)

T1 = time.perf_counter()
pred_mask = predict(image) #(4,320,480,5)
T2 =time.perf_counter()
print('推理时间{0}'.format((T2-T1)))
pred_mask = tf.argmax(pred_mask, axis=-1) #(4,320,480)
pred_mask = np.array(pred_mask)  
pred_mask = pred_mask.transpose(1,2,0)
plt.imshow(pred_mask,cmap=newcmp)
# plt.imshow(pred_mask)
plt.show()

