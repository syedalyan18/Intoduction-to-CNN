import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter,uniform_filter


feature_map=np.array([
    [1,2,3,0],
    [4,5,6,1],
    [7,8,9,2],
    [0,1,2,3]
])

# max_pooled=maximum_filter(feature_map,size=2,mode='constant')
# avg_pooled=uniform_filter(feature_map,size=2,mode='constant')


# fig,axes=plt.subplots(1,3,figsize=(12,4))
# axes[0].imshow(feature_map, cmap='viridis')
# axes[0].set_title("original Feature Map")
# axes[1].imshow(max_pooled, cmap='viridis')
# axes[1].set_title("Max Pooled")
# axes[2].imshow(avg_pooled, cmap='viridis')
# axes[2].set_title("Avg Pooled")
# plt.show()


import tensorflow as tf

input_tensor=tf.constant(feature_map.reshape(1,4,4,1),dtype=tf.float32)

max_pool=tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=2,padding='valid')

max_pooled_tensor=max_pool(input_tensor)



avg_pool=tf.keras.layers.AveragePooling2D(pool_size=(2,2),strides=2,padding='valid')

avg__pooled_tensor=avg_pool(input_tensor)


print(f"Max Pooled Tensor : \n {tf.squeeze(max_pooled_tensor).numpy()}")
print(f"Avg Pooled Tensor : \n {tf.squeeze(avg__pooled_tensor).numpy()}")