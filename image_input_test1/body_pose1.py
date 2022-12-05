# Import TF and TF Hub libraries.
import tensorflow as tf
import  numpy as np
import time


# Initialize the TFLite interpreter
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

start_time=time.time()
# Load the input image.
image_path = '../测试图像/1.jpg'
image = tf.io.read_file(image_path)
# image = np.array(image, dtype=np.float32)

image = tf.compat.v1.image.decode_jpeg(image)
image = tf.expand_dims(image, axis=0)
# Resize and pad the image to keep the aspect ratio and fit the expected size.
image = tf.image.resize_with_pad(image, 256, 256)
# image = image.astype(np.uint8)

# TF Lite format expects tensor type of float32.
input_image = tf.cast(image, dtype=tf.uint8)
# input_image = tf.cast(image, dtype=tf.float32)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.set_tensor(input_details[0]['index'], input_image.numpy())

interpreter.invoke()

# Output is a [1, 1, 17, 3] numpy array.
keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
print(keypoints_with_scores)
end_time=time.time()
print('use time = ',(end_time-start_time)*1000)