

import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
import scipy.misc
#import model
# model=tf.keras.models.load_model('model.ckpt')
import cv2
from subprocess import call
import numpy as np


smoothed_angle = 0
#sess = tf.compat.v1.InteractiveSession()
#saver = tf.compat.v1.train.Saver()
#saver.restore(sess, "model.ckpt")

img = cv2.imread('C:\Z\minor2\steering_wheel_image.jpg',0)
rows,cols = img.shape

smoothed_angle = 0
degrees=20
model = tf.keras.models.load_model('C:\Z\minor2\model')

# model = tf.model.create_model()
cap = cv2.VideoCapture('video5.avi')
while(cv2.waitKey(10) != ord('q')):
    ret, frame = cap.read()
    #image=tf.image.crop_to_bounding_box(frame, 106, 0, 150, 455)
    image = tf.image.resize(frame, [66, 200]) / 255.0
    #image=tf.image.resize(frame,[66,200])
    #image=image/255
    #image=np.array(image)
    image = np.array(image)
    image= image.reshape(1,66,200,3)

    print(type(image))
    print(np.shape(image))
    import time
    #time.sleep(10)

    degrees = model.predict(image)
    degrees = int(degrees)
    #degree=model.y.eval(feed_dict={model.x: [image], model.keep_prob: 1.0})[0][0] * 180 / scipy.pi
    #call("clear")
    print("Predicted steering angle: " + str(degrees) + " degrees")
    cv2.imshow('frame', frame)
    #make smooth angle transitions by turning the steering wheel based on the difference of the current angle
    #and the predicted angle
    smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)
    M = cv2.getRotationMatrix2D((cols/2,rows/2),-smoothed_angle,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    cv2.imshow("steering wheel", dst)

cap.release()
cv2.destroyAllWindows()