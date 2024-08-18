import tensorflow as tf
from keras.initializers import glorot_uniform
#Reading the model from JSON file
with open('model-bw.json', 'r') as json_file:
    json_savedModel= json_file.read()
#load the model architecture 
model_j = tf.keras.models.model_from_json(json_savedModel)
model_j.summary()

model_j.load_weights('model-bw.h5')
model_j.compile(loss='sparse_categorical_crossentropy',
         optimizer='SGD',
         metrics=['accuracy'])
