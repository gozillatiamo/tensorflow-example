import os
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_addons as tfa
import numpy as np

from tkinter import filedialog, Tk
from keras.preprocessing import image

class myCallBack(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') > 0.95):
            print("\nReached 95% accuracy so cancelling training!")
            self.model.stop_training = True

def train_model():
    # MODEL DEFINITION START
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(300, 300, 3)),
        tf.keras.layers.Conv2D(16, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])

    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
    # MODEL DEFINITION END

    # EXTRACT PHASE START
    data = tfds.load('horses_or_humans', split='train', as_supervised=True)
    val_data = tfds.load('horses_or_humans', split='test', as_supervised=True)
    # EXTRACT PHASE END

    # TRANSFORM PHASE START
    def augmentimages(image, label):
        image = tf.cast(image, tf.float32)
        image = (image/255)
        image = tf.image.random_flip_left_right(image)
        image = tfa.image.rotate(image, 40, interpolation='NEAREST')
        return image, label

    train = data.map(augmentimages)
    train_batches = train.shuffle(100).batch(32)
    validation_batches = val_data.batch(32)
    # TRANSFORM PHASE START

    # LOAD PHASE START
    callback = myCallBack()
    history = model.fit(train_batches, epochs=5, 
                        # callbacks=[callback],
                        validation_data=validation_batches, validation_steps=1)
    # LOAD PHASE END
    return model

def choose_image(model):

    root = Tk()
    root.withdraw()
    files = filedialog.askopenfilenames()
    for file in files: 
        filename = os.path.basename(file)
        img = image.load_img(file, target_size=(300, 300))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        images = np.vstack([x])
        classes = model.predict(images, batch_size=10)  
        print(classes)
        print(classes[0])

        if classes[0] > 0.5:
            print(filename + " is a human")
        else:
            print(filename + " is a horse")
    

choose_image(train_model())
