import os
from tkinter import Tk, filedialog
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_addons as tfa
import numpy as np
import multiprocessing

from keras.preprocessing import image

def train_model():
    # MODEL DEFINITION START
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3,3), activation='relu',
                            input_shape=(300, 300, 3)),
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
    tfds.load('horses_or_humans', split='train', with_info=True)
    file_pattern = f'/home/gozillatiamo/tensorflow_datasets/horses_or_humans/3.0.0/horses_or_humans-train.tfrecord*'
    files = tf.data.Dataset.list_files(file_pattern)
    cores = multiprocessing.cpu_count()
    print(cores)
    train_dataset = files.interleave(
        tf.data.TFRecordDataset,
        cycle_length=cores,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )


    val_data = tfds.load('horses_or_humans', split='test', as_supervised=True)
    # EXTRACT PHASE END

    # TRANSFORM PHASE START
    def read_tfrecord(serialized_example):
        feature_description={
            "image": tf.io.FixedLenFeature((), tf.string, ""),
            "label": tf.io.FixedLenFeature((), tf.int64, -1),
        }
        example = tf.io.parse_single_example(
            serialized_example, feature_description
        )
        image= tf.io.decode_jpeg(example['image'], channels=3)
        image = tf.cast(image, tf.float32)
        image = image/255
        image = tf.image.resize(image, (300, 300))
        # image = tf.image.random_flip_left_right(image)
        # image = tf.image.random_flip_up_down(image)
        # image = tfa.image.rotate(image, 40, interpolation='NEAREST')
        return image, example['label']

    train_dataset = train_dataset.map(read_tfrecord, num_parallel_calls=cores)
    train_dataset = train_dataset.cache()
    # TRANSFORM PHASE START

    # LOAD PHASE START
    train_batches = train_dataset.shuffle(1024).batch(32)
    train_dataset = train_batches.prefetch(tf.data.experimental.AUTOTUNE) 

    validation_batches = val_data.batch(32)
    history = model.fit(train_dataset, epochs=5, verbose=1,
                        validation_data=validation_batches, validation_steps=1)
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
