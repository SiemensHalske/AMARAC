from keras.regularizers import l1_l2
import tensorflow as tf
import keras
from keras import layers
import os
import custom_optimizers as optim

# Create the data directory if it doesn't exist
if not os.path.exists('../data/'):
    os.makedirs('../data/')

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()

# Normalize pixel values to be between 0 and 1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Save the data to the data directory


def save_images(x, y, folder):
    file_path = f'../data/{folder}/{y}_{y}.png'
    tf.io.write_file(file_path, tf.image.encode_png(tf.cast(x, tf.uint8)))


def tf_save_images(x, y, folder):
    py_func = tf.py_function(
        func=save_images,
        inp=[x, y, folder],
        Tout=[]
    )
    # Make sure the py_function op is triggered
    with tf.control_dependencies([py_func]):
        y = tf.identity(y)
    return y


train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).map(
    lambda x, y: (tf.image.encode_png(tf.cast(x, tf.uint8)), y)
).map(
    lambda x, y: (tf_save_images(x, y, 'train'), y)
).batch(64)

# Define the model architecture

L1 = 0.0025
L2 = 0.005

model = keras.Sequential(
    [
        keras.Input(shape=(32, 32, 3)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
        layers.BatchNormalization(),
        # layers.Conv2D(256, kernel_size=(3, 3), activation="relu"),
        # layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.2),
        layers.Dense(128, activation="relu", kernel_regularizer=l1_l2(l1=L1, l2=L2)),
        layers.BatchNormalization(),
        layers.Dense(64, activation="relu", kernel_regularizer=l1_l2(l1=L1, l2=L2)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(100, activation="softmax"),
    ]
)

_ = optim.optimizer_adam(learning_rate=0.0005)
c_optimizer = _.optimizer

# Compile the model
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=c_optimizer, metrics=["accuracy"])

# Train the model
model.fit(
    x_train,
    y_train,
    batch_size=32,
    epochs=10,
    validation_split=0.1
)

# Evaluate the model on the test set
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
