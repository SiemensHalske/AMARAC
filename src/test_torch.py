from keras.regularizers import l1_l2
import tensorflow as tf
import keras
from keras import layers
import os
import custom_optimizers as optim
from keras.callbacks import ReduceLROnPlateau
from keras.applications.mobilenet_v2 import MobileNetV2

L1 = 0.0005
L2 = 0.0005

BATCH_SIZE = 64
EPOCHS = 10

# Define the learning rate scheduler


reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=1,
    min_lr=0.0001,
    verbose=1,
    mode='auto',
    cooldown=0,
    min_delta=0.0001
)


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
).batch(BATCH_SIZE)

# Define the model architectures

# Model architecture
input_shape = (32, 32, 3)
resize_shape = (96, 96)
inputs = keras.Input(shape=input_shape)

base_model = MobileNetV2(input_shape=resize_shape +
                         (3,), include_top=False, weights='imagenet')
x = layers.experimental.preprocessing.Resizing(*resize_shape)(inputs)
x = base_model(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(1024, activation="relu",
                 kernel_regularizer=l1_l2(l1=L1, l2=L2))(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)
predictions = layers.Dense(100, activation="softmax")(x)
model_mobilenet = keras.Model(inputs=inputs, outputs=predictions)


# Model architecture
model_cnn = keras.Sequential([
    keras.Input(shape=(32, 32, 3)),

    # Block 1
    layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),

    # Block 2
    layers.Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),

    # Block 3
    layers.Conv2D(256, kernel_size=(3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.Conv2D(256, kernel_size=(3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),

    # Fully Connected Layers
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(1024, activation="relu",
                 kernel_regularizer=l1_l2(l1=L1, l2=L2)),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(100, activation="softmax")
])

_ = optim.optimizer_rmsprop(
    learning_rate=0.001,
    momentum=0.9,
    decay=0.0,
    rho=0.9,
    epsilon=1e-07,
    centered=False,
    name='RMSprop'
)
c_optimizer = _.optimizer

# Compile the model
model_mobilenet.compile(loss="sparse_categorical_crossentropy",
                        optimizer=c_optimizer, metrics=["accuracy"])

# Define the callbacks
print_callback = keras.callbacks.Callback()


def on_epoch_end(epoch, logs=None):
    print(
        f"\nEpoch {epoch+1}: Learning rate = {model_mobilenet.optimizer.lr.numpy()}")
    for metric, value in logs.items():
        print(f"{metric}: {value:.4f}\n*********************************")
    print("\n")


print_callback.on_epoch_end = on_epoch_end

# Train the model
model_mobilenet.fit(
    x_train,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.1,
    callbacks=[print_callback, reduce_lr],
)

# Evaluate the model on the test set
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
