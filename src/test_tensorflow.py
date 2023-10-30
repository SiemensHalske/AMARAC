import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
import tensorflow as tf
import time
# Define a convolutional neural network
from keras import regularizers

# Configuration variables
IMAGE_SIZE = (28, 28, 1)
NUM_CLASSES = 10
BATCH_SIZE = 64
NUM_EPOCHS = 5
ROTATION_RANGE = 10
ZOOM_RANGE = 0.1
WIDTH_SHIFT_RANGE = 0.1
HEIGHT_SHIFT_RANGE = 0.1


model = keras.Sequential([
    keras.layers.Conv2D(32, kernel_size=(
        3, 3), activation='relu', input_shape=IMAGE_SIZE, kernel_regularizer=regularizers.l2(0.01)),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu',
                        kernel_regularizer=regularizers.l2(0.01)),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu',
                       kernel_regularizer=regularizers.l2(0.01)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

# Optimizers

optimizer_adam = keras.optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    weight_decay=0.0,
    ema_momentum=0.0,
    amsgrad=False
)

optimizer_sgd = keras.optimizers.SGD(
    learning_rate=0.01,
    momentum=0.0,
    nesterov=False
)

optimizer_rmsprop = keras.optimizers.RMSprop(
    learning_rate=0.001,
    rho=0.9,
    momentum=0.0,
    epsilon=1e-07,
    centered=False,
    weight_decay=0.0
)


# Compile the model with categorical crossentropy loss and Adam optimizer
from tensorflow.compat.v1 import RunMetadata

model.compile(loss='categorical_crossentropy',
              optimizer=optimizer_rmsprop, metrics=['accuracy'])

# Load the MNIST dataset and preprocess the data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape((60000, *IMAGE_SIZE)).astype('float32') / 255
x_test = x_test.reshape((10000, *IMAGE_SIZE)).astype('float32') / 255
y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

# Data augmentation
datagen = ImageDataGenerator(rotation_range=ROTATION_RANGE, zoom_range=ZOOM_RANGE,
                             width_shift_range=WIDTH_SHIFT_RANGE, height_shift_range=HEIGHT_SHIFT_RANGE)
datagen.fit(x_train)

# Train the model for NUM_EPOCHS epochs
start_time = time.time()
run_metadata = RunMetadata()
history = model.fit(datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
                    steps_per_epoch=len(x_train) / BATCH_SIZE, epochs=NUM_EPOCHS,
                    validation_data=(x_test, y_test),
                    callbacks=[TensorBoard(log_dir='./logs', histogram_freq=1, 
                                           write_graph=True, write_images=True,
                                           run_metadata=run_metadata)])
end_time = time.time()

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)

# Print the training and validation accuracy and loss
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']

print('Training accuracy:', train_acc)
print('Validation accuracy:', val_acc)
print('Training loss:', train_loss)
print('Validation loss:', val_loss)

print('Time taken:', end_time - start_time)
