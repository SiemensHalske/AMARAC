from keras.optimizers import Adam
from keras.regularizers import l2
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.layers import Dropout, BatchNormalization
from keras.datasets import cifar100
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

# Define config variables
BATCH_SIZE = 64
EPOCHS = 20
VALIDATION_SPLIT = 0.2
LEARNING_RATE = 0.0005
L2_REGULARIZATION = 0.009
SAVE_MODEL_ACCURACY_THRESHOLD = 0.8
SAVE_MODEL_LOSS_THRESHOLD = 0.5

# Load and preprocess the dataset
(train_images, train_labels), (test_images, test_labels) = cifar100.load_data()
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0
train_labels = to_categorical(train_labels, 100)
test_labels = to_categorical(test_labels, 100)

# Build the model

model = Sequential([
    Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
  
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Flatten(),

    Dense(512, activation='relu', kernel_regularizer=l2(L2_REGULARIZATION)),
    Dropout(0.5),

    Dense(100, activation='softmax', kernel_regularizer=l2(L2_REGULARIZATION))
])

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3
)


# Create the optimizer object with several settings
optimizer = Adam(
    learning_rate=LEARNING_RATE,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False
)

# Compile the model with the optimizer object
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
model.fit(train_images, train_labels, epochs=EPOCHS,
          batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc:.2f}")
print(f"Test loss: {test_loss:.2f}")
print("-------------------------------------\n")
model.summary()

# Save the model if the accuracy
# is greater than the threshold and the loss
# is less than the threshold
if test_acc > SAVE_MODEL_ACCURACY_THRESHOLD and \
        test_loss < SAVE_MODEL_LOSS_THRESHOLD:
    model.save('cifar100_model.h5')
    print("\n\nModel saved successfully!")
else:
    print("\n\nMetrics not good enough! Model not saved!")
