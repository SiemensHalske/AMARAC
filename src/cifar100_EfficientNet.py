import tensorflow as tf
from keras.datasets import cifar100
from keras import layers
from keras.callbacks import ReduceLROnPlateau
from keras import regularizers


def execute_step(step_description, func):
    print("========================================")
    print(f"{step_description}...")
    func()
    print("Done.")
    print("========================================")


def load_data():
    global train_images, train_labels, test_images, test_labels
    (train_images, train_labels), (test_images,
                                   test_labels) = cifar100.load_data(
                                       label_mode='fine')


def preprocess_data():
    global train_images, train_labels, test_images, test_labels
    train_images = tf.keras.applications.efficientnet.preprocess_input(
        train_images)
    test_images = tf.keras.applications.efficientnet.preprocess_input(
        test_images)
    train_labels = tf.keras.utils.to_categorical(train_labels, 100)
    test_labels = tf.keras.utils.to_categorical(test_labels, 100)


def build_model():
    global model
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False, weights='imagenet', input_shape=(32, 32, 3))
    base_model.trainable = True
    model = tf.keras.Sequential([
        base_model,
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(100, activation='softmax',
                     kernel_regularizer=regularizers.L1L2(l1=0.01, l2=0.001))
    ])


def compile_model():

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'],
                  run_eagerly=True)


execute_step("Loading CIFAR-100 dataset", load_data)
execute_step("Preprocessing data", preprocess_data)
execute_step("Building EfficientNetB0 model", build_model)
execute_step("Compiling model", compile_model)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=1e-6
)

# Train the model (commented out for now)
execute_step("Training model", lambda: model.fit(
    train_images,
    train_labels,
    epochs=25,
    batch_size=64,
    validation_split=0.2,
    callbacks=[reduce_lr]
))
