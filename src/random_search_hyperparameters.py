from kerastuner.tuners import RandomSearch
import tensorflow as tf
from keras import datasets, layers, models


def build_model(hp):
    model = models.Sequential()

    # First Conv2D Layer
    hp_units_1 = hp.Int('units_1', min_value=32, max_value=256, step=32)
    model.add(layers.Conv2D(hp_units_1, (3, 3), activation='relu',
              padding='same', input_shape=(32, 32, 3)))
    model.add(layers.BatchNormalization())

    # Second Conv2D Layer
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # Third and Fourth Conv2D Layer
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # Fifth and Sixth Conv2D Layer
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # Flatten and Dense Layers
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu',
              kernel_regularizer=tf.keras.regularizers.l2(0.009)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(100, activation='softmax',
              kernel_regularizer=tf.keras.regularizers.l2(0.009)))

    # Tunable learning rate
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


# Load and preprocess the dataset
(train_images, train_labels), (test_images,
                               test_labels) = datasets.cifar100.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0
train_labels = tf.keras.utils.to_categorical(train_labels, 100)
test_labels = tf.keras.utils.to_categorical(test_labels, 100)

tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=3,
    directory='random_search_cifar100',
    project_name='cifar100'
)

tuner.search_space_summary()

# Search for the best parameters
tuner.search(train_images, train_labels, epochs=10, validation_split=0.2)

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of units in the first Conv2D layer is {best_hps.get('units_1')}
and the optimal learning rate for the optimizer is {best_hps.get('learning_rate')}.
""")

# Build the model with the optimal hyperparameters and train it on the data
best_model = tuner.hypermodel.build(best_hps)
best_model.fit(train_images, train_labels, epochs=20, validation_split=0.2, batch_size=32)
