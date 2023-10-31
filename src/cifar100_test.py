import sys
import tensorflow as tf
import numpy as np
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QTextEdit, QScrollBar
from PyQt5.QtWidgets import QVBoxLayout, QWidget
from PyQt5.QtGui import QFont, QTextCursor
from PyQt5.QtCore import Qt

# Load and normalize CIFAR-100 data
(train_images, train_labels), (test_images,
                               test_labels) = tf.keras.datasets.cifar100.load_data(label_mode='fine')
train_images, test_images = train_images / 255.0, test_images / 255.0

# Define and compile the model (same as before)
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu',
                           input_shape=(32, 32, 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(100, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# Custom print callback
class CustomPrintCallback(tf.keras.callbacks.Callback):
    class CustomPrintCallback(tf.keras.callbacks.Callback):
        def __init__(self, text_edit):
            super().__init__()
            self.text_edit = text_edit

        def on_epoch_end(self, epoch, logs=None):
            output_text = []
            output_text.append(
                f"Epoch {epoch+1}: Loss: {logs['loss']}, Accuracy: {logs['accuracy']}, Val Loss: {logs['val_loss']}, Val Accuracy: {logs['val_accuracy']}\n")

            # Inference on a few random test samples
            random_indices = np.random.choice(len(test_images), 5)
            random_test_images = test_images[random_indices]
            random_test_labels = test_labels[random_indices]

            predictions = model.predict(random_test_images)
            predicted_labels = np.argmax(predictions, axis=1)

            output_text.append(
                f"Sample Inference - True labels: {random_test_labels.flatten()}\n")
            output_text.append(
                f"Sample Inference - Predicted labels: {predicted_labels}\n")

            cursor = self.text_edit.textCursor()
            cursor.movePosition(QTextCursor.End)
            cursor.insertText(''.join(output_text))
            self.text_edit.setTextCursor(cursor)
            self.text_edit.ensureCursorVisible()

    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()

            # Create the text edit widget
            self.text_edit = QTextEdit(self)
            self.text_edit.setReadOnly(True)
            self.text_edit.setFont(QFont('Consolas', 10))
            self.text_edit.setLineWrapMode(QTextEdit.NoWrap)

            # Create the button widget
            self.button = QPushButton('Run Program', self)
            self.button.setFont(QFont('Consolas', 10))
            self.button.clicked.connect(self.run_program)

            # Set the central widget
            central_widget = QWidget(self)
            self.setCentralWidget(central_widget)

            # Create the layout
            layout = QVBoxLayout(central_widget)
            layout.addWidget(QLabel('Output:', self))
            layout.addWidget(self.text_edit)
            layout.addWidget(self.button)

        def run_program(self):
            # Train the model
            history = model.fit(
                train_images, train_labels,
                epochs=10,
                batch_size=32,
                validation_split=0.2,
                callbacks=[CustomPrintCallback(
                    self.text_edit), reduce_lr, checkpoint_callback]
            )

            # Final Evaluation
            output_text = []
            output_text.append("\n--- Final Model Evaluation ---\n")

            # Print the model summary
            model.summary()

            # Load the best weights (assuming the last epoch is the best; you can choose otherwise)
            model.load_weights("training_checkpoints/cp-0010.ckpt")

            # Evaluate on the test set
            test_loss, test_acc = model.evaluate(
                test_images, test_labels, verbose=0)
            output_text.append(f"Test Loss: {test_loss}\n")
            output_text.append(f"Test Accuracy: {test_acc * 100}%\n")

            # Inference on a few random test samples
            random_indices = np.random.choice(len(test_images), 5)
            random_test_images = test_images[random_indices]
            random_test_labels = test_labels[random_indices]

            predictions = model.predict(random_test_images)
            predicted_labels = np.argmax(predictions, axis=1)

            output_text.append(
                f"Sample Inference - True labels: {random_test_labels.flatten()}\n")
            output_text.append(
                f"Sample Inference - Predicted labels: {predicted_labels}\n")

            cursor = self.text_edit.textCursor()
            cursor.movePosition(QTextCursor.End)
            cursor.insertText(''.join(output_text))
            self.text_edit.setTextCursor(cursor)
            self.text_edit.ensureCursorVisible()


# Reduce Learning Rate on Plateau
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=2,
    min_lr=0.001
)

# Checkpoint settings
checkpoint_path = "training_checkpoints/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create checkpoint callback
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    verbose=1
)

# Make sure checkpoint directory exists
os.makedirs(checkpoint_dir, exist_ok=True)

# Function to run the program


def run_program(gui_obj: QMainWindow):
    # Train the model
    history = model.fit(
        train_images, train_labels,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        callbacks=[CustomPrintCallback(), reduce_lr, checkpoint_callback]
    )

    # Final Evaluation

    # Get the absolute path of the checkpoint file
    # Evaluate on the test set
    test_loss, test_acc = 0, 0
    if os.path.exists("training_checkpoints/cp-0010.ckpt.index"):
        test_loss, test_acc = model.evaluate(
            test_images, test_labels, verbose=0)
        output_text.append(f"Test Accuracy: {test_acc * 100}%\n")

        checkpoint_path = os.path.abspath("training_checkpoints/cp-0010.ckpt")

        # Load the weights from the checkpoint file
        model.load_weights(checkpoint_path)

    # Inference on a few random test samples
    random_indices = np.random.choice(len(test_images), 5)
    random_test_images = test_images[random_indices]
    random_test_labels = test_labels[random_indices]

    predictions = model.predict(random_test_images)
    predicted_labels = np.argmax(predictions, axis=1)

    output_text.append(
        f"Final Inference - True labels: {random_test_labels.flatten()}\n")
    output_text.append(
        f"Final Inference - Predicted labels: {predicted_labels}\n")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set the window title
        self.setWindowTitle("CIFAR-100 Classifier")

        # Add a label to the main window
        title_label = QLabel("CIFAR-100 Classifier", self)
        font = QFont()
        font.setBold(True)
        title_label.setFont(font)
        title_label.setGeometry(10, 10, 200, 30)

        # Add a button to run the program
        run_button = QPushButton("Run", self)
        run_button.setGeometry(10, 50, 100, 30)
        run_button.clicked.connect(self.run_program)

        # Add a text box to display the output
        global output_text
        output_text = QTextEdit(self)
        output_text.setGeometry(10, 90, 480, 300)

        # Add a scrollbar to the text box
        scrollbar = QScrollBar(self)
        scrollbar.setGeometry(490, 90, 20, 300)
        output_text.setVerticalScrollBar(scrollbar)

        # Set the size of the window
        self.setGeometry(100, 100, 520, 400)

    # Function to run the program
    def run_program(self):
        output_text = []
        output_text.append("\n--- Final Model Evaluation ---\n")

        # Print the model summary
        model.summary()

        # Load the best weights (assuming the last epoch is the best; you can choose otherwise)
        model.load_weights("training_checkpoints/cp-0010.ckpt")

        # Evaluate on the test set
        test_loss, test_acc = model.evaluate(
            test_images, test_labels, verbose=0)
        output_text.append(f"Test Loss: {test_loss}\n")
        output_text.append(f"Test Accuracy: {test_acc * 100}%\n")

        # Inference on a few random test samples
        random_indices = np.random.choice(len(test_images), 5)
        random_test_images = test_images[random_indices]
        random_test_labels = test_labels[random_indices]

        predictions = model.predict(random_test_images)
        predicted_labels = np.argmax(predictions, axis=1)

        output_text.append(
            f"Final Inference - True labels: {random_test_labels.flatten()}\n")
        output_text.append(
            f"Final Inference - Predicted labels: {predicted_labels}\n")

        cursor = self.text_edit.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(''.join(output_text))
        self.text_edit.setTextCursor(cursor)
        self.text_edit.ensureCursorVisible()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
