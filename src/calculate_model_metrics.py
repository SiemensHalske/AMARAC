def parse_kernel_size(kernel_size_str):
    if not kernel_size_str:
        return 0
    return int(kernel_size_str.split('(')[1].split(',')[0])


def parse_params(params_str):
    params_dict = {}
    for param in params_str.split(','):
        if '=' in param:
            key, value = param.split('=')
            params_dict[key.strip()] = value.strip()
    return params_dict


def calculate_metrics_from_model(model_text):
    total_parameters = 0
    total_neurons = 0
    prev_filters = None

    for line in model_text.strip().split('\n'):
        line = line.strip()
        if 'Conv2D' in line or 'Dense' in line:
            params_str = line.split('(')[1].split(')')[0]
            params = parse_params(params_str)
            if 'Conv2D' in line:
                filters = int(params.get('filters', 0))
                kernel_size_str = params.get('kernel_size', '0')
                kernel_size = parse_kernel_size(kernel_size_str) if '(' in kernel_size_str else int(kernel_size_str or 0)
                if prev_filters is None:
                    input_shape = 3  # Assuming a standard RGB image
                    params_count = (kernel_size * kernel_size * input_shape + 1) * filters
                else:
                    params_count = (kernel_size * kernel_size * prev_filters + 1) * filters
                neurons = filters
                prev_filters = filters
            elif 'Dense' in line:
                units = int(params.get('units', 0))
                params_count = (prev_filters + 1) * units
                neurons = units
            else:
                params_count = 0
                neurons = 0
        else:
            params_count = 0
            neurons = prev_filters if prev_filters is not None else 0

        total_parameters += params_count
        total_neurons += neurons

    return total_parameters, total_neurons


# Your model text goes here
model_text = """
Input(shape=(32, 32, 3))
Conv2D(32, kernel_size=(3, 3), activation="relu")
BatchNormalization()
Conv2D(64, kernel_size=(3, 3), activation="relu")
BatchNormalization()
MaxPooling2D(pool_size=(2, 2))
Conv2D(128, kernel_size=(3, 3), activation="relu")
BatchNormalization()
Conv2D(256, kernel_size=(3, 3), activation="relu")
BatchNormalization()
MaxPooling2D(pool_size=(2, 2))
Flatten()
Dropout(0.5)
Dense(128, activation="relu")
BatchNormalization()
Dense(64, activation="relu")
Dropout(0.5)
Dense(10, activation="softmax")
"""

total_parameters, total_neurons = calculate_metrics_from_model(model_text)
print(f"Total Parameters: {total_parameters}")
print(f"Total Neurons: {total_neurons}")
