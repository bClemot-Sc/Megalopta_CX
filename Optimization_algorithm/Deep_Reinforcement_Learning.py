## Deep Reinforcement Learning algorithm
## Author: Bastien Clémot

## ----- Import packages
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.constraints import MinMaxNorm
from kerastuner.tuners import RandomSearch
import matplotlib.pyplot as plt


## ----- Generate data

# Sinusoid shape function
def sinusoid_shape(direction, ratio):
    x = np.arange(16)
    a = -0.56
    b = 0.81
    c = -4.20
    d = 0.42
    y = a * np.sin(b * (x + c)) + d
    y = list(y * ratio)
    while max(y) != y[direction-1]:
        y.append(y.pop(0))
    y = [i if i >= 0 else 0 for i in y]
    return y

# Generate training data
input_data = []
output_data = []
for _ in range(1000):
    ratio1 = random.random()
    ratio2 = 1 - ratio1
    direction1, direction2 = random.sample(range(1, 9), 2)
    sinusoid1 = sinusoid_shape(direction1,ratio1)
    sinusoid2 = sinusoid_shape(direction2,ratio2)
    input_data.append(sinusoid1 + sinusoid2)
    best_sinusoid = sinusoid1 if max(sinusoid1) > max(sinusoid2) else sinusoid2
    output_data.append(best_sinusoid)
input_data = np.array(input_data)
output_data = np.array(output_data)

print("Data generated!")
    
    
## ----- Function to build the model
def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Input(shape=(32,)))
    for i in range(hp.Int('num_layers', 1, 5)):
        units = hp.Int(f'units_{i}', 16, 64, 16)
        layer = layers.Dense(units=units, activation='sigmoid', kernel_constraint=MinMaxNorm(min_value=-1.0, max_value=1.0))
        model.add(layer)
    model.add(layers.Dense(16, activation='sigmoid'))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    return model


## ----- Define tuner
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=100,
    executions_per_trial=1,
    directory='/tuner/',
    project_name='brain_model'
)
print("Searching for optimal set of hyperparameters...")
tuner.search(input_data, output_data, epochs=1000, validation_split=0.2)


## ----- Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]


print("Training the model with optimal set of hyperparameters...")
## ----- Build model with optimal parameters and train it
model = tuner.hypermodel.build(best_hps)
history = model.fit(input_data, output_data, epochs=1000, validation_split=0.2)
model.save("best_sinusoid_model.h5")


## ----- Plot validation accuracy and loss
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Validation Accuracy and Loss')
plt.legend()
plt.show()

print("Done!")