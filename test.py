import numpy as np
from tensorflow import keras

# load model
pixelGenModel = keras.models.load_model('pixelGenV1.h5')

generated_sequence = np.expand_dims(initial_frame, axis=1)  # Start with initial frame

for i in range(7):  # To generate next 7 frames
    # Use the model to predict the next frame
    next_frame = pixelGenModel.predict(generated_sequence)
    
    # Take the last frame of the generated sequence
    next_frame = next_frame[:, -1:, :, :, :]
    
    # Concatenate it with the previous frames
    generated_sequence = np.concatenate([generated_sequence, next_frame], axis=1)

# `generated_sequence` now contains the generated 8 frames
