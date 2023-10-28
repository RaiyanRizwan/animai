from tensorflow import keras
import data_preprocessing
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import image

# Dependencies
Sequential = keras.models.Sequential
Conv2D = keras.layers.Conv2D
Conv2DTranspose = keras.layers.Conv2DTranspose
LSTM = keras.layers.LSTM
Reshape = keras.layers.Reshape
TimeDistributed = keras.layers.TimeDistributed
Dense = keras.layers.Dense

should_load = True

train_anims, target_anims, test_anims = data_preprocessing.preprocess()

if should_load:
    pixelGenModel = keras.models.load_model('pixelGenV1.h5')
else:
    """
    pixelGenModel 
    input: sequences of 12 frames, 32 x 32, 3 color channels (RGB)
    input --> conv2D (extract features) --> reshape --> LSTM --> reshape --> conv2DTranspose --> output (12 frames)

    The convolutional layer is used to extract the features of each frame (the TimeDistributed wrapper enables the model to work on each frame 
    independently), which are then fed into the LSTM (memory layer), and back into the transposed convolutional layer to regenerate the images. 
    The high level idea is that the LSTM predicts the features of the next frame, for each frame passed in, based on dependencies between previous
    frames. The loss function computes the difference between the predicted and actual frames, which is why it's crucial that the target dataset 
    is passed in as the input dataset shifted to the left by one (since the target for frame 1 is frame 2).
    """

    pixelGenModel = keras.models.Sequential()
    pixelGenModel.add(TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'), input_shape=(15, 128, 128, 3)))

    pixelGenModel.add(Reshape((15, -1))) 
    pixelGenModel.add(LSTM(50, return_sequences=True)) 
    pixelGenModel.add(Dense(128 * 128 * 3)) 
    pixelGenModel.add(Reshape((15, 128, 128, 3)))

    pixelGenModel.add(TimeDistributed(Conv2DTranspose(3, (3, 3), activation='relu', padding='same')))

    pixelGenModel.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    pixelGenModel.summary()
    
    pixelGenModel.fit(x=train_anims, y=target_anims, batch_size=8, epochs=10)
    pixelGenModel.save('pixelGenV1.h5')

def display_frame(frame):
    plt.imshow(frame.astype('uint8'))
    plt.axis('off')
    plt.show()

# --- TEST --- 

test_anim = test_anims[0]
display_frame(test_anim[0])

initial_frame = np.expand_dims(test_anim[0], axis=0)
initial_sequence = np.tile(initial_frame, (15, 1, 1, 1))
initial_sequence = np.expand_dims(initial_sequence, axis=0)

# Generate the next 14 frames
for i in range(14):
    next_frame = pixelGenModel.predict(initial_sequence)
    
    for f in range(15):
        display_frame(next_frame[0][f])
    
    # Take the last frame from the predicted sequence
    next_frame = next_frame[0][-1]
    
    # Roll the frames in the initial_sequence to the left by one frame
    initial_sequence = np.roll(initial_sequence, shift=-1, axis=1)
    
    # Insert the new frame at the end of the initial_sequence
    initial_sequence[:, -1] = next_frame


