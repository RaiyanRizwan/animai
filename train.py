from tensorflow import keras
import data_preprocessing

# Dependencies
Sequential = keras.models.Sequential
Conv2D = keras.layers.Conv2D
Conv2DTranspose = keras.layers.Conv2DTranspose
LSTM = keras.layers.LSTM
Reshape = keras.layers.Reshape
TimeDistributed = keras.layers.TimeDistributed
Dense = keras.layers.Dense

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

"""
80-20 Split

Train: Animation[] (list of Animation sequences)
Target: ShiftedAnimation[]
Test: Frame[]

Animation: Frame[] x 11 (from image 1-11)
ShiftedAnimation: Frame[] x 11 (from image 2-12)
Frame: Datatype (128,128,3)
"""

train_anims, target_anims, test_anims = data_preprocessing.preprocess()
pixelGenModel.fit(x=train_anims, y=target_anims, batch_size=8, epochs=5)

pixelGenModel.save('pixelGenV1.h5')
