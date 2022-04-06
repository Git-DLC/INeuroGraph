from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout


def create(image_shape, exit_counts):
    shape = (image_shape[0], image_shape[1], 1)
    model = Sequential([
        Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=shape),
        MaxPooling2D((2, 2), strides=2),
        Dropout(0.2),
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        MaxPooling2D((2, 2), strides=2),
        Dropout(0.2),
        Flatten(),
        Dense(256, activation='relu'),
        Dense(exit_counts, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

