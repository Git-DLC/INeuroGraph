from keras.callbacks import ModelCheckpoint


def train(model, generator, steps_per_epoch, dir_path, epochs=5):

    # Now it's not only saving best
    filepath = dir_path + "/weights/weights-improvement-{epoch:02d}-{loss:.4f}-.hdf5"

    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=1,
        save_best_only=True,
        mode='min')
    desired_callbacks = [checkpoint]

    model.fit(generator, steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=desired_callbacks)
    model.save(dir_path + '/' + "model")

    return model
