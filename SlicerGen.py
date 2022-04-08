import numpy
from keras.utils import np_utils


# Generator. Slice texts by small parts for neuro.
class SlicerGen:
    def __init__(self, step, batch_size, shape):
        self.batch_size = batch_size
        self.step = step
        self.shapeX = shape[0]
        self.shapeY = shape[1]
        self.slice_length = shape[0]*shape[1]

    # slice training texts for neuro train
    def slice_train(self, texts_with_author_id, authors_count):
        counter = 0
        while True:
            for pair in texts_with_author_id:
                text = pair[0]
                author_id = pair[1]
                x = []
                y = []
                for i in range(0, len(text) - self.slice_length, self.step):
                    x.append(text[i:i + self.slice_length])
                    y.append(author_id)
                    counter += 1
                    if counter >= self.batch_size:
                        Xtrain = numpy.reshape(x, (counter, self.shapeX, self.shapeY, 1))
                        Ytrain = np_utils.to_categorical(y, authors_count)
                        counter = 0
                        yield Xtrain, Ytrain
                        x = []
                        y = []

                if counter != 0:
                    Xtrain = numpy.reshape(x, (counter, self.shapeX, self.shapeY, 1))
                    Ytrain = np_utils.to_categorical(y, authors_count)
                    counter = 0
                    yield Xtrain, Ytrain
                    x = []
                    y = []

    # slice test text for neuro
    def slice_predict(self, text):
        difference = len(text) % self.slice_length

        for i in range(0, len(text) - self.slice_length, self.slice_length):
            text_slice = text[i:i + self.slice_length]
            yield numpy.reshape(text_slice, (1, self.shapeX, self.shapeY, 1))

        # For last slice that too small for neuro: extend by zeros at the end
        if difference != 0:
            text_slice = text[len(text)-difference:] + [0] * (self.slice_length-difference)
            yield numpy.reshape(text_slice, (1, self.shapeX, self.shapeY, 1))

    # get size for epoch
    def get_batch_size_per_epoch(self, texts_with_author_id):
        result = 0
        for text in texts_with_author_id:
            result += len(text[0])
        result = result / self.batch_size

        return result
