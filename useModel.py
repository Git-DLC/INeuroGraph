import keras
import os


# class for model use.
class ModelUse:
    def __init__(self, pathToModel, Convertor, SlicerGen):
        self.Convertor = Convertor
        self.SlicerGen = SlicerGen
        self.model = keras.models.load_model(pathToModel)

    def change_weights(self, path):
        weights = []
        for fileName in os.listdir(path + "/weights"):
            if fileName.endswith(".hdf5"):
                weights.append(fileName)

        # looking for the min loss
        minimum = float("inf")
        current = weights[0]
        for weight in weights:
            loss = float(weight.rsplit("-", 2)[1])
            if loss < minimum:
                minimum = loss
                current = weight

        self.model.load_weights(path + "/weights/" + current)

    def predict(self, text):
        convertedText = self.Convertor.convert(text)

        predictions = []
        slicerGen = self.SlicerGen
        text_slicer = slicerGen.slice_predict(convertedText)

        # only 1 slice per prediction (could use batch_size)
        for textSlice in text_slicer:
            prediction = self.model.predict(textSlice)
            predictions.append(prediction[0])

        return predictions