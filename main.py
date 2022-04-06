import os
import pickle

import textToGraphs
import CountConvertor
import SlicerGen
import neuroGenTest
import neuroTrain
import useModel


# get texts for testing
def getTestTexts():
    myTextNames = []
    myTexts = []
    myAuthors = []

    with open(textsPath + "/" + "WarAndPeace.txt", 'r', encoding="utf-8") as file:
        myTexts.append(file.read())
        myAuthors.append("Толстой")
        myTextNames.append("WarAndPeace")

    with open(textsPath + "/" + "CrimeAndPunishment.txt", 'r', encoding="utf-8") as file:
        myTexts.append(file.read())
        myAuthors.append("Достоевский")
        myTextNames.append("CrimeAndPunishment")

    with open(textsPath + "/" + "EugeneOnegin.txt", 'r', encoding="utf-8") as file:
        myTexts.append(file.read())
        myAuthors.append("Пушкин")
        myTextNames.append("EugeneOnegin")

    return myTextNames, myTexts, myAuthors


def only_predict():
    useConvertor = CountConvertor.CountConvertor(MaxCounter)
    with open(Path + "/dicts/wordNum.pickle", 'rb') as file:
        temp = pickle.load(file)
        useConvertor.wordNumDict = temp

    with open(Path + "/dicts/idToAuthor.pickle", 'rb') as file:
        temp = pickle.load(file)
        idToAuthorDict = temp
    useSlicer = SlicerGen.SlicerGen(Step, Batch_size, Shape)

    modelPredictor = useModel.ModelUse(Path + "/model", useConvertor, useSlicer)
    modelPredictor.change_weights(Path)

    print("Start Input")
    while True:
        myinput = input()
        predictions = modelPredictor.predict(myinput)
        j = 1
        for prediction in predictions:
            print("Slice #", j)
            i = 0
            for authorGuess in prediction:
                print(idToAuthorDict[i], " - ", round(authorGuess*10000)/100)
                i += 1
            i = 0
            j += 1
            print()


skipToUse = True
MaxCounter = 500
Step = 50
Batch_size = 100
Shape = (50, 50)
Epochs = 5

Path = "Data/Users/test"
textsPath = "Data/Texts"

if skipToUse:
    only_predict()

myTextNames, myTexts, myAuthors = getTestTexts()
print("Text read")

Convertor = CountConvertor.CountConvertor(MaxCounter)
print("Got Convertor")

# will create files
textToGraphs.convert_texts(myTextNames, myTexts, myAuthors, Convertor, Path)
print("Text converted")

graphsAuthors = []
for fileName in os.listdir(Path + "/graphs"):
    if fileName.endswith(".tuple-pickle"):
        with open(Path + "/graphs/" + fileName, 'rb') as file:
            temp = pickle.load(file)
            graphsAuthors.append(temp)

with open(Path + "/dicts/idToAuthor.pickle", 'rb') as file:
    idToAuthor = pickle.load(file)

slicer = SlicerGen.SlicerGen(Step, Batch_size, Shape)
print("Slicer ready")

iNeuro = neuroGenTest.create(Shape, len(idToAuthor))
print("Neuro created")

iNeuro = neuroTrain.train(
    iNeuro,
    slicer.slice_train(graphsAuthors, len(idToAuthor)),
    slicer.get_batch_size_per_epoch(graphsAuthors),
    Path,
    Epochs
)
print("Neuro Trained")

modelPredictor = useModel.ModelUse(Path + "/model", Convertor, slicer)
modelPredictor.change_weights(Path)

print("Start Input")
while True:
    myinput = input()
    predictions = modelPredictor.predict(myinput)
    j = 1
    for prediction in predictions:
        print("Slice #", j)
        i = 0
        for authorGuess in prediction:
            print(idToAuthor[i], " - ", round(authorGuess*10000)/100)
            i += 1
        i = 0
        j += 1
        print()
