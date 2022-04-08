import pickle
import os
import shutil


# will create graph files + dict files
def convert_texts(namesArray, textsArray, authorsArray, classConventor, path):
    # don't contain path tests
    def do_tests():
        if not isinstance(textsArray, list) or not isinstance(authorsArray, list):
            raise Exception("Inputs are not arrays!")

        if len(textsArray) == 0 or len(authorsArray) == 0:
            raise Exception("Arrays are empty!")

        if len(textsArray) != len(authorsArray):
            raise Exception("Length of both arrays are not equal!")

        for text in textsArray:
            if not isinstance(text, str):
                raise Exception("Input texts are not all string")
            if len(text) == 0:
                raise Exception("Input texts arrays contain blank text")

        for author in authorsArray:
            if not isinstance(author, str):
                raise Exception("Input authors are not all string")
            if len(author) == 0:
                raise Exception("Input author arrays contain blank author")

    do_tests()

    convertedTexts = classConventor.start(textsArray)

    authorToId = {}
    idToAuthor = {}

    i = 0
    for author in authorsArray:
        if author not in authorToId:
            authorToId[author] = i
            idToAuthor[i] = author
            i += 1

    # If directory already exists: remove and make new one
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path)
    i = 0

    os.makedirs(path+"/graphs")
    for text in convertedTexts:
        with open(path + "/graphs/" + authorsArray[i] + "-" + namesArray[i] + ".tuple-pickle", 'wb') as file:
            pickle.dump((text, authorToId[authorsArray[i]]), file)
        i += 1

    os.makedirs(path + "/dicts")
    with open(path + "/dicts/authorToId.pickle", 'wb') as file:
        pickle.dump(authorToId, file)

    with open(path + "/dicts/idToAuthor.pickle", 'wb') as file:
        pickle.dump(idToAuthor, file)
