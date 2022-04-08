from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from string import punctuation


# Text to Graph algorithm by count
class CountConvertor:
    def __init__(self, maxCounter):
        self.maxCounter = maxCounter
        self.wordnet_lemmatizer = WordNetLemmatizer()
        self.wordNumDict = {}

    # convert texts + init wordNumDict (for starting texts)
    def start(self, texts):
        wordCountDict = {}
        # texts --> lemmanized_texts + wordCountDict
        lemmanized_texts = []
        for text in texts:
            text = text.translate(str.maketrans("", "", punctuation))
            tokenized_text = word_tokenize(text)
            lemmanized_text = []

            for word in tokenized_text:
                lemma = self.wordnet_lemmatizer.lemmatize(word)
                lemmanized_text.append(lemma)
                if lemma not in wordCountDict:
                    wordCountDict[lemma] = 1
                else:
                    wordCountDict[lemma] = wordCountDict[lemma] + 1
            lemmanized_texts.append(lemmanized_text)

        # Sorting wordCountDict for cut
        wordCountDict = dict(
            sorted(
                wordCountDict.items(), key=lambda x: x[1], reverse=True
            )
        )

        # cutting top words in wordCountDict
        wordToNum = {}
        cur = 1
        for key in wordCountDict.keys():
            if cur >= self.maxCounter + 1:
                break
            wordToNum[key] = cur
            cur += 1
        self.maxCounter = cur
        # lemmanized_texts --> graphTexts
        result = []
        for text in lemmanized_texts:
            graphText = []
            for word in text:
                if word not in wordToNum:
                    graphText.append(0)
                else:
                    graphText.append(wordToNum[word] / self.maxCounter)
            result.append(graphText)

        self.wordNumDict = wordToNum
        return result

    # convert text using old wordNumDict (for new text)
    def convert(self, text):
        # text --> lemmanized_texts
        text = text.translate(str.maketrans("", "", punctuation))
        tokenized_text = word_tokenize(text)

        graphText = []
        for word in tokenized_text:
            lemma = self.wordnet_lemmatizer.lemmatize(word)
            if lemma not in self.wordNumDict:
                graphText.append(0)
            else:
                graphText.append(self.wordNumDict[word] / self.maxCounter)

        return graphText
