import config

class Vocabulary:
    def __init__(self):
        """
        Init Vocabulary with the three starting tokens (PAD, EOS and SOS)
        """
        self.trimmed = False
        self.initDictionaries()

    def initDictionaries(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {config.PAD: "PAD", config.SOS: "SOS", config.EOS: "EOS"}
        self.num_words = 3  # Count default tokens

    def addSentence(self, sentence):
        """
        Read sentence and add words in Vocabulary

        :param sentence: sentence to retrieve words
        """
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        """
        Add specific word in Vocabulary

        :param word: A token
        """
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1