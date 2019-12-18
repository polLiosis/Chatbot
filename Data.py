import unicodedata, re, vocabulary, config, glob, json, os


def createTrainSet(questions, replies):
    """
    Create train set. Each line in train set is a QA pair

    :param questions: The input sentence of the user
    :param replies: The reply of the system/bot
    """
    extracted_dataset = open(config.train_dataset, 'a')
    for question, reply in zip(questions, replies):
        extracted_dataset.write(question + "\t" + reply + "\n")


def filterDialogs(dialogs):
    """
    Remove dialogs that contain more than eleven (11) turns

    :param dialogs: dialogs between bot and user
    :return: filtered dialogs
    """
    print("Number of extracted dialogs: ", len(dialogs))
    print("Filtering dialogs...")
    filtered_turns = []
    for dialog in dialogs:
        if len(dialog) == 11:
            filtered_turns.append(dialog)
    print("Final number of dialogs: ", len(filtered_turns))
    return filtered_turns


def generateQAs(turns):
    """
    Generate QA pairs

    :param turns: the dialogs between the bot and the user
    :return: Two lists(Questions and answers/replies)
    """
    questions, replies = [], []
    for turn in turns:
        questions.append(turn[1])
        questions.append(turn[3])
        questions.append(turn[5])
        questions.append(turn[7])
        questions.append(turn[9])
        replies.append(turn[2])
        replies.append(turn[4])
        replies.append(turn[6])
        replies.append(turn[8])
        replies.append(turn[10])
    return questions, replies


def extractQAPairs():
    """
    Extract QA pairs (One list for questions and one list for answers) from given dataset

    :return: Questions and answers as lists
    """
    dialogs = []
    for file in glob.glob(os.path.join(config.DATA_PATH, '*.txt')):
        with open(file, 'r') as json_file:
            for dialog in json_file:
                dialogs.append(json.loads(dialog))

    turns = []
    counter = 0
    for dialog in dialogs:
        turns.append(dialogs[counter]["turns"])
        counter += 1

    filtered_turns = filterDialogs(turns)
    questions, replies = generateQAs(filtered_turns)

    return questions, replies


def loadDataset(dataset):
    """
    Retrieve QA pairs (as lists) from given dataset

    :param dataset: path of dataset to load data (the path includes the file name)
    :return: two list with strings (questions and answers)
    """
    questions, replies = [], []
    with open(dataset, 'r') as file:
        for line in file:
            items = line[:-1].split("\t")
            question = items[0]
            reply = items[1]
            questions.append(question)
            replies.append(reply)
    return questions, replies

def unicodeToAscii(s):
    """
    Unicode string to ASCII

    :param s: string
    :return: turn unicode string to ASCII
    """
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def preprocessString(s):
    """
    Preprocess sentence (lowercased, trimmed and without symbols and similar characters)

    :param s: string to preprocess
    :return: preprocessed string
    """
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

def generatePairs(lines):
    """
    Generate pairs

    :param lines: Lines to generate pairs. Each line contains a QA pair
    :return: Generated pairs
    """
    print("Generating pairs...")
    pairs = [[preprocessString(s) for s in l.split('\t')] for l in lines]
    return pairs

def generateVocabulary(dataset):
    """
    Generare vocabulary and QA pairs from given dataset

    :param dataset: Datasets
    :return: Generated vocabulary and QA pairs from the specific dataset
    """
    print("Generating vocabulary...")
    lines = open(dataset, encoding='utf-8'). \
        read().strip().split('\n')
    voc = vocabulary.Vocabulary()
    pairs = generatePairs(lines)
    return voc, pairs

def checkLengthThres(p):
    """
    Check the length of a sentence (based on a specifc maximum value/threshold)

    :param p: A QA pair
    :return: A Boolean.
    """
    return len(p[0].split(' ')) < config.MAX_LEN and len(p[1].split(' ')) < config.MAX_LEN

def filterPairs(pairs):
    """
    Filter pairs (check if the sentences exceed a specific threshold)

    :param pairs: QA pairs
    :return: filtered pairs (Length)
    """
    print("Checking maximum length threshold for QA pairs...")
    return [pair for pair in pairs if checkLengthThres(pair)]

def startPreproccesing(dataset):
    """
    Preprocess the dataset

    :param dataset: given dataset
    :return: Vocabulary and QA pairs
    """
    print("Preprocessing...")
    voc, pairs = generateVocabulary(dataset)
    pairs = filterPairs(pairs)
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    return voc, pairs
