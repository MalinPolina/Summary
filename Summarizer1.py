import nltk.data
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
import string


def read_file(path, stop_words):
    translator = str.maketrans('', '', string.punctuation)
    data = open(path)
    file = data.read()
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    text = sent_detector.tokenize(file.strip())

    sentences = []
    for sentence in text:
        new_sentence = sentence.translate(translator)
        sentences.append(word_tokenize(new_sentence.lower()))
    sentences.pop()

    new_file = file.translate(translator)
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(new_file.lower())
    wordl = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    word_list = list(wordl)

    return text, sentences, word_list

def sentence_vector (sentences, word_list, stop_words):
    lemmatizer = WordNetLemmatizer()
    vsentences = []
    vsentence = [0] * len(word_list)
    for sentence in sentences:
        sent = [lemmatizer.lemmatize(w) for w in sentence if w not in stop_words]
        for w in sent:
            vsentence[word_list.index(w)] += 1
        vsentences.append(vsentence)
    vsentences.pop()

    return vsentences

def build_adjmatrix(vsentences):
    adjacency_matrix = np.zeros((len(vsentences), len(vsentences)))

    for i in range(len(vsentences)):
        for j in range(len(vsentences)):
            if i == j:
                continue
            adjacency_matrix[i][j] = 1 - cosine_distance(vsentences[i], vsentences[j])

    return adjacency_matrix

def page_rank(adjacency_matrix):

    A = adjacency_matrix / adjacency_matrix.max(axis=0)
    N = A.shape[1]
    scores = np.random.rand(N, 1)
    scores = scores / np.linalg.norm(scores, 1)
    prev_scores = np.ones((N, 1), dtype=np.float32) * 100
    d = 0.85

    while np.linalg.norm(scores - prev_scores, 2) > 1.0e-7:
        prev_scores = scores
        scores = d * np.matmul(A, scores) + (1 - d) / N

    return scores


def generate_summary(path, top_n=5):
    stop_words = stopwords.words('english')
    summarize_text = []

    text, sentences, word_list = read_file(path, stop_words)

    vsentences = sentence_vector(sentences, word_list, stop_words)

    adjacency_matrix = build_adjmatrix(vsentences)

    #graph = nx.from_numpy_array(adjacency_matrix)
    #scores = nx.pagerank(graph, weight='weight')

    scores = page_rank(adjacency_matrix)
    #print(scores)

    rank_vector = sorted([scores for scores in enumerate(text)], key=lambda item: item[1], reverse=True)

    #print("Ranked_sentence order are ", rank_vector)

    for i in range(top_n):
        summarize_text.append("".join(rank_vector[i][1]))

    print("Summary: \n", " ".join(summarize_text))
    
    #file = open('test.txt', 'w')
    #file.write("\n ".join(summarize_text))
    #file.close()

generate_summary( "Cthulhu.txt", 5)
