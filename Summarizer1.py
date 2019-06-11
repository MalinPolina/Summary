import nltk.data
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
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
        vs = tuple(vsentence)
        vsentences.append(vs)
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
    max_iter = 100
    d = 0.85
    tol = 1.0e-7

    A = np.zeros(np.shape(adjacency_matrix))
    for i in range(np.size(adjacency_matrix, 0)):
       for j in range(np.size(adjacency_matrix, 1)):
           A[i][j] = adjacency_matrix[i][j]/np.linalg.norm(adjacency_matrix[i])

    N = A.shape[1]
    v = np.random.rand(N, 1)
    v = v / np.linalg.norm(v, 1)

    K = np.ones([N, N])/N
    for i in range(N):
        K[i][i] = 0
    M = np.transpose(np.matmul(K, A))

    for _ in range(max_iter):
        last_v = v
        v = d * np.matmul(M, last_v) / N + (1.0 - d) / N
        err = max(abs(v - last_v))
        if err < tol:
            lv = dict(enumerate(v))
            return lv
    print('Error')


def generate_summary(path, top_n=5):
    stop_words = stopwords.words('english')
    summarize_text = []

    text, sentences, word_list = read_file(path, stop_words)

    vsentences = sentence_vector(sentences, word_list, stop_words)

    adjacency_matrix = build_adjmatrix(vsentences)

    #graph = nx.from_numpy_array(adjacency_matrix)
    #scores = nx.pagerank(graph, weight='weight')
    scores = page_rank(adjacency_matrix)

    rank_vector = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    lscores=[]
    for rank in rank_vector:
        lscores.append(rank[0])

    for i in range(top_n):
        summarize_text.append("".join(text[lscores[i]]))

    print("Summary: \n", " ".join(summarize_text))

    #file = open('test.txt', 'w')
    #file.write("\n ".join(summarize_text))
    #file.close()


if __name__ == '__main__':
    generate_summary( "Cthulhu.txt", 5)
