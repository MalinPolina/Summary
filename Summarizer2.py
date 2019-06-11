import nltk.data
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import networkx as nx
import string
import itertools


def read_file(path, stop_words):
    translator = str.maketrans('', '', string.punctuation)
    data = open(path)
    file = data.read()
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    text = sent_detector.tokenize(file.strip())
    tuple(text)

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

def build_graph(nodes, vectors):
    vectnodes = zip(nodes, vectors)
    gr = nx.Graph() 
    gr.add_nodes_from(nodes)
    
    doublePairs = list(itertools.combinations(vectnodes, 2))
    for pair in doublePairs:
        first = pair[0][1]
        second = pair[1][1]
        fnode = pair[0][0]
        snode = pair[1][0]
        distance = 1 - cosine_distance(first, second)
        gr.add_edge(fnode, snode, weight=distance)

    return gr

def page_rank(G, weight='weight'):
    max_iter = 100
    d = 0.85
    tol = 1.0e-6
    
    if len(G) == 0:
        return {}
        
    if not G.is_directed():
        D = G.to_directed()
    else:
        D = G
    W = nx.stochastic_graph(D, weight='weight')
    N = G.number_of_nodes()
    rank = dict.fromkeys(W, 1.0 / N)
    pvector = dict.fromkeys(W, 1.0 / N)
    dweights = pvector
    dnodes = [n for n in W if W.out_degree(n, weight='weight') == 0.0]
    
    for _ in range(max_iter):
        last = rank
        rank = dict.fromkeys(last.keys(), 0)
        dsum = d * sum(last[n] for n in dnodes)
        for n in rank:
            for nbr in W[n]:
                rank[nbr] += d * last[n] * W[n][nbr][weight]
            rank[n] += dsum * dweights[n] + (1.0 - d) * pvector[n]
        err = sum([abs(rank[n] - last[n]) for n in rank])
        if err < N * tol:
            return rank

def generate_summary(path, top_n=5):
    stop_words = stopwords.words('english')
    summarize_text = []

    text, sentences, word_list = read_file(path, stop_words)
    vsentences = sentence_vector(sentences, word_list, stop_words)

    graph = build_graph(text, vsentences)
    calculated_page_rank = page_rank(graph, weight='weight')
    #calculated_page_rank = nx.pagerank(graph, weight='weight')
    keynodes = sorted(calculated_page_rank, key=calculated_page_rank.get, reverse=True)
    
    for k in range(top_n):
        summarize_text.append("".join(keynodes[k]))
    print("Summary: \n", " ".join(summarize_text))

if __name__ == '__main__':
    generate_summary("Cthulhu.txt", 5)
