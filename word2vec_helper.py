import multiprocessing
from gensim.models import Word2Vec

def embedding_sentences(sentences, embedding_size = 128, window = 5, min_count = 5, file_to_load = None, file_to_save = None):
    '''
    embeding_size 词嵌入维数
    window : 上下文窗口
    min_count : 词频少于min_count会被删除
    '''
    ## 模型保存与载入
    if file_to_load is not None:
        w2vModel = Word2Vec.load(file_to_load)
    else:
        w2vModel = Word2Vec(sentences, size = embedding_size, window = window, min_count = min_count, workers = multiprocessing.cpu_count())
        if file_to_save is not None:
            w2vModel.save(file_to_save)#保存模型

    all_vectors = []
    embeddingDim = w2vModel.vector_size
    # 嵌入维数
    embeddingUnknown = [0 for i in range(embeddingDim)]
    for sentence in sentences:
        this_vector = []
        for word in sentence:
            if word in w2vModel.wv.vocab:
                this_vector.append(w2vModel[word])
            else:
                this_vector.append(embeddingUnknown)
        all_vectors.append(this_vector)
    return all_vectors

def test():
    vectors = embedding_sentences([['first', 'sentence'], ['second', 'sentence']], embedding_size = 4, min_count = 1)
    print(vectors)
test()