import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold,StratifiedKFold
from DataReader import FeatureDictionary, DataParser
from matplotlib import pyplot as plt
import codecs
from collections import defaultdict

import config
from KDFM import DeepAFM
from metrics import gini_norm, mse_norm


def load_data():
    dfTrain = pd.read_csv(config.TRAIN_FILE)
    dfTest = pd.read_csv(config.TEST_FILE)

    dfTrain.drop(config.TEXT_COLS, axis=1, inplace=True)
    dfTest.drop(config.TEXT_COLS, axis=1, inplace=True)

    cols = [c for c in dfTrain.columns if c not in ['review_ratting']]
    cols = [c for c in cols if (not c in config.IGNORE_COLS)]

    X_train = dfTrain[cols].values
    y_train = dfTrain['review_ratting'].values
    X_test = dfTest[cols].values
    y_test = dfTest['review_ratting'].values

    # Xm_train = xmTrain[xm_cols].values
    # Xm_test = xmTest[xm_cols].values

    return dfTrain, dfTest, X_train, y_train, X_test, y_test

def read_text_data(filename, word2idx, sequence_len):
    unknown_id = word2idx.get("UNKNOWN", 0)
    data_x, data_mask = [], []
    try:
        file = pd.read_csv(filename, sep=',')
        for row in range(len(file)):
            user_review = file['user_review'][row].strip().split(" ")
            app_review = file['app_review'][row].strip().split(" ")
            description = file['description'][row].strip().split(" ")

            user_sent_idx = [word2idx.get(word.strip(), unknown_id) for word in user_review[:-1]]
            app_sent_idx = [word2idx.get(word.strip(), unknown_id) for word in app_review[:-1]]
            des_sent_idx = [word2idx.get(word.strip(), unknown_id) for word in description[:-1]]

            # padding
            pad_idx = word2idx.get("<a>", unknown_id)
            ux, umask = np.ones(sequence_len, np.int32) * pad_idx, np.zeros(sequence_len, np.int32)
            ax, amask = np.ones(sequence_len, np.int32) * pad_idx, np.zeros(sequence_len, np.int32)
            dx, dmask = np.ones(sequence_len, np.int32) * pad_idx, np.zeros(sequence_len, np.int32)
            if len(user_sent_idx) < sequence_len:
                ux[:len(user_sent_idx)] = user_sent_idx
                umask[:len(user_sent_idx)] = 1
            else:
                ux = user_sent_idx[:sequence_len]
                umask[:] = 1
            if len(app_sent_idx) < sequence_len:
                ax[:len(app_sent_idx)] = app_sent_idx
                amask[:len(app_sent_idx)] = 1
            else:
                ax = app_sent_idx[:sequence_len]
                amask[:] = 1
            if len(des_sent_idx) < sequence_len:
                dx[:len(des_sent_idx)] = des_sent_idx
                dmask[:len(des_sent_idx)] = 1
            else:
                dx = des_sent_idx[:sequence_len]
                dmask[:] = 1
            temp = []
            temp_mask = []
            temp.append(ux)
            temp.append(ax)
            temp.append(dx)
            data_x.append(temp)
            temp_mask.append(umask)
            temp_mask.append(amask)
            temp_mask.append(dmask)
            data_mask.append(temp_mask)
    except Exception as e:
        print("load file Exception," + e)

    return data_x, data_mask

def build_vocab(filename):
    word2idx, idx2word = defaultdict(), defaultdict()
    try:
        with codecs.open(filename, mode="r", encoding="utf-8") as rf:
            for line in rf.readlines():
                items = line.strip().split(" ")
                if len(items) != 2:
                    continue  # 跳出本次循环
                word_id = int(items[0].strip())  # strip() 用于移除字符串头尾指定的字符
                word = items[1].strip()
                idx2word[word_id] = word  # key：word_id   value: word
                word2idx[word] = word_id  # key: word     value: word_id

            print("build_vocab finish")
            rf.close()

        word2idx["UNKNOWN"] = len(idx2word)  # word2idx key：UNKNOWN 中放的 value：idx2word的长度
        idx2word[len(idx2word)] = "UNKNOWN"  # idx2word key: idx2word的长度 中放的 UNKNOWN
        word2idx["<a>"] = len(idx2word)
        idx2word[len(idx2word)] = "<a>"
    except Exception as e:
        print(e)
    return word2idx, idx2word

def load_embedding(embedding_size, word2idx=None, filename=None):
    if filename is None and word2idx is not None:
        return load_embedding_random_init(word2idx, embedding_size)
    else:
        return load_embedding_from_file(filename)

def load_embedding_random_init(word2idx, embedding_size):
    embeddings=[]
    for word, idx in word2idx.items():
        vec = [0.01 for i in range(embedding_size)]
        embeddings.append(vec)
    return np.array(embeddings, dtype="float32")

def load_embedding_from_file(embedding_file):
    word2vec_embeddings = np.array([[float(v) for v in line.strip().split(' ')] for line in open(embedding_file).readlines()], dtype=np.float32)
    embedding_size = word2vec_embeddings.shape[1]
    unknown_padding_embedding = np.random.normal(0, 0.1, (2,embedding_size))

    embeddings = np.append(word2vec_embeddings, unknown_padding_embedding.astype(np.float32), axis=0)
    return embeddings

def run_base_model_nfm(dfTrain, dfTest, folds, pnn_params):
    fd = FeatureDictionary(dfTrain=dfTrain,
                           dfTest=dfTest,
                           numeric_cols=config.NUMERIC_COLS,
                           ignore_cols=config.IGNORE_COLS,
                           xm_cols=config.XM_COLS)
    data_parser = DataParser(feat_dict=fd)

    # 新添
    word2idx, idx2word = build_vocab(config.word_file)

    # Xi_train ：列的序号
    # Xv_train ：列的对应的值
    Xi_train, Xv_train, y_train = data_parser.parse(df=dfTrain)
    Xt_train, Xm_train = read_text_data(config.TRAIN_FILE, word2idx, config.num_unroll_steps)  # read data TODO：config 与 pnn_params
    Xi_test, Xv_test, y_test = data_parser.parse(df=dfTest)
    Xt_test, Xm_test = read_text_data(config.TEST_FILE, word2idx, config.num_unroll_steps)


    pnn_params['feature_size_one_hot'] = fd.feat_dim
    pnn_params['word_embeddings'] = load_embedding(config.embedding_size, filename=config.embedding_file)  # read data

    #TODO:change
    y_train_meta = np.zeros((dfTrain.shape[0], 1), dtype=float)
    y_test_meta = np.zeros((dfTest.shape[0], 1), dtype=float)

    results_cv = np.zeros(len(folds), dtype=float)
    results_epoch_train = np.zeros((len(folds), pnn_params['epoch']), dtype=float)
    results_epoch_valid = np.zeros((len(folds), pnn_params['epoch']), dtype=float)
    results_epoch_train_mae = np.zeros((len(folds), pnn_params['epoch']), dtype=float)
    results_epoch_valid_mae = np.zeros((len(folds), pnn_params['epoch']), dtype=float)

    def _get(x, l): return [x[i] for i in l]

    for i, (train_idx, valid_idx) in enumerate(folds):
        Xi_train_, Xv_train_, y_train_, Xt_train_, Xm_train_ = \
            _get(Xi_train, train_idx), _get(Xv_train, train_idx), _get(y_train, train_idx), \
            _get(Xt_train, train_idx), _get(Xm_train, train_idx)

        Xi_valid_, Xv_valid_, y_valid_, Xt_valid_, Xm_valid_ = \
            _get(Xi_train, valid_idx), _get(Xv_train, valid_idx), _get(y_train, valid_idx), \
            _get(Xt_train, valid_idx), _get(Xm_train, valid_idx)

        afm = DeepAFM(**pnn_params)
        Xim_train_ = []
        Xvm_train_ = []
        Xim_valid_ = []
        Xvm_vaild_ = []
        Xim_test = []
        Xvm_test = []


        afm.fit(Xi_train_, Xv_train_, Xim_train_, Xvm_train_, Xt_train_, y_train_,
                Xi_valid_, Xv_valid_, Xim_valid_, Xvm_vaild_, Xt_valid_,y_valid_)

        y_train_meta[valid_idx, 0] = afm.predict(Xi_valid_, Xv_valid_, Xim_valid_, Xvm_vaild_, Xt_valid_)
        y_test_meta[:, 0] += afm.predict(Xi_test, Xv_test, Xim_test, Xvm_test, Xt_test)


        results_cv[i] = mse_norm(y_valid_, y_train_meta[valid_idx])
        results_epoch_train[i] = afm.train_result
        results_epoch_valid[i] = afm.valid_result

        results_epoch_train_mae[i] = afm.mae_train_result
        results_epoch_valid_mae[i] = afm.mae_valid_result

    y_test_meta /= float(len(folds))

    # save result
    if pnn_params["use_afm"] and pnn_params["use_deep"]:
        clf_str = "KDFM"
    elif pnn_params["use_afm"]:
        clf_str = "AFM"
    elif pnn_params["use_deep"]:
        clf_str = "DNN"
    print("%s: %.5f (%.5f)" % (clf_str, results_cv.mean(), results_cv.std()))
    filename = "%s_Mean%.5f_Std%.5f.csv" % (clf_str, results_cv.mean(), results_cv.std())
    filename1 = "params%s_Mean%.5f_Std%.5f.csv" % (clf_str, results_cv.mean(), results_cv.std())
    _make_submission(y_test, y_test_meta, filename)
    _make_pnn_params(pnn_params.keys(), pnn_params.values(), filename1)

    _plot_fig(results_epoch_train, results_epoch_valid, clf_str+'mse', "mse")
    _plot_fig(results_epoch_train_mae, results_epoch_valid_mae, clf_str+'mae', "mae")

def _make_submission(target, y_pred, filename="submission.csv"):
    pd.DataFrame({"id": range(len(target)), "target": target, "predict": y_pred.flatten()}).to_csv(
        os.path.join(config.SUB_DIR, filename), index=False, float_format="%.5f")

def _make_pnn_params(key, value, filename="pnn_params.csv"):
    pd.DataFrame({"key": key, "value": value}).to_csv(
        os.path.join(config.SUB_DIR, filename), index=False, float_format="%.5f")


def _plot_fig(train_results, valid_results, model_name, algor):
    colors = ["red", "blue", "green"]
    xs = np.arange(1, train_results.shape[1]+1)
    plt.figure()
    legends = []
    for i in range(train_results.shape[0]):
        plt.plot(xs, train_results[i], color=colors[i], linestyle="solid", marker="o")
        plt.plot(xs, valid_results[i], color=colors[i], linestyle="dashed", marker="o")
        legends.append("train-%d"%(i+1))
        legends.append("valid-%d"%(i+1))
    plt.xlabel("Epoch")
    if algor == 'mae':
        plt.ylabel("mae")
    if algor == 'mse':
        plt.ylabel("mse")
    plt.title("%s"%model_name)
    plt.legend(legends)
    plt.savefig("fig/%s.png"%model_name)
    plt.close()

# TODO: lack of feature_size & word_embeddings
pnn_params = {
    "use_afm": True,
    "use_deep": True,
    #"field_size": 6,
    "feature_size_one_hot": 1,
    "field_size_one_hot": 3,
    "feature_size_multi_value": 0,
    "field_size_multi_value": 0,
    "embedding_size": 8,
    "attention_size": 10,

    "deep_layers": [32, 32, 32],
    "dropout_deep": [0.5, 0.5, 0.5, 0.5],
    "deep_layer_activation": tf.nn.relu,

    "epoch": 25,
    "batch_size": 128,
    "learning_rate": 0.001,
    "optimizer": "adam",

    "random_seed": config.RANDOM_SEED,
    "l2_reg": 0.2,

    "rnn_size": 100,
    "num_rnn_layers": 1,
    "keep_lstm": 0.5,
    "num_unroll_steps": 100,  # 句子长度
    "verbose": True,
    "topics": 10
}


# load data
dfTrain, dfTest, X_train, y_train, X_test, y_test = load_data()

# folds
# TODO: StratifiedKFold 修改为 KFold
folds = list(StratifiedKFold(n_splits=config.NUM_SPLITS, shuffle=True,
                             random_state=config.RANDOM_SEED).split(X_train, y_train))

run_base_model_nfm(dfTrain, dfTest, folds, pnn_params)

# ------------------ FM Model ------------------
afm_params = pnn_params.copy()
pnn_params["use_deep"] = False
run_base_model_nfm(dfTrain, dfTest, folds, afm_params)

# ------------------ DNN Model ------------------
# dnn_params = pnn_params.copy()
# pnn_params["use_afm"] = False
# run_base_model_nfm(dfTrain, dfTest, folds, dnn_params)
