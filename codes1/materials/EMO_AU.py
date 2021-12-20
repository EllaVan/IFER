import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import json

# get ex-au probality
def get_ex_au():
    ex_au_path = 'materials/RAF_ex_au.csv'
    data_ex_au = pd.read_csv(ex_au_path)
    exs = list(data_ex_au.iloc[:, 0])
    ex_au = np.array(data_ex_au.iloc[:, 1:])
    return exs, ex_au


# get train and test class name
def getnode(exs, tr_node_num):
    train_nodes = exs[:tr_node_num]
    test_nodes = exs[tr_node_num:]
    all_nodes = exs
    return train_nodes, test_nodes, all_nodes


# method to calculate ex trans matrix
def getTransitionProb(x1, x2):
    prob_sum = np.sum(x1 * x2)
    x1_num = len(np.nonzero(x1)[0])
    x2_num = len(np.nonzero(x2)[0])
    x1_x2 = prob_sum / x2_num
    x2_x1 = prob_sum / x1_num
    return x1_x2, x2_x1


# get ex trans matrix
def getTransitionMatrix(ex_au, threhold):
    num_exs = ex_au.shape[0]
    num_aus = ex_au.shape[1]
    trans_ex = np.zeros((num_exs, num_exs))
    self_connection = np.identity(num_exs)
    b = trans_ex
    for i in range(num_exs - 1):
        for j in range(i + 1, num_exs):
            y1, y2 = getTransitionProb(ex_au[i], ex_au[j])
            b[i][j] = y1
            b[j][i] = y2
    for i in range(num_exs):
        trans_ex[i] = b[i] / np.sum(b[i])
    # for i in range(num_exs):
    #     for j in range(num_exs):
    #         if trans_ex[i][j] > threhold:
    #             trans_ex[i][j] = 1
    #         else:
    #             trans_ex[i][j] = 0
    trans_ex = trans_ex + self_connection
    return trans_ex


# 计算表情转移矩阵方法2（对称矩阵）
def getTransitionMatrix2(ex_au, threhold):
    num_exs = ex_au.shape[0]
    num_aus = ex_au.shape[1]
    trans_ex_min = np.zeros((num_exs, num_exs))
    self_connection = np.identity(num_exs)
    trans_ex_max = trans_ex_min
    for i in range(num_exs - 1):
        nau_exi = len(np.nonzero(ex_au[i])[0])
        for j in range(i + 1, num_exs):
            nau_exj = len(np.nonzero(ex_au[i])[0])
            nau_exij = len(np.nonzero(ex_au[i] * ex_au[j])[0])
            if nau_exij >= np.min((nau_exi, nau_exj)) * threhold:
                trans_ex_min[i][j] = 1
                trans_ex_min[j][i] = 1
            else:
                trans_ex_min[i][j] = 0
                trans_ex_min[j][i] = 0
            if nau_exij >= np.max((nau_exi, nau_exj)) * threhold:
                trans_ex_max[i][j] = 1
                trans_ex_max[j][i] = 1
            else:
                trans_ex_max[i][j] = 0
                trans_ex_max[j][i] = 0
    trans_ex_min = trans_ex_min + self_connection
    trans_ex_max = trans_ex_max + self_connection
    return trans_ex_min, trans_ex_max


def genoutname(threshold):
    outname = 'RAF_graph' + str(threshold) + '.json'
    return outname


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ex_au_path', default='RAF_ex_au.csv')
    parser.add_argument('--output', default='RAF_graph.json')
    parser.add_argument('--num_expression', default=17)

    parser.add_argument('--au_au_path', default='au_au.csv')
    parser.add_argument('--num_au', default=26)
    parser.add_argument('--tr_node_num', default=6)
    parser.add_argument('--ex_au_threshold', default=0.5)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # get class name and ex-au correlation
    print('get expression names and expression-au correlation')
    exs, ex_au = get_ex_au()

    # get train class name and test class name
    print('get train class name and test class name')
    train_nodes, test_nodes, all_nodes = getnode(exs, args.tr_node_num)

    # get class transition matrix
    print('generate transition matrix')
    trans = getTransitionMatrix(ex_au, args.ex_au_threshold)
    # trans_min, trans_max = getTransitionMatrix2(ex_au, 0.6)

    # dump graph nodes, class vectors, au vectors, trans matrix
    # print('dumping graph ...')
    # outname = args.output
    # # outname = genoutname(args.ex_au_threshold)
    # obj = {'train_nodes': train_nodes, 'test_nodes': test_nodes, 'nodes': all_nodes,
    #        'class_embedding': class_embedding.tolist(),
    #        'edges': trans.tolist(), 'ex_au': ex_au.tolist(), 'au_au': au_au.tolist()}
    # torch.save(au_embedding, args.au_embedding_raw_path)
    # json.dump(obj, open(outname, 'w'))
