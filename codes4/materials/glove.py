import torch
import torch.nn.functional as F


class GloVe():

    def __init__(self, file_path):
        self.dimension = None
        self.embedding = dict()
        with open(file_path, 'r', encoding="utf-8") as f:
            for line in f.readlines():
                strs = line.strip().split()
                word = strs[0]
                vector = torch.FloatTensor(list(map(float, strs[1:])))
                self.embedding[word] = vector
                if self.dimension is None:
                    self.dimension = len(vector)

    def _fix_word(self, word):  # 复合词的词向量表示是内含所有词的平均
        terms = word.replace('_', ' ').split(' ')
        ret = self.zeros()
        cnt = 0
        for term in terms:
            v = self.embedding.get(term)
            if v is None:
                subterms = term.split('-')
                subterm_sum = self.zeros()
                subterm_cnt = 0
                for subterm in subterms:
                    subv = self.embedding.get(subterm)
                    if subv is not None:
                        subterm_sum += subv
                        subterm_cnt += 1
                if subterm_cnt > 0:
                    v = subterm_sum / subterm_cnt
            if v is not None:
                ret += v
                cnt += 1
        return ret / cnt if cnt > 0 else None

    def __getitem__(self, words):
        if type(words) is str:
            words = [words]
        ret = self.zeros()
        cnt = 0
        for word in words:
            v = self.embedding.get(word)
            if v is None:
                v = self._fix_word(word)
            if v is not None:
                ret += v
                cnt += 1
        if cnt > 0:
            return ret / cnt
        else:
            return self.zeros()

    def zeros(self):
        return torch.zeros(self.dimension)

    def getdimension(self):
        return self.dimension


def get_au_embedding(glove, au_action_path):
    au_description = {}
    with open(au_action_path, "r") as f:
        for line in f.readlines():
            line = line.strip('\n')  # 去掉列表中每一个元素的换行符
            terms = line.split(':')
            au_description[terms[0]] = terms[1]
    au_name = list(au_description)
    au_vectors = []
    au_embedding = {}
    for i in range(len(au_name)):
        au_vectors.append(glove[au_description[au_name[i]]])
    au_vectors = torch.stack(au_vectors)
    au_vectors = F.normalize(au_vectors)
    for i in range(len(au_name)):
        au_embedding[au_name[i]] = au_vectors[i]
    return au_embedding


if __name__ == '__main__':
    glove = GloVe('/media/database/data4/wf/GraphNet-FER/work01/materials/glove.6B.300d.txt')
    au_embedding = get_au_embedding(glove, 'AU_action.txt')

