from collections import Counter

PAD = 0
UNK = 1
BOS = 2
EOS = 3

class Tokenizer:
    def __init__(self):
        self.encoder = {
            '<unk>':UNK, '<pad>':PAD, '<bos>':BOS, '<eos>':EOS
        }
        self.decoder = {}
        self.merge = []

    @staticmethod
    def _drop_long_seq(dataset, max_len):
        return [s for s in dataset if len(s)<=len]

    def apply_merges(self, sentence: str) -> list:
        sentence = list(sentence)
        for a, b in self.merge:
            pairs = list(zip(sentence[:-1],sentence[1:]))
            merge_idx = []
            for i, pair in enumerate(pairs):
                if pair == (a,b):
                    merge_idx.append(i)

            tok = a+b
            for i in reversed(merge_idx):
                if sentence[i]!=a or sentence[i+1]!=b:
                    continue
                sentence.pop(i+1)
                sentence[i] = tok
        return sentence

    def encode(self, sentence: list) -> list:
        encoded_sent = []
        for tok in sentence:
            encoded_sent.append(self.encoder.get(tok,self.encoder['<unk>']))
        return encoded_sent

    def decode(self, encoded_sent: list) -> list:
        sent = []
        for tok in encoded_sent:
            sent.append(self.decoder.get(tok))
        return sent

    

    def train(self, dataset: list, num_merges: int, max_seq_len: int):
        vocab = set()
        dataset = self._drop_long_seq(dataset, max_seq_len)
        for i in range(len(dataset)):
            dataset[i] = list(dataset[i])
            vocab.update(dataset[i])

        for i in range(num_merges):
            pair_counter = Counter()
            for sent in dataset:
                pair_counter.update(zip(sent[:-1],sent[1:]))

            best_pair = max(pair_counter, key=pair_counter.get)
            self.merge.append(best_pair)

        pass

    def save(self, path):
        pass

    def load(self, path):
        pass