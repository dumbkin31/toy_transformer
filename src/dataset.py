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
        return [s for s in dataset if len(s)<=max_len]

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
            sent.append(self.decoder.get(tok, '<unk>'))
        return sent

    def train(self, dataset: list, num_merges: int):
        # assumes that every element of the dataset ends with \n
        dataset = ''.join(dataset)
        vocab = set(dataset)
        if '\n' in vocab:
            vocab.remove('\n')
        pairs = list(zip(dataset[:-1],dataset[1:]))
        pair_counter = Counter(pairs)
        next_idx = []
        prev_idx = []
        pair_positions = {k: set() for k in pair_counter}
        for i, pair in enumerate(pairs):
            pair_positions[pair].add(i)

        for i in range(len(dataset)):

            if dataset[i] != '\n' and i>0 and dataset[i-1] != '\n':
                prev_idx.append(i-1)
            else:
                prev_idx.append(-1)

            if dataset[i] != '\n' and i<len(dataset)-1 and dataset[i+1] != '\n':
                next_idx.append(i+1)
            else:
                next_idx.append(-1)

        for a,b in list(pair_positions.keys()):
            if a == '\n' or b == '\n':
                pair_positions.pop((a,b))
                pair_counter.pop((a,b))

        for i in range(num_merges):
            # find max freq pair
            if not len(pair_counter):
                break
            best_pair = max(pair_counter, key=pair_counter.get)
            self.merge.append(best_pair)
            # count = pair_counter.pop(best_pair)
            a, b = best_pair
            tok = a+b
            vocab.add(a+b)

            # update position info and find positions of adjacent pairs
            positions = pair_positions.get(best_pair)
            lnew = set()
            consumed = set()
            for pos in sorted(positions):
                if pos in consumed:
                    continue
                lnew.add(prev_idx[pos])
                consumed.add(next_idx[pos])
                next_idx[pos] = next_idx[next_idx[pos]]
                prev_idx[next_idx[pos]] = pos if next_idx[pos]>-1 else prev_idx[next_idx[pos]]

            for pair in list(pair_positions.keys()):
                #right pairs
                cons_intersect = pair_positions[pair].intersection(consumed)
                pair_positions[pair] -= consumed
                pair_counter[pair] -= len(cons_intersect)
                for pos in sorted(cons_intersect):
                    left_idx = prev_idx[pos]
                    right_idx = next_idx[left_idx]
                    if right_idx in positions-consumed:
                        new_pair = (tok,tok)
                    else:
                        new_pair = (tok,pair[1])

                    pair_counter[new_pair] += 1
                    pair_positions[new_pair] = pair_positions.get(new_pair, set()) | {left_idx}

                #left pairs
                if pair == best_pair:
                    continue
                left_intersect = pair_positions[pair].intersection(lnew)
                pair_positions[pair] -= lnew
                pair_counter[pair] -= len(left_intersect)
                new_pair = (pair[0],tok)
                pair_positions[new_pair] = pair_positions.get(new_pair, set()) | left_intersect
                pair_counter[new_pair] = len(pair_positions[new_pair])

            pair_positions.pop(best_pair)
            pair_counter.pop(best_pair)

            for pair in list(pair_positions.keys()):
                if pair_counter[pair]<=0:
                    pair_counter.pop(pair)
                    pair_positions.pop(pair)

        buffer = max(self.encoder.values())+1
        for i, tok in enumerate(vocab):
            self.encoder[tok] = i+buffer

        for key, value in self.encoder.items():
            self.decoder[value] = key

    def save(self, path):
        pass

    def load(self, path):
        pass

tok = Tokenizer()

tok.train(list('abcabc\nabcabc\n'),5)

print(tok.merge)
print(tok.encoder)
print(tok.decoder)