import json
import argparse
import heapq
import pickle

from pathlib import Path
from collections import defaultdict

PAD = 0
UNK = 1
BOS = 2
EOS = 3

OUT_DIR = "../tokenizer/"
IN_DIR = "../Dataset_A1/"


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

    def _apply_merges_to_tokens(self, tokens: list) -> list:
        """Run the learned merge rules over a single token list.
    
        Extracted so it can be applied per-segment (e.g. per word for
        whitespace pretokenization) without cross-segment merges.
        """
        for a, b in self.merge:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                    new_tokens.append(a + b)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens
    
    def apply_merges(self, sentence: str, pretokenize: str = "char") -> list:
        """Apply learned merge rules to tokenize a sentence.
    
        pretokenize must match whatever mode was used in train():
            "char"       - default; merges applied over the whole
                        sentence as single characters.
            "bits8"      - sentence is chunked into 8-character groups
                        before merges are applied.
            "whitespace" - sentence is split on whitespace; merges are
                        applied to each word independently (never
                        across a word boundary), and the resulting
                        token lists are concatenated.
        """
        if pretokenize == "bits8":
            tokens = [sentence[i:i + 8] for i in range(0, len(sentence), 8)]
            return self._apply_merges_to_tokens(tokens)
    
        if pretokenize == "whitespace":
            result = []
            for word in sentence.split():
                result.extend(self._apply_merges_to_tokens(list(word)+['</w>']))
            return result
    
        # "char"
        tokens = list(sentence)
        return self._apply_merges_to_tokens(tokens)


    def encode(self, sentence: list) -> list:
        return [self.encoder.get(tok, self.encoder['<unk>']) for tok in sentence]

    def decode(self, encoded_sent: list) -> str:
        decoded_list = [self.decoder.get(tok, '<unk>') for tok in encoded_sent]
        decoded_str = "".join(decoded_list).replace('</w>',' ')
        return decoded_str

    def train(self, dataset: list, num_merges: int, pretokenize: str = "char"):
        """Train BPE tokenizer.

        Uses per-line doubly-linked lists to avoid joining into one
        giant string. Global pair counts aggregate across all lines.
        Each merge only touches positions of the affected pair — no
        full-vocabulary scan. A max-heap provides O(log K) best-pair
        lookup with lazy deletion.

        pretokenize:
            "char"       - default, same as before: each dataset line
                        is one segment, initial tokens are single
                        characters.
            "bits8"      - each dataset line is one segment, but the
                        initial tokens are consecutive 8-character
                        (bit) groups instead of single bits. Use for
                        ciphertext.
            "whitespace" - each dataset line is split on whitespace;
                        every word becomes its OWN independent
                        segment, so merges can never cross a word
                        boundary. Initial tokens inside a word are
                        still single characters. Use for plaintext.
        """
        print("Initialising...", flush=True)

        # ---- Pretokenize lines into segments of initial tokens ----
        # A "segment" is an independent linked list — merges never
        # cross segment boundaries. That's exactly what whitespace
        # pretokenization needs (word boundaries become segment
        # boundaries); bits8/char just change what the initial unit
        # *inside* a segment is, and keep one segment per line.
        segments = []
        for line in dataset:
            content = line.rstrip('\n')
            if not content:
                segments.append([])
                continue
            if pretokenize == "bits8":
                segments.append(
                    [content[i:i + 8] for i in range(0, len(content), 8)]
                )
            elif pretokenize == "whitespace":
                for word in content.split():
                    segments.append(list(word)+['</w>'])
            else:  # "char"
                segments.append(list(content))

        num_lines = len(segments)

        # ---- Build per-segment data structures ----
        # Each segment gets its own tokens list + linked list arrays.
        # We track (line_idx, position) tuples in pair_positions.

        all_tokens = []     # all_tokens[line_idx] = list of current tokens
        all_next = []       # all_next[line_idx] = list of next-pointers
        all_prev = []       # all_prev[line_idx] = list of prev-pointers

        # Global pair tracking
        # pair -> set of (line_idx, pos) tuples
        pair_pos_lists = defaultdict(list)

        vocab = set()

        for li, tokens in enumerate(segments):
            clen = len(tokens)
            if clen == 0:
                all_tokens.append([])
                all_next.append([])
                all_prev.append([])
                continue

            all_tokens.append(tokens)

            # Build linked list for this segment
            nxt = list(range(1, clen)) + [-1]  # next[i] = i+1, last = -1
            prv = [-1] + list(range(0, clen - 1))  # prev[i] = i-1, first = -1
            all_next.append(nxt)
            all_prev.append(prv)

            # Collect vocab
            for tok in tokens:
                vocab.add(tok)

            # Collect pair positions
            for j in range(clen - 1):
                pair_pos_lists[(tokens[j], tokens[j + 1])].append((li, j))

        # Convert to sets and count
        pair_positions = {}
        pair_count = {}
        for pair, pos_list in pair_pos_lists.items():
            pair_positions[pair] = set(pos_list)
            pair_count[pair] = len(pos_list)
        del pair_pos_lists

        # Max-heap (negated counts, lazy deletion)
        heap = [(-c, p) for p, c in pair_count.items()]
        heapq.heapify(heap)

        print(f"Init done. {num_lines} segments, "
            f"unique pairs={len(pair_count)}.  "
            f"Starting {num_merges} merges...", flush=True)

        # ---- Merge loop (UNCHANGED — operates generically on tokens,
        # so it doesn't care whether they started as single chars,
        # 8-bit groups, or word-scoped chars) ----
        _heappush = heapq.heappush  # local ref for speed

        for merge_i in range(num_merges):
            if (merge_i + 1) % 1000 == 0:
                print(f"  merge {merge_i + 1}/{num_merges}", flush=True)

            # Find best pair via heap with lazy deletion
            best_pair = None
            while heap:
                neg_count, candidate = heapq.heappop(heap)
                if candidate in pair_count and pair_count[candidate] == -neg_count:
                    best_pair = candidate
                    break
            if best_pair is None:
                break

            a, b = best_pair
            new_tok = a + b
            vocab.add(new_tok)
            self.merge.append(best_pair)

            # Pop positions of the best pair
            positions = pair_positions.pop(best_pair)
            del pair_count[best_pair]

            for li, pos in positions:
                toks = all_tokens[li]
                nxt = all_next[li]
                prv = all_prev[li]

                right = nxt[pos]

                # Validate — may have been invalidated by an earlier
                # position in this same merge batch
                if right == -1 or toks[pos] != a or toks[right] != b:
                    continue

                left = prv[pos]
                right_right = nxt[right]

                # Remove old left pair: (toks[left], a) at (li, left)
                if left != -1:
                    old_lp = (toks[left], a)
                    if old_lp in pair_positions:
                        key = (li, left)
                        pp = pair_positions[old_lp]
                        if key in pp:
                            pp.discard(key)
                            pair_count[old_lp] -= 1
                            if pair_count[old_lp] <= 0:
                                del pair_count[old_lp]
                                del pair_positions[old_lp]

                # Remove old right pair: (b, toks[right_right]) at (li, right)
                if right_right != -1:
                    old_rp = (b, toks[right_right])
                    if old_rp in pair_positions:
                        key = (li, right)
                        pp = pair_positions[old_rp]
                        if key in pp:
                            pp.discard(key)
                            pair_count[old_rp] -= 1
                            if pair_count[old_rp] <= 0:
                                del pair_count[old_rp]
                                del pair_positions[old_rp]

                # Perform the merge
                toks[pos] = new_tok
                toks[right] = None  # consumed

                # Update linked list
                nxt[pos] = right_right
                if right_right != -1:
                    prv[right_right] = pos

                # Add new left pair
                if left != -1:
                    new_lp = (toks[left], new_tok)
                    key = (li, left)
                    if new_lp not in pair_positions:
                        pair_positions[new_lp] = set()
                        pair_count[new_lp] = 0
                    pair_positions[new_lp].add(key)
                    pair_count[new_lp] += 1
                    _heappush(heap, (-pair_count[new_lp], new_lp))

                # Add new right pair
                if right_right != -1:
                    new_rp = (new_tok, toks[right_right])
                    key = (li, pos)
                    if new_rp not in pair_positions:
                        pair_positions[new_rp] = set()
                        pair_count[new_rp] = 0
                    pair_positions[new_rp].add(key)
                    pair_count[new_rp] += 1
                    _heappush(heap, (-pair_count[new_rp], new_rp))

        # Build encoder from vocab
        buffer = max(self.encoder.values()) + 1
        for i, tok in enumerate(sorted(vocab, key=lambda x: len(x))):
            self.encoder[tok] = i + buffer

        for key, value in self.encoder.items():
            self.decoder[value] = key

        print("Training complete.", flush=True)

    def save(self, path):
        data = {
            "encoder" : self.encoder,
            "merge" : self.merge
        }
        with open(path,"w") as victory:
            json.dump(data,victory,indent=4)

    def load(self, path):
        with open(path,"r") as victory:
            data = json.load(victory)
        self.encoder = data["encoder"]
        self.merge = [tuple(pair) for pair in data["merge"]]
        self.decoder = {v:k for k,v in self.encoder.items()}



def prep_data(tokpath, datapath, outpath, type="char"):
    tokenizer = Tokenizer()
    tokenizer.load(tokpath)
    with open(datapath, "r") as data_file:
        data = data_file.readlines()

    tokenized_data = []
    i=0
    for sentence in data:
        tok_sent = tokenizer.apply_merges(sentence, type)
        tok_sent = tokenizer.encode(tok_sent)
        tokenized_data.append(tok_sent)
        i+=1
        if i%100==0:
            print(f"{i} sentences completed")

    with open(outpath, "wb") as out_file:
        pickle.dump(tokenized_data, out_file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--merges", type=int, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--pretok", required=True)

    args = parser.parse_args()

    num_merges = args.merges
    dataset = args.dataset
    pretok = args.pretok

    datapath = IN_DIR+dataset
    outpath = OUT_DIR+dataset.split('.')[0]+str(num_merges)+f'_{pretok}'+'.json'

    with open(datapath,"r") as file:
        data = file.readlines()

    tokenizer = Tokenizer()

    if Path(outpath).exists():
        tokenizer.load(outpath)
    else:
        tokenizer.train(data, num_merges, pretokenize=pretok)
        tokenizer.save(outpath)

    print("Tokenizing the dataset...")
    tokenized_path = f"../{dataset.split('.')[0]}_tokenized.pkl"
    prep_data(outpath, datapath, tokenized_path, type=pretok)