import json
import argparse
import heapq

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

    def apply_merges(self, sentence: str) -> list:
        """Apply learned merge rules to tokenize a sentence.

        Rebuilds the token list each pass instead of using list.pop(),
        avoiding O(N) shifts per pop.
        """
        tokens = list(sentence)
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

    def encode(self, sentence: list) -> list:
        return [self.encoder.get(tok, self.encoder['<unk>']) for tok in sentence]

    def decode(self, encoded_sent: list) -> list:
        return [self.decoder.get(tok, '<unk>') for tok in encoded_sent]

    def train(self, dataset: list, num_merges: int):
        """Train BPE tokenizer.

        Uses per-line doubly-linked lists to avoid joining into one
        giant string. Global pair counts aggregate across all lines.
        Each merge only touches positions of the affected pair — no
        full-vocabulary scan. A max-heap provides O(log K) best-pair
        lookup with lazy deletion.
        """
        print("Initialising...", flush=True)

        # ---- Build per-line data structures ----
        # Each line gets its own tokens list + linked list arrays.
        # We track (line_idx, position) tuples in pair_positions.

        num_lines = len(dataset)

        # Per-line storage
        all_tokens = []     # all_tokens[line_idx] = list of current tokens
        all_next = []       # all_next[line_idx] = list of next-pointers
        all_prev = []       # all_prev[line_idx] = list of prev-pointers

        # Global pair tracking
        # pair -> set of (line_idx, pos) tuples
        pair_pos_lists = defaultdict(list)

        vocab = set()

        for li, line in enumerate(dataset):
            # Strip trailing newline for content
            content = line.rstrip('\n')
            clen = len(content)
            if clen == 0:
                all_tokens.append([])
                all_next.append([])
                all_prev.append([])
                continue

            tokens = list(content)
            all_tokens.append(tokens)

            # Build linked list for this line
            nxt = list(range(1, clen)) + [-1]  # next[i] = i+1, last = -1
            prv = [-1] + list(range(0, clen - 1))  # prev[i] = i-1, first = -1
            all_next.append(nxt)
            all_prev.append(prv)

            # Collect vocab
            for c in content:
                vocab.add(c)

            # Collect pair positions
            for j in range(clen - 1):
                pair_pos_lists[(content[j], content[j + 1])].append((li, j))

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

        print(f"Init done. {num_lines} lines, "
              f"unique pairs={len(pair_count)}.  "
              f"Starting {num_merges} merges...", flush=True)

        # ---- Merge loop ----
        _heappush = heapq.heappush  # local ref for speed

        for merge_i in range(num_merges):
            if (merge_i + 1) % 1 == 0:
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

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--merges", type=int, required=True)
    parser.add_argument("--dataset", required=True)

    args = parser.parse_args()

    num_merges = args.merges
    dataset = args.dataset

    datapath = IN_DIR+dataset
    outpath = OUT_DIR+dataset.split('.')[0]+str(num_merges)+'.json'

    with open(datapath,"r") as file:
        data = file.readlines()

    tokenizer = Tokenizer()

    if Path(outpath).exists():
        tokenizer.load(outpath)
    else:
        tokenizer.train(data, num_merges)
        tokenizer.save(outpath)