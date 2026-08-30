import matplotlib.pyplot as plt
import pickle

from tokenizer import Tokenizer

PLOTS_DIR = "../outputs/"

def plot_helper(x, y, xl, yl, title, fname):
    plt.figure(figsize=(12, 5))

    plt.plot(x,y,linewidth=1)

    plt.title(title)
    plt.xlabel(xl)
    plt.ylabel(yl)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR+fname)
    plt.show()

def plot_lengths(cipher, plain):
    with open(cipher,"r") as ci_file:
        ci_data = ci_file.readlines()

    with open(plain,"r") as pl_file:
        pl_data = pl_file.readlines()

    ci_len = []
    pl_len = []
    ratio = []
    for i in range(len(ci_data)):
        ci_len.append(len(ci_data[i].rstrip('\n')))
        pl_len.append(len(pl_data[i].rstrip('\n')))
        ratio.append(ci_len[-1]/pl_len[-1])

    plot_helper(ci_len, pl_len, "cipher", "plain", "len vs len", "lengths.png")
    plot_helper(range(1,5001), ratio, "id", "ratio", "ratio", "ratio.png")

def calc_tok_lengths(cipher, plain):
    ci_tok = Tokenizer(); pl_tok = Tokenizer()
    ci_tok.load(cipher); pl_tok.load(plain)
    with open("../Dataset_A1/brown_cipher.txt","r") as ci_file:
        ci_data = sorted(ci_file.readlines(), key=lambda x: -len(x))[:100]

    with open("../Dataset_A1/brown_plain.txt","r") as pl_file:
        pl_data = sorted(pl_file.readlines(), key=lambda x: -len(x))[:100]

    tok_ratio = []
    ci_toks_sent = []
    pl_toks_sent = []
    for i in range(100):
        ci_sent = ci_tok.apply_merges(ci_data[i], "bits8")
        pl_sent = pl_tok.apply_merges(pl_data[i], "whitespace")
        tok_ratio.append(len(ci_sent)/len(pl_sent))
        ci_toks_sent.append(len(ci_sent)**(-1)/len("".join(ci_sent))**(-1))
        pl_toks_sent.append(len(pl_sent)**(-1)/len("".join(pl_sent))**(-1))
    print(tok_ratio)
    print(ci_toks_sent)
    print(pl_toks_sent)



if __name__=="__main__":
    # DATA_DIR = "../Dataset_A1/"
    # plot_lengths(DATA_DIR+"brown_cipher.txt", DATA_DIR+"brown_plain.txt")

    # TOK_DIR = "../tokenizer/"
    # calc_tok_lengths(TOK_DIR+"brown_cipher2000_bits8.json",TOK_DIR+"brown_plain5000_whitespace.json")

    # with open("../brown_plain_tokenized.pkl","rb") as file:
    #     data = pickle.load(file)

    # tok = Tokenizer()
    # tok.load(TOK_DIR+"brown_plain5000_whitespace.json")
    # for sentence in data[:3]:
    #     print(tok.decode(sentence))

    with open("../brown_cipher_tokenized.pkl","rb") as file:
        data = pickle.load(file)

    data = sorted(data, key=lambda x: -len(x))
    print(len(data[1]))

