'''
Constants used
'''
# General Flags
DEBUG = False
AUTONLAB = True
BASE_DIR = "/home/scratch/bpatra/" if AUTONLAB else "../"
DATA_DIR = BASE_DIR + "Data/"
VOCAB_FILE = DATA_DIR + ("word2ix.dat" if DEBUG else "word2ix_text8.dat")
SUBWORD_VOCAB_FILE = DATA_DIR + ("subword2ix.dat" if DEBUG else "subword2ix_text8.dat")
NGRAM_DICT = DATA_DIR + "ngram_dict.pkl"
BPE_DICT = DATA_DIR + "BPE/bpe_vocab.txt"
UNIGRAM_TABLE_FILE = DATA_DIR + ("unigram_table.npy" if DEBUG else "unigram_table_text8.npy")
CONSTRAINTS_DIR = DATA_DIR + "Linguistic_Constraints/"
PPDB_SYN_FILE = CONSTRAINTS_DIR + "ppdb_synonyms.txt"
PPDB_ANT_FILE = CONSTRAINTS_DIR + "ppdb_antonyms.txt"
WORDNET_ANT_FILE = CONSTRAINTS_DIR + "wordnet_antonyms.txt"
TEXT = DATA_DIR + ("tiny_text8" if DEBUG else "text8")
PAD_TOK = "PAD_TOK"
MIN_FREQ = 5
