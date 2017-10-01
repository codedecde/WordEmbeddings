import argparse
import sys
import io
from Lang import Vocab
from DataProcessing import iterator
from word2vec import Word2Vec


def get_arguments():
    def check_boolean(args, attr_name):
        assert hasattr(args, attr_name), "%s not found in parser" % (attr_name)
        bool_set = set(["true", "false"])
        args_value = getattr(args, attr_name)
        args_value = args_value.lower()
        assert args_value in bool_set, "Boolean argument required for attribute %s" % (attr_name)
        args_value = False if args_value == "false" else True
        setattr(args, attr_name, args_value)
        return args
    parser = argparse.ArgumentParser(description='Word2Vec in pytorch')
    parser.add_argument('-n_embed', action="store", default=300, dest="n_embed", type=int)
    parser.add_argument('-batch', action="store", default=256, dest="batch_size", type=int)
    parser.add_argument('-lr', action="store", default=0.025, dest="lr", type=float)
    parser.add_argument('-momentum', action="store", default=0.9, dest="momentum", type=float)
    parser.add_argument('-n_epochs', action="store", default=15, dest="n_epochs", type=int)
    parser.add_argument('-window', action="store", default=8, dest="window", type=int)
    parser.add_argument('-neg_samples', action="store", default=100, dest="neg_samples", type=int)
    # Using strings as a proxy for boolean flags. Checks happen later
    args = parser.parse_args(sys.argv[1:])
    # Checks for the boolean flags
    return args


if __name__ == "__main__":
    args = get_arguments()
    data_dir = "../"
    data_file = data_dir + "Data/text8"
    THRESHOLD = -1
    raw_data = io.open(data_file, encoding='utf-8', mode='r', errors='replace').read(THRESHOLD).split(u' ')[1:]
    vocab_file = data_dir + "Models/Vocab_Mincount_10.pkl"
    vocab = Vocab()
    vocab.load_file(vocab_file)
    data = []
    for word in raw_data:
        if word in vocab.word2ix:
            data.append(word)
    batch_size = args.batch_size
    data_iterator = iterator(data, vocab, num_samples=args.neg_samples, batch_size=batch_size, window_size=args.window)
    w2v = Word2Vec(num_classes=len(vocab), embed_size=300, num_words=len(data))
    steps_per_epoch = len(data) // batch_size if len(data) % batch_size == 0 else (len(data) // batch_size) + 1
    w2v.fit(data_iterator, n_epochs=args.n_epochs, steps_per_epoch=steps_per_epoch)
    # Save the word embeddings
    model_save_file = data_dir + "Models/python_model.npy"
    w2v.save_embeddings(model_save_file)
    # Save the model (so that can be used for training again)
    model_save_file = data_dir + "Models/python_model_file.pth"
    w2v.save_model(model_save_file)
