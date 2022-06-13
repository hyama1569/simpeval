import pickle
import argparse
from utils_neural_jacana import *

if __name__ == '__main__':
    with open('./src/wikiauto_sources.pickle', 'rb') as f:
        sources = pickle.load(f)
    with open('./src/wikiauto_targets.pickle', 'rb') as f:
        targets = pickle.load(f)

    parser = argparse.ArgumentParser()
    parser.add_argument("--batchsize", default=1, type=int)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--max_epoch", default=6, type=int)
    parser.add_argument("--max_span_size", default=1, type=int)
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--max_sent_length", default=70, type=int)
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--dataset", default='mtref', type=str)
    parser.add_argument("--sure_and_possible", default='True', type=str)
    parser.add_argument("--distance_embedding_size", default=128, type=int)
    parser.add_argument("--use_transition_layer", default='False', type=str, help='if False, will set transition score to 0.')
    parser.add_argument("batch_size", default=1, type=int)
    parser.add_argument("my_device", default='cuda', type=str)
    args = parser.parse_args(args=[])

    model = prepare_model(args)
    aligns = get_alignment(model, args, sources, targets)
    with open('./src/aligns_wikiaoto.pickle', 'wb') as f:
                pickle.dump(aligns, f)
