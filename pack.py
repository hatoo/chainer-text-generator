import glob
import pickle
import argparse


def load_texts(path):
    texts = []
    for filepath in glob.glob(path + "/*.txt"):
        with open(filepath, 'r', encoding='utf-8') as f:
            texts.append(f.read())
    return texts


def gen_dict(texts):
    texts = ''.join(texts)
    chars = list(set(texts))
    chars = sorted(chars)
    # 0 is terminate signal
    i_to_c = ['#'] + chars
    c_to_i = {}
    for i, c in enumerate(chars):
        c_to_i[c] = i + 1
    return (i_to_c, c_to_i)


def gen_vec(texts, c_to_i):
    # 0 is Terminate signal
    return [[c_to_i[c] for c in list(x)] + [0] for x in texts]


def main(path, out):
    # Load all texts
    print('Loading texts ...')
    texts = load_texts(path)
    # Generate Dictionary
    print('Create dictionary ...')
    i_to_c, c_to_i = gen_dict(texts)
    # Generate Vector
    print('Vectorize texts ...')
    data = gen_vec(texts, c_to_i)
    # Save
    with open(out, 'wb') as f:
        print('Saving ...')
        pickle.dump((data, i_to_c, c_to_i), f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('target')
    parser.add_argument('--out', '-o', default='out.pickle',
                        help='Path to output the packed data')
    args = parser.parse_args()
    main(args.target, args.out)
