from data_utils import load_data,dump_picle

if __name__ == '__main__':
    data_dir = ".\\data_gen\\"
    features, labels = load_data(data_dir)
    dump_picle(features, labels)