from data_utils import load_data,dump_picle

if __name__ == '__main__':
    data_dir = ".\\data\\"
    features, labels = load_data(data_dir,16000)
    dump_picle(features, labels)