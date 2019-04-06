from data_utils import load_data,dump_picle,load_data_pcm

if __name__ == '__main__':
    data_dir = ".\\data_gen\\"
    features, labels = load_data_pcm(data_dir,32000)
    dump_picle(features, labels)