import os

from lxml import objectify
import pandas as pd

# dev paths
dev_audio = './data/CoE_dataset/Dev_Set/audio_descriptors/'
dev_audio_avg = './data/CoE_dataset/Dev_Set/audio_avg/'
dev_text = './data/CoE_dataset/Dev_Set/text_descriptors/'
dev_vis = './data/CoE_dataset/Dev_Set/vis_descriptors/'
dev_xml = './data/CoE_dataset/Dev_Set/XML/'

# test paths
test_audio = './data/CoE_dataset/Test_Set/audio_descriptors/'
test_audio_avg = './data/CoE_dataset/Dev_Set/audio_avg/'
test_text = './data/CoE_dataset/Test_Set/text_descriptors/'
test_vis = './data/CoE_dataset/Test_Set/vis_descriptors/'
test_xml = './data/CoE_dataset/Test_Set/XML/'


def dir_contents(input_dir):
    """returns list of files contained in input dir"""

    files = os.listdir(input_dir)
    files.sort()

    # for f in files:
    #     instring = re.split("[.]", f)

    return files


def load_audio(fdir, fn, avg=False):
    df = pd.read_csv(fdir + fn, sep=',', header=None)
    df.set_index(0, inplace=True)
    if avg:
        df = df.fillna(0)
        df = df.mean(axis=1)
    return fn, df


def load_parse_xml(fdir, fn):
    # keys used as metadata in paper
    metakeys = ['language', 'year', 'genre', 'country', 'runtime', 'rated']
    ratingkeys = ['metascore', 'imdbRating', 'tomatoUserRating']  # 'tomatoRating',
    with open(fdir + fn, 'rb') as f:
        obj = objectify.fromstring(f.read())

    d = dict(obj['movie'].items())
    title = obj['movie'].attrib['title']
    meta = {key: d[key] for key in metakeys}
    rating = {key: d[key] for key in ratingkeys}
    return title, meta, rating


def load_save_all_audios(path=dev_audio):

    audios = dir_contents(path)
    names = []
    dfs = []
    for a in audios:
        name, df = load_audio(path, a, avg=True)

        # save averaged files
        save_path = './data/CoE_dataset/Test_Set/audio_avg/' + name
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path)


def load_all_avg_audios(path=dev_audio_avg):
    audios = dir_contents(path)
    names = []
    dfs = []
    for a in audios:
        name, df = load_audio(path, a)
        names.append(name)
        dfs.append(df.T)

    audio_df = pd.concat(dfs)

    # set row names
    audio_df.set_axis(names, axis=0, inplace=True)

    # set col names
    col_names = [f'audio_{i}' for i in range(audio_df.shape[1])]
    audio_df.set_axis(col_names, axis=1, inplace=True)
    return audio_df


def load_all_xmls():
    xmls = dir_contents(dev_xml)
    titles = []
    metadicts = []
    ratingsdicts = []
    for x in xmls:
        name, metadata, ratings = load_parse_xml(dev_xml, x)
        titles.append(name)
        metadicts.append(metadata)
        ratingsdicts.append(ratings)

    meta_df = pd.DataFrame(metadicts, index=titles)
    ratings_df = pd.DataFrame(ratingsdicts, index=titles)
    return meta_df, ratings_df


def load_all():
    audio_df = load_all_avg_audios()
    meta, rat = load_all_xmls()


def test():
    # load and average audio
    # audios = dir_contents(dev_audio)
    # name, df = load_audio(dev_audio, audios[0], avg=True)
    # print(df)

    # test loading first xml
    xmls = dir_contents(dev_xml)
    name, d1, d2 = load_parse_xml(dev_xml, xmls[0])
    print(d1)
    print(d2)


if __name__ == "__main__":
    # test()
    load_all()


# todo: import
# audio: import files, average each movie per line, use lines as columns/features
# text: do nothing for now
# visual: ??
# XML: parse attributes, filter attributes
