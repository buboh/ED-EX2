import os

from lxml import objectify
import numpy as np
import pandas as pd

from sklearn.preprocessing import MultiLabelBinarizer

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
    # todo: deal with categorical values (one-hot encoding)
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


def load_all_xmls(which=dev_xml): # dev_xml or test_xml
    xmls = dir_contents(which)
    titles = []
    metadicts = []
    ratingsdicts = []
    for x in xmls:
        name, metadata, ratings = load_parse_xml(which, x)
        titles.append(name)
        metadicts.append(metadata)
        ratingsdicts.append(ratings)

    # metadata
    meta_df = pd.DataFrame(metadicts, index=titles)
    print(meta_df.iloc[95:105, :])

    # multi-hot encoding for metadata
    tfd = {'country': None, 'genre': None, 'language': None, 'rated': None}
    for column in ['country', 'genre', 'language', 'rated']:
        # split string into list of labels
        col = meta_df.loc[:, column].str.split(', ')

        # convert lists of labels to multi-hot-encoding
        mlb = MultiLabelBinarizer()
        tf = mlb.fit_transform(col)
        tfd[column] = pd.DataFrame(tf, index=meta_df.index, columns=mlb.classes_)

    runtime = meta_df.loc[:, 'runtime'].str.strip(' min')
    runtime.replace('N/A', 0, inplace=True)
    # runtime.fillna(0, inplace=True)
    runtime = runtime.astype('int')

    year = meta_df.loc[:, 'year'].astype('int')

    # concat transformed columns
    meta_df_tf = pd.concat([elem for elem in tfd.values()] + [runtime, year], axis=1)

    # ratings
    ratings_df = pd.DataFrame(ratingsdicts, index=titles)
    ratings_df.replace('N/A', np.nan, inplace=True)
    ratings_df = ratings_df.astype('float')

    # ugly, but apply wouldn't work for some reason
    ratings_df['imdbRating'].fillna((ratings_df['imdbRating'].mean()), inplace=True)
    ratings_df['metascore'].fillna((ratings_df['metascore'].mean()), inplace=True)
    ratings_df['tomatoUserRating'].fillna((ratings_df['tomatoUserRating'].mean()), inplace=True)
    # ratings_df.apply(lambda col: col.fillna(col.mean()), axis=0)

    return meta_df_tf, ratings_df


def load_all(save=True):
    audio_df = load_all_avg_audios()
    meta, rat = load_all_xmls(test_xml)

    # save averaged files
    if save:
        # meta_save_path = './data/csv_files/metadata_files/test_data_meta.csv'
        rating_save_path = './data/csv_files/user_rating_files/test_data_rating.csv'
        meta_save_path = './data/csv_files/metadata_files/dev_data_meta.csv'
        # rating_save_path = './data/csv_files/user_rating_files/dev_data_rating.csv'
        os.makedirs(os.path.dirname(meta_save_path), exist_ok=True)
        os.makedirs(os.path.dirname(rating_save_path), exist_ok=True)
        # meta.to_csv(meta_save_path)
        rat.to_csv(rating_save_path)

    print()


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

# todo: write exporter for feature combinations (to arff files)
