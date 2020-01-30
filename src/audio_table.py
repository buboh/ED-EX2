import pandas as pd
import numpy as np
import arff


#load average audio data and merge with ‚goodforairplane‘ attribute

data_dev = pd.read_excel('data/CoE_dataset/Dev_Set/dev_set_groundtruth_and_trailers.xls').set_index('movie')
data_test = pd.read_csv('data/CoE_dataset/Test_Set/test_set_inclGoodforairplane.csv').set_index('movie')

indices = ['movie','MFCC_1', 'MFCC_2', 'MFCC_3', 'MFCC_4', 'MFCC_5',
           'MFCC_6', 'MFCC_7', 'MFCC_8', 'MFCC_9', 'MFCC_10', 'MFCC_11', 'MFCC_12', 'MFCC_13']

def audio_selection(file_name, set_selection):
    try:
        if set_selection == 'test':
            my_data = np.genfromtxt('data/CoE_dataset/Test_Set/audio_avg/'+ file_name + '.csv', delimiter=',')
        elif set_selection == 'dev':
            my_data = np.genfromtxt('data/CoE_dataset/Dev_Set/audio_avg/'+ file_name + '.csv', delimiter=',')
        else: 
            return np.full((13), np.nan)
    except:
        return np.full((13), np.nan)
    return my_data[:,1]

#function below loads audio data and merges with attribute 'goodforairplane'
#set_selection has to be 'dev' or 'test'

def audio_data(df, set_selection):
    titles = df.index.tolist()
    split_data = pd.DataFrame(columns = indices).set_index('movie')
    for title in titles:
        split_data.loc[title] = audio_selection(df.loc[title]['filename'], set_selection)
    split_data = split_data.dropna()
    
    if set_selection == 'test':
        return(pd.concat([split_data, data_test['goodforairplane']], axis=1, sort=True).reindex(split_data.index))
    
    if set_selection == 'dev':
        return(pd.concat([split_data, data_dev['goodforairplane']], axis=1, sort=True).reindex(split_data.index))
    
    return(split_data)

# test data import
dev_data_audio = audio_data(data_dev, 'dev')

# dev data import
test_data_audio = audio_data(data_test, 'test')

dev_data_audio.to_csv('data/csv_files/audio_files/dev_data_audio.csv')
test_data_audio.to_csv('data/csv_files/audio_files/test_data_audio.csv')
