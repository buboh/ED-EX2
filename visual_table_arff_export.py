import pandas as pd
import numpy as np
import arff
# Unzip CoE_dataset_offical_release.ip in 'working_directory/data/..' (I had to to save the xls as csv)

# Created test.csv by merging test_set_trailers.xls with dataset_complete.xlsx (tab test)
data_dev = pd.read_excel('data/CoE_dataset/Dev_Set/dev_set_groundtruth_and_trailers.xls').set_index('movie')
data_test = pd.read_csv('data/CoE_dataset/Test_Set/test_set_inclGoodforairplane.csv').set_index('movie')


# csv below was extraced from 'data/mediaeval/dataset_complete.xlsx' tab test
#data_test = pd.read_csv('data/CoE_dataset/test_set_trailers.csv').set_index('movie')



columns_avg = ['movie']
for i in range(826):
    columns_avg.append('visual_'+ str(i+1))
columns_all = columns_avg.copy()

for i in range(826):
    columns_all.append('visual_'+ str(i+827))

# in case we have to ommit NaN values for the mean -> np.nanmean()

def visual_mean(file_name, set_selection):
    try:
        if set_selection == 'test':
            my_data = np.genfromtxt('data/CoE_dataset/Test_Set/vis_descriptors/'+ file_name + '.csv', delimiter=',')
            vis = (my_data[0]+my_data[1])/2
        elif set_selection == 'dev':
            my_data = np.genfromtxt('data/CoE_dataset/Dev_Set/vis_descriptors/'+ file_name + '.csv', delimiter=',')
            vis = (my_data[0]+my_data[1])/2
        else: 
            vis = np.full((826), np.nan)
    except:
        vis = np.full((826), np.nan)
    return vis

def visual_all(file_name, set_selection):
    try:
        if set_selection == 'test':
            my_data = np.genfromtxt('data/CoE_dataset/Test_Set/vis_descriptors/'+ file_name + '.csv', delimiter=',')
            vis = np.concatenate((my_data[0], my_data[1]))
        elif set_selection == 'dev':
            my_data = np.genfromtxt('data/CoE_dataset/Dev_Set/vis_descriptors/'+ file_name + '.csv', delimiter=',')
            vis = np.concatenate((my_data[0], my_data[1]))
        else: 
            vis = np.full((826*2), np.nan)
    except:
        vis = np.full((826*2), np.nan)
    return vis

def visual_data(df, columns_, set_selection):
    titles = df.index.tolist()
    
    split_data = pd.DataFrame(columns = columns_).set_index('movie')
    
    if len(columns_) == 827:
        for title in titles:
            try:
                split_data.loc[title] = visual_mean(df.loc[title]['filename'], set_selection)
            except:
                continue
                
    if len(columns_) == 1653:
        for title in titles:
            try:
                split_data.loc[title] = visual_all(df.loc[title]['filename'], set_selection)
            except:
                continue
    split_data = split_data.dropna()
    return(split_data)

vis_dev = visual_data(data_dev, columns_all, 'dev')
vis_dev_avg = visual_data(data_dev, columns_avg, 'dev')

vis_test = visual_data(data_test, columns_all, 'test')
vis_test_avg = visual_data(data_test, columns_avg, 'test')


arff.dump('data/WEKA_files/visual_files/dev_data_vis.arff', vis_dev.values
      , relation='visual_descriptors'
      , names=vis_dev.columns)

arff.dump('data/WEKA_files/visual_files/dev_data_vis_avg.arff', vis_dev_avg.values
      , relation='visual_descriptors'
      , names=vis_dev_avg.columns)

arff.dump('data/WEKA_files/visual_files/test_data_vis.arff', vis_test.values
      , relation= 'visual_descriptors'
      , names=vis_test.columns)

arff.dump('data/WEKA_files/visual_files/test_data_vis_avg.arff', vis_test_avg.values
      , relation= 'visual_descriptors'
      , names=vis_test_avg.columns)

vis_dev.to_csv('data/csv_files/visual_files/dev_data_visual.csv')
vis_dev_avg.to_csv('data/csv_files/visual_files/dev_data_visual_avg.csv')

vis_test.to_csv('data/csv_files/visual_files/test_data_visual.csv')
vis_test_avg.to_csv('data/csv_files/visual_files/test_data_visual_avg.csv')
