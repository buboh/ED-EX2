import pandas as pd
import numpy as np
import arff
# Unzip CoE_dataset_offical_release.ip in 'working_directory/data/..' (I had to to save the xls as csv)

# Created test.csv by merging test_set_trailers.xls with dataset_complete.xlsx (tab test)
data_dev = pd.read_excel('data/CoE_dataset/Dev_Set/dev_set_groundtruth_and_trailers.xls').set_index('movie')
data_test = pd.read_csv('data/CoE_dataset/Test_Set/test_set_inclGoodforairplane.csv').set_index('movie')



meta_dev = pd.read_csv('data/csv_files/metadata_files/dev_data_meta.csv').set_index('Unnamed: 0')
meta_test = pd.read_csv('data/csv_files/metadata_files/test_data_meta.csv').set_index('Unnamed: 0')
rating_dev = pd.read_csv('data/csv_files/user_rating_files/dev_data_rating.csv').set_index('Unnamed: 0')
rating_test = pd.read_csv('data/csv_files/user_rating_files/test_data_rating.csv').set_index('Unnamed: 0')

visual_dev = pd.read_csv('data/csv_files/visual_files/dev_data_visual.csv').set_index('movie')
visual_test = pd.read_csv('data/csv_files/visual_files/test_data_visual.csv').set_index('movie')

visual_avg_dev = pd.read_csv('data/csv_files/visual_files/dev_data_visual_avg.csv').set_index('movie')
visual_avg_test = pd.read_csv('data/csv_files/visual_files/test_data_visual_avg.csv').set_index('movie')

x = sorted(list(set(data_dev.index.tolist()) - set(meta_dev.index.tolist())))
y = sorted(list(set(meta_dev.index.tolist()) - set(data_dev.index.tolist())))

for i in range(len(x)):
    meta_dev = meta_dev.rename(index={y[i]: x[i]})
    rating_dev = rating_dev.rename(index={y[i]: x[i]})
    
x = sorted(list(set(data_test.index.tolist()) - set(meta_test.index.tolist())))
y = sorted(list(set(meta_test.index.tolist()) - set(data_test.index.tolist())))

for i in range(len(x)):
    meta_test = meta_test.rename(index={y[i]: x[i]})
    rating_test = rating_test.rename(index={y[i]: x[i]})

rating_dev_arff = pd.concat([rating_dev, data_dev['goodforairplane']], axis=1, sort=True).reindex(rating_dev.index)
rating_test_arff = pd.concat([rating_test, data_test['goodforairplane']], axis=1, sort=True).reindex(rating_test.index)    

meta_dev_arff = pd.concat([meta_dev, data_dev['goodforairplane']], axis=1, sort=True).reindex(meta_dev.index)
meta_test_arff = pd.concat([meta_test, data_test['goodforairplane']], axis=1, sort=True).reindex(meta_test.index)

meta_visual_dev_arff = pd.concat([meta_dev_arff, visual_dev], axis=1, sort=True).reindex(meta_dev_arff.index)
meta_visual_test_arff = pd.concat([meta_test_arff, visual_test], axis=1, sort=True).reindex(meta_test_arff.index)

meta_visual_avg_dev_arff = pd.concat([meta_dev_arff, visual_avg_dev], axis=1, sort=True).reindex(meta_dev_arff.index)
meta_visual_avg_test_arff = pd.concat([meta_test_arff, visual_avg_test], axis=1, sort=True).reindex(meta_test_arff.index)

meta_rating_dev_arff = pd.concat([meta_dev_arff, rating_dev], axis=1, sort=True).reindex(meta_dev_arff.index)
meta_rating_test_arff = pd.concat([meta_test_arff, rating_test], axis=1, sort=True).reindex(meta_test_arff.index)


arff.dump('data/WEKA_files/user_rating_files/dev_data_rating.arff', rating_dev_arff.values
      , relation='rating'
      , names=rating_dev_arff.columns)

arff.dump('data/WEKA_files/user_rating_files/test_data_rating.arff', rating_test_arff.values
      , relation='rating'
      , names=rating_test_arff.columns)
      
arff.dump('data/WEKA_files/metadata_files/dev_data_meta.arff', meta_dev_arff.values
      , relation='metadata'
      , names=meta_dev_arff.columns)

arff.dump('data/WEKA_files/metadata_files/test_data_meta.arff', meta_test_arff.values
      , relation='metadata'
      , names=meta_test_arff.columns)

arff.dump('data/WEKA_files/metadata_user_rating_files/dev_data_meta_rating.arff', meta_rating_dev_arff.values
      , relation='metadata_rating'
      , names=meta_rating_dev_arff.columns)

arff.dump('data/WEKA_files/metadata_user_rating_files/test_data_meta_rating.arff', meta_rating_test_arff.values
      , relation='metadata_rating'
      , names=meta_rating_test_arff.columns)

arff.dump('data/WEKA_files/metadata_visual_files/dev_data_meta_visual.arff', meta_visual_dev_arff.values
      , relation='metadata_visual'
      , names=meta_visual_dev_arff.columns)

arff.dump('data/WEKA_files/metadata_visual_files/test_data_meta_visual.arff', meta_visual_test_arff.values
      , relation='metadata_visual'
      , names=meta_visual_test_arff.columns)

arff.dump('data/WEKA_files/metadata_visual_files/dev_data_meta_visual_avg.arff', meta_visual_avg_dev_arff.values
      , relation='metadata_visual'
      , names=meta_visual_avg_dev_arff.columns)

arff.dump('data/WEKA_files/metadata_visual_files/test_data_meta_visual_avg.arff', meta_visual_avg_test_arff.values
      , relation='metadata_visual'
      , names=meta_visual_avg_test_arff.columns)