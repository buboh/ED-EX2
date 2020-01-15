import pandas as pd
import numpy as np
import arff
# Unzip CoE_dataset_offical_release.ip in 'working_directory/data/..' (I had to to save the xls as csv)

# Created test.csv by merging test_set_trailers.xls with dataset_complete.xlsx (tab test)
data_dev = pd.read_excel('data/CoE_dataset/Dev_Set/dev_set_groundtruth_and_trailers.xls').set_index('movie')
data_test = pd.read_csv('data/CoE_dataset/Test_Set/test_set_inclGoodforairplane.csv').set_index('movie')

meta_dev = pd.read_csv('data/csv_files/metadata_files/dev_data_meta.csv').set_index('Unnamed: 0')
meta_test = pd.read_csv('data/csv_files/metadata_files/test_data_meta.csv').set_index('Unnamed: 0')
meta_dev.columns = meta_dev.columns.str.replace(' ', '_')
meta_test.columns = meta_test.columns.str.replace(' ', '_')
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
    
col_not_in_test = list(set(meta_dev.columns) - set(meta_test.columns))
col_not_in_dev = list(set(meta_test.columns) - set(meta_dev.columns))

meta_test.drop((col_not_in_dev), axis = 1, inplace = True)

for i in col_not_in_test:
    meta_test[i] = 0

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


vis_dev = pd.concat([visual_dev, data_dev['goodforairplane']], axis=1, sort=True).reindex(visual_dev.index)
vis_dev_avg = pd.concat([visual_avg_dev, data_dev['goodforairplane']], axis=1, sort=True).reindex(visual_avg_dev.index)
vis_test = pd.concat([visual_test, data_test['goodforairplane']], axis=1, sort=True).reindex(visual_test.index)
vis_test_avg = pd.concat([visual_avg_test, data_test['goodforairplane']], axis=1, sort=True).reindex(visual_avg_test.index)


meta_dev_arff.sort_index(axis = 1, inplace = True)
meta_test_arff.sort_index(axis = 1, inplace = True)

meta_dev_arff
meta_test_arff

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


vis_dev.to_csv('data/csv_files/visual_files/dev_data_vis_flight.csv')

vis_dev_avg.to_csv('data/csv_files/visual_files/dev_data_vis_avg_flight.csv')

vis_test.to_csv('data/csv_files/visual_files/test_data_vis_flight.csv')

vis_test_avg.to_csv('data/csv_files/visual_files/test_data_vis_avg_flight.csv')


rating_dev_arff.to_csv('data/csv_files/user_rating_files/dev_data_rating_flight.csv')

rating_test_arff.to_csv('data/csv_files/user_rating_files/test_data_rating_flight.csv')
      
meta_dev_arff.to_csv('data/csv_files/metadata_files/dev_data_meta_flight.csv')

meta_test_arff.to_csv('data/csv_files/metadata_files/test_data_meta_flight.csv')

meta_rating_dev_arff.to_csv('data/csv_files/metadata_user_rating_files/dev_data_meta_rating_flight.csv')

meta_rating_test_arff.to_csv('data/csv_files/metadata_user_rating_files/test_data_meta_rating_flight.csv')

meta_visual_dev_arff.to_csv('data/csv_files/metadata_visual_files/dev_data_meta_visual_flight.csv')

meta_visual_test_arff.to_csv('data/csv_files/metadata_visual_files/test_data_meta_visual_flight.csv')

meta_visual_avg_dev_arff.to_csv('data/csv_files/metadata_visual_files/dev_data_meta_visual_avg_flight.csv')

meta_visual_avg_test_arff.to_csv('data/csv_files/metadata_visual_files/test_data_meta_visual_avg_flight.csv')