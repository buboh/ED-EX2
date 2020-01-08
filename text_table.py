import pandas as pd
import numpy as np
import arff

data_dev = pd.read_excel('data/CoE_dataset/Dev_Set/dev_set_groundtruth_and_trailers.xls', usecols =['movie', 'goodforairplane']).set_index('movie')
data_test = pd.read_csv('data/CoE_dataset/Test_Set/test_set_inclGoodforairplane.csv', usecols =['movie', 'goodforairplane']).set_index('movie')

text_dev = pd.read_csv('data/CoE_dataset/Dev_Set/text_descriptors/tdf_idf_dev.csv')

text_test = pd.read_csv('data/CoE_dataset/Test_Set/text_descriptors/tdf_idf_test.csv')

new_test_keyword_selection = list(set(text_dev.columns).intersection(set(text_test.columns)))

index_dev = list(text_dev.columns.values)
text_dev = text_dev.dropna(axis = 1)

text_dev.index = index_dev
text_dev = text_dev.transpose()
text_dev.index = data_dev.index.tolist()
temp = text_dev.columns.tolist()

#dev set contains a keyword with space, will be replaced below
for i in range(len(temp)):
    temp[i] = temp[i].replace(' ', '_')

text_dev.columns = temp
text_dev[text_dev != 0] = 1


index_test = list(text_test.columns.values)
text_test = text_test.dropna(axis = 1)

text_test.index = index_test
text_test = text_test.transpose()[new_test_keyword_selection]
text_test.index = data_test.index.tolist()
text_test[text_test != 0] = 1

missing_test_keywords = list(set(temp) - set(new_test_keyword_selection))

for keyword in missing_test_keywords:
    text_test[keyword] = np.zeros((223, 1))

text_dev = pd.concat([text_dev, data_dev['goodforairplane']], axis=1, sort=True).reindex(text_dev.index)
text_test = pd.concat([text_test, data_test['goodforairplane']], axis=1, sort=True).reindex(text_test.index)