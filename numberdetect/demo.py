import pickle
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = data_dict['data']
labels = data_dict['labels']
print(labels)