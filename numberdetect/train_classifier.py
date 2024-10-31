# import pickle
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score

# # Load data from pickle file
# data_dict = pickle.load(open('./data.pickle', 'rb'))
# data = data_dict['data']
# labels = data_dict['labels']

# # Determine the median length of sequences
# median_length = int(np.median([len(seq) for seq in data]))

# # Pad sequences with zeros to ensure they all have the median length
# def pad_sequence(seq, target_length):
#     return seq[:target_length] + [0] * (target_length - len(seq))

# data_padded = np.array([pad_sequence(seq, median_length) for seq in data])

# # Convert labels to a NumPy array
# labels = np.asarray(labels)

# # Split the dataset into training and testing sets
# x_train, x_test, y_train, y_test = train_test_split(data_padded, labels, test_size=0.2, shuffle=True, stratify=labels)

# # Initialize the RandomForestClassifier model
# model = RandomForestClassifier()

# # Train the model
# model.fit(x_train, y_train)


# y_predict = model.predict(x_test)

# score = accuracy_score(y_predict, y_test)

# print('{}% of samples were classified correctly!'.format(score * 100))

# with open('model.p', 'wb') as f:
#     pickle.dump({'model': model}, f)


import tensorflow as tf
import tensorflow_decision_forests as tfdf
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data from pickle file
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = data_dict['data']
labels = data_dict['labels']

# Determine the median length of sequences
median_length = int(np.median([len(seq) for seq in data]))

# Pad sequences with zeros to ensure they all have the median length
def pad_sequence(seq, target_length):
    return seq[:target_length] + [0] * (target_length - len(seq))

data_padded = np.array([pad_sequence(seq, median_length) for seq in data])

# Convert labels to a NumPy array
labels = np.asarray(labels)

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data_padded, labels, test_size=0.2, shuffle=True, stratify=labels)

# Convert the data to TensorFlow Datasets
def numpy_to_tf_dataset(features, labels):
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.batch(100)  # Adjust batch size if needed
    return dataset

train_ds = numpy_to_tf_dataset(x_train, y_train)
test_ds = numpy_to_tf_dataset(x_test, y_test)

# Initialize the TensorFlow Decision Forest RandomForestModel
model = tfdf.keras.RandomForestModel()

# Train the model
model.fit(train_ds)

# Evaluate the model
evaluation = model.evaluate(test_ds)
accuracy = evaluation['accuracy']

print(f'{accuracy * 100:.2f}% of samples were classified correctly!')

# Save the model using TensorFlow's SavedModel format
model.save('tfdf_model')

# If you want to save it in a pickle-like format:
with open('tfdf_model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
