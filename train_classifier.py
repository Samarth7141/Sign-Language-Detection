import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load the data
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Inspect the data to understand its structure
data = data_dict['data']
labels = data_dict['labels']

# Print the first few elements to see the structure
print(f'First few elements of data: {data[:3]}')
print(f'First few elements of labels: {labels[:3]}')

# Ensure that all elements in data are of the same length
# Find the maximum length of the data points
max_length = max(len(d) for d in data)

# Pad the sequences to have the same length
padded_data = np.array([np.pad(d, (0, max_length - len(d)), 'constant') for d in data])

# Convert labels to a NumPy array
labels = np.asarray(labels)

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(padded_data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Create and train the model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Make predictions
y_predict = model.predict(x_test)

# Calculate the accuracy
score = accuracy_score(y_test, y_predict)

print(f'{score * 100}% of samples were classified correctly!')

# Save the model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
