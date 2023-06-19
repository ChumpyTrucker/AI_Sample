from sklearn import tree

# Training data
# Format: [features, target]
training_data = [
    [1, 1, 1, 0, 'Yes'],
    [1, 0, 0, 1, 'No'],
    [0, 1, 0, 0, 'No'],
    [0, 1, 1, 1, 'Yes'],
    [0, 0, 1, 0, 'No'],
    [1, 0, 1, 1, 'Yes'],
]

# Features and target labels
features = [row[:-1] for row in training_data]
target = [row[-1] for row in training_data]

# Create and train the decision tree classifier
classifier = tree.DecisionTreeClassifier()
classifier.fit(features, target)

# Test data
test_data = [
    [1, 1, 0, 0],  # Expected: Yes
    [0, 1, 0, 1],  # Expected: No
]

# Predict the class labels for the test data
predictions = classifier.predict(test_data)

# Print the predictions
for i, prediction in enumerate(predictions):
    print("Test data", i+1, "predicted as:", prediction)