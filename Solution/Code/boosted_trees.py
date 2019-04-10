import numpy as np
import json
import DecisionTree as dt
import sys

class DataSet:
    def __init__(self, input_file):
        self.input_file = input_file
        self.input_meta_dataset = self.load_file_and_metadata()
        self.metadata = np.array(self.input_meta_dataset["metadata"])
        self.features = np.array(self.input_meta_dataset["metadata"]["features"])
        self.labels = self.input_meta_dataset["metadata"]["features"][-1][1]
        self.dataset = np.array(self.input_meta_dataset["data"])
        self.shape = self.dataset.shape

    def load_file_and_metadata(self):
        file = open(self.input_file)
        try:
            return json.load(file)
        except ValueError:
            print("Decoding JSON has failed from json file: {}".format(self.input_file))
            exit(1)

if __name__ == '__main__':

    if (len(sys.argv)<5):
        print("Please pass 4 arguments. 1) # of Trees 2) Maximum Depth 3) Training File Path, 4) Testing File path ")
        sys.exit(1)

    Tree = int(sys.argv[1])
    max_depth = int(sys.argv[2])
    train_file = sys.argv[3]
    test_file = sys.argv[4]
    train = DataSet(train_file)
    test = DataSet(test_file)

    n_instances, n_features = train.shape


    instance_weights = np.full(n_instances, (1/n_instances))
    normalized_weights = instance_weights / np.sum(instance_weights)

    training_instance_weights = []
    train_labels = train.dataset[:, -1]
    test_labels = test.dataset[:, -1]
    tree_weights = []
    test_dataset_predictions = []
    test_dataset_predictions_combined = []
    for t in range(Tree):

        training_instance_weights.append(normalized_weights)
        dt_on_train_dataset = dt.DecisionTree()
        dt_on_train_dataset.fit(train.dataset[:,:-1], train.dataset[:,-1], train.features, max_depth=max_depth, instance_weights=normalized_weights)
        dt_prediction_matrix_train = dt_on_train_dataset.predict(train.dataset[:,:-1], prob=True)
        dt_prediction_matrix_test = dt_on_train_dataset.predict(test.dataset[:,:-1], prob=True)

        predictions_on_train_dataset = []
        for i in range(train.shape[0]):
            predictions_on_train_dataset.append(train.labels[np.argmax(dt_prediction_matrix_train[i])])
        predictions_on_train_dataset = np.array(predictions_on_train_dataset)

        weighted_error = np.sum((predictions_on_train_dataset != train_labels).astype(int) * normalized_weights)
        if np.any(weighted_error >= 1 - (1 / len(train.labels))):
            break
        alpha = np.log((1 - weighted_error) / weighted_error) + np.log(len(train.labels) - 1)
        tree_weights.append(alpha)
        instance_weights = np.nan_to_num(
            normalized_weights * np.exp(alpha * (predictions_on_train_dataset != train_labels).astype(int)))
        normalized_weights = instance_weights / np.sum(instance_weights)

        predictions_on_test_dataset = []
        for i in range(test.shape[0]):
            predictions_on_test_dataset.append(train.labels[np.argmax(dt_prediction_matrix_test[i])])
        predictions_on_test_dataset = np.array(predictions_on_test_dataset)
        test_dataset_predictions.append(predictions_on_test_dataset)
        for i in range(test.shape[0]):
            dt_prediction_matrix_test[i, :] = 0
            dt_prediction_matrix_test[i, train.labels.index(predictions_on_test_dataset[i])] = alpha
        test_dataset_predictions_combined.append(dt_prediction_matrix_test)



    for weights in np.array(training_instance_weights).T:
        for index in range(len(weights)-1):
            print("{0:.12f}".format(weights[index]), end=",")
        print("{0:.12f}".format(weights[-1]))

    print()
    for index in range(len(tree_weights)-1):
        print("{0:.12f}".format(tree_weights[index]), end=",")
    print("{0:.12f}".format(tree_weights[-1]))

    print()

    combined_prediction_matrix = np.average(test_dataset_predictions_combined, axis=0)
    predictions = []
    for i in range(test.shape[0]):
        predictions.append(train.labels[np.argmax(combined_prediction_matrix[i])])

    num_of_corrects = 0
    for row_index in range(test.shape[0]):
        for p_id in range(len(test_dataset_predictions)):
            print(test_dataset_predictions[p_id][row_index], end=",")
        test_label = test.dataset[row_index, -1]
        print("{0},{1}".format(predictions[row_index], test_label))
        if (predictions[row_index] == test_label.astype(type(predictions[row_index]))):
            num_of_corrects += 1

    print("\n{0:.12f}".format(num_of_corrects / test.shape[0]))