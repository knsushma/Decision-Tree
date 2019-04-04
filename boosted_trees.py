import numpy as np
import json
import DecisionTree as dt


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

    train = DataSet("./Resources/digits_train.json")
    test = DataSet("./Resources/digits_test.json")

    # train = DataSet("./Resources/heart_train.json")
    # test = DataSet("./Resources/heart_test.json")

    # train = DataSet("./Resources/mushrooms_train.json")
    # test = DataSet("./Resources/mushrooms_test.json")

    # train = DataSet("./Resources/winequality_train.json")
    # test = DataSet("./Resources/winequality_test.json")

    Tree = 2
    max_depth = 3
    n_instances, n_features = train.shape


    instance_weights = np.full(n_instances, (1/n_instances))
    normalized_weights = instance_weights / np.sum(instance_weights)
    #training_instance_weights.append(normalized_weights)

    training_instance_weights = []
    labels_class = train.dataset[:, -1]
    tree_weights = []
    test_dataset_predictions = []
    for t in range(Tree):

        training_instance_weights.append(normalized_weights)
        dt_on_train_dataset = dt.DecisionTree()
        dt_on_train_dataset.fit(train.dataset[:,:-1], train.dataset[:,-1], train.features, max_depth=max_depth, instance_weights=normalized_weights)
        predictions_on_train_dataset = dt_on_train_dataset.predict(train.dataset[:,:-1], prob=False)
        predictions_on_test_dataset = dt_on_train_dataset.predict(test.dataset[:,:-1], prob=False)

        # train_dataset_predictions = []
        # for i in range(train.shape[0]):
        #     train_dataset_predictions.append(train.labels[np.argmax(dt_prediction_matrix_train[i])])

        test_dataset_predictions.append(predictions_on_test_dataset)

        weighted_error = np.sum((predictions_on_train_dataset != labels_class).astype(int) * normalized_weights)
        if np.any(weighted_error >= 1 - (1 / len(train.labels))):
            break

        alpha = np.log((1 - weighted_error) / weighted_error) + np.log(len(train.labels) - 1)
        tree_weights.append(alpha)

        instance_weights = np.nan_to_num(normalized_weights * np.exp(alpha * (predictions_on_train_dataset != labels_class).astype(int)))
        normalized_weights = instance_weights / np.sum(instance_weights)

        print()

    print(np.array(training_instance_weights).T)
    for weights in np.array(training_instance_weights).T:
        for index in range(len(weights)-1):
            print("{0:.12f}".format(weights[index]), end=",")
        print("{0:.12f}".format(weights[-1]))

    print()
    for index in range(len(tree_weights)-1):
        print("{0:.12f}".format(tree_weights[index]), end=",")
    print("{0:.12f}".format(tree_weights[-1]))

    print()
    num_of_corrects = 0
    for row_index in range(test.shape[0]):
        for p_id in range(len(test_dataset_predictions)):
            print(test_dataset_predictions[p_id][row_index], end=",")
        test_label = test.dataset[row_index, -1]
        print("{0}".format(test_label))
        # if (combined_prediction == test_label.astype(type(combined_prediction))):
        #     num_of_corrects += 1