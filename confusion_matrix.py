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

    def bagged_tree_predictions(self, Tree, max_depth, test_obj):
        random_indices = []
        dt_predictions = []
        combined_prediction = []
        for t in range(1, Tree + 1):
            random_indices.append(np.random.choice(self.shape[0], self.shape[0]))
            resampled_train_dataset = np.take(self.dataset, np.array(random_indices[-1]).T, axis=0)
            dt_on_train_dataset = dt.DecisionTree()
            dt_on_train_dataset.fit(resampled_train_dataset[:, :-1], resampled_train_dataset[:, -1], self.features, max_depth=max_depth)
            dt_predictions.append(dt_on_train_dataset.predict(test_obj.dataset[:, :-1], prob=True))

        for row_index in range(test_obj.shape[0]):
            dt_combined_predictions = []
            for p_id in range(len(dt_predictions)):
                dt_combined_predictions.append(dt_predictions[p_id][row_index])
            combined_prediction.append(np.argmax(np.sum(np.array(dt_combined_predictions), axis=0)))
        return combined_prediction

    def boosted_tree_prediction(self, Tree, max_depth, test_obj):
        n_instances, n_features = self.shape
        instance_weights = np.full(n_instances, (1 / n_instances))
        normalized_weights = instance_weights / np.sum(instance_weights)

        training_instance_weights = []
        training_set_labels = self.dataset[:, -1]

        test_dataset_predictions = []
        test_dataset_predictions_combined = []
        for t in range(Tree):
            training_instance_weights.append(normalized_weights)
            dt_on_train_dataset = dt.DecisionTree()
            dt_on_train_dataset.fit(self.dataset[:, :-1], self.dataset[:, -1], self.features, max_depth=max_depth, instance_weights=normalized_weights)
            dt_prediction_matrix_train = dt_on_train_dataset.predict(self.dataset[:, :-1], prob=True)
            dt_prediction_matrix_test = dt_on_train_dataset.predict(test_obj.dataset[:, :-1], prob=True)

            predictions_on_train_dataset = []
            for i in range(self.shape[0]):
                predictions_on_train_dataset.append(self.labels[np.argmax(dt_prediction_matrix_train[i])])
            predictions_on_train_dataset = np.array(predictions_on_train_dataset)

            weighted_error = np.sum((predictions_on_train_dataset != training_set_labels).astype(int) * normalized_weights)
            if np.any(weighted_error >= 1 - (1 / len(self.labels))):
                break
            alpha = np.log((1 - weighted_error) / weighted_error) + np.log(len(self.labels) - 1)
            instance_weights = np.nan_to_num(
                normalized_weights * np.exp(alpha * (predictions_on_train_dataset != training_set_labels).astype(int)))
            normalized_weights = instance_weights / np.sum(instance_weights)

            predictions_on_test_dataset = []
            for i in range(test_obj.shape[0]):
                predictions_on_test_dataset.append(self.labels[np.argmax(dt_prediction_matrix_test[i])])
            predictions_on_test_dataset = np.array(predictions_on_test_dataset)
            test_dataset_predictions.append(predictions_on_test_dataset)
            for i in range(test_obj.shape[0]):
                dt_prediction_matrix_test[i, :] = 0
                dt_prediction_matrix_test[i, self.labels.index(predictions_on_test_dataset[i])] = alpha
            test_dataset_predictions_combined.append(dt_prediction_matrix_test)

        combined_predictions_matrix = np.average(test_dataset_predictions_combined, axis=0)
        predictions = []
        for i in range(test_obj.shape[0]):
            predictions.append(np.argmax(combined_predictions_matrix[i]))

        return predictions

    def get_label_class_indices(self, dataset):
        label_class_indices = []
        for label in dataset[:, -1]:
            label_class_indices.append(self.labels.index(label))
        return label_class_indices

    def print_confusion_matrix(self, actual_class, predicted_class):
        for index in range(test.shape[0]):
            confusion_matrix[predicted_class[index], actual_class[index]] += 1
        for row in confusion_matrix:
            for index in range(len(row)-1):
                print(row[index], end=",")
            print(row[-1])

if __name__ == '__main__':
    np.random.seed(0)

    train = DataSet("./Resources/digits_train.json")
    test = DataSet("./Resources/digits_test.json")

    # train = DataSet("./Resources/heart_train.json")
    # test = DataSet("./Resources/heart_test.json")

    # train = DataSet("./Resources/mushrooms_train.json")
    # test = DataSet("./Resources/mushrooms_test.json")

    # train = DataSet("./Resources/winequality_train.json")
    # test = DataSet("./Resources/winequality_test.json")

    actual_class = train.get_label_class_indices(test.dataset)
    confusion_matrix = np.zeros((len(train.labels), len(train.labels)), dtype=object)
    type = "boost"
    Tree = 5
    max_depth = 2
    if (type == "bag"):
        predicted_class = train.bagged_tree_predictions(Tree, max_depth, test)
        train.print_confusion_matrix(actual_class, predicted_class)
    else:
        predicted_class = train.boosted_tree_prediction(Tree, max_depth, test)
        train.print_confusion_matrix(actual_class, predicted_class)

    #print(Counter(predictions))