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
    np.random.seed(0)

    # train = DataSet("./Resources/digits_train.json")
    # test = DataSet("./Resources/digits_test.json")

    # train = DataSet("./Resources/heart_train.json")
    # test = DataSet("./Resources/heart_test.json")

    # train = DataSet("./Resources/mushrooms_train.json")
    # test = DataSet("./Resources/mushrooms_test.json")

    # train = DataSet("./Resources/winequality_train.json")
    # test = DataSet("./Resources/winequality_test.json")

    if (len(sys.argv)<5):
        print("Please pass 4 arguments. 1) # of Trees 2) Maximum Depth 3) Training File Path, 4) Testing File path ")
        sys.exit(1)

    Tree = int(sys.argv[1])
    max_depth = int(sys.argv[2])
    train_file = sys.argv[3]
    test_file = sys.argv[4]
    train = DataSet(train_file)
    test = DataSet(test_file)

    random_indices = []
    dt_predictions = []
    for t in range(1,Tree+1):
        random_indices.append(np.random.choice(train.shape[0], train.shape[0]))
        resampled_train_dataset = np.take(train.dataset, np.array(random_indices[-1]).T, axis=0)
        dt_on_train_dataset = dt.DecisionTree()
        dt_on_train_dataset.fit(resampled_train_dataset[:,:-1], resampled_train_dataset[:,-1], train.features, max_depth=max_depth)
        dt_predictions.append(dt_on_train_dataset.predict(test.dataset[:,:-1], prob=True))

    resampled_dataset_indices = np.array(random_indices).T
    for row in resampled_dataset_indices:
        for i in range(len(row)-1):
            print(row[i], end=",")
        print(row[-1])
    print()

    num_of_corrects = 0
    for row_index in range(test.shape[0]):
        dt_combined_predictions = []
        for p_id in range(len(dt_predictions)):
            print(train.labels[np.argmax(dt_predictions[p_id][row_index])], end=",")
            dt_combined_predictions.append(dt_predictions[p_id][row_index])
        combined_prediction = train.labels[np.argmax(np.sum(np.array(dt_combined_predictions), axis=0))]
        test_label = test.dataset[row_index, -1]
        print("{0},{1}".format(combined_prediction, test_label))
        if (combined_prediction == test_label.astype(type(combined_prediction))):
            num_of_corrects +=1

    print("\n{0:.12f}".format(num_of_corrects/test.shape[0]))