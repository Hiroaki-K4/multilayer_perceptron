import csv
import sys
import random
import numpy as np
from layers import Affine, Sigmoid, Softmax, BinaryCrossEntropy
from network import MultilayerPerceptron


def main(data_path: str):
    with open(data_path) as f:
        reader = csv.reader(f, delimiter=",")
        input_data_list = [row for row in reader]
    feature_list = []
    label_list = []
    for data in input_data_list:
        feature_num = []
        for i in range(2, len(data)):
            feature_num.append(float(data[i]))
        feature_list.append(feature_num)

        if data[1] == "M":
            label_list.append(1)
        elif data[1] == "B":
            label_list.append(0)

    if len(feature_list) != len(label_list):
        raise RuntimeError("The length between feature list and label list is different")

    batch_size = 100
    random_idx = random.sample(range(len(feature_list)), batch_size)
    batch_feature_list = []
    batch_label_list = []
    for i in range(len(feature_list)):
        if i in random_idx:
            batch_feature_list.append(feature_list[i])
            batch_label_list.append(label_list[i])

    if len(batch_feature_list) != len(batch_label_list) or len(batch_feature_list) != batch_size:
        raise RuntimeError("The length of batch data is wrong")
    feature_arr = np.array(batch_feature_list)
    label_arr = np.array(batch_label_list)
    input_size = feature_arr.shape[1]
    hidden_layer_size = 50

    input_layer = Affine(input_size, hidden_layer_size)
    sigmoid_layer_0 = Sigmoid()
    hidden_layer_0 = Affine(hidden_layer_size, hidden_layer_size)
    sigmoid_layer_1 = Sigmoid()
    hidden_layer_1 = Affine(hidden_layer_size, 2)
    sigmoid_layer_2 = Sigmoid()
    softmax_layer = Softmax()
    loss_layer = BinaryCrossEntropy()
    layers = [input_layer, sigmoid_layer_0,  hidden_layer_0, sigmoid_layer_1, hidden_layer_1, sigmoid_layer_2, softmax_layer]
    net = MultilayerPerceptron(layers, loss_layer)

    input_arr = feature_arr

    net.predict(input_arr, layers)
    loss = net.calculate_loss(input_arr, label_arr, layers)
    print("loss: ", loss)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Argument is wrong. Please pass the file path as an argument like `python3 train.py dataset/wdbc.csv`."
        )
        exit(1)

    main(sys.argv[1])
