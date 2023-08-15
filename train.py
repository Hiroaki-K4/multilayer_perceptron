import csv
import sys
import random
import numpy as np
from layers import Affine, Sigmoid, Softmax, BinaryCrossEntropy
from network import MultilayerPerceptron
from normalize import get_min_max, normalize_range


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

    min_list, max_list = get_min_max(np.array(feature_list))
    normed_arr = normalize_range(np.array(feature_list), min_list, max_list)
    normed_feature_list = normed_arr.tolist()

    if len(normed_feature_list) != len(label_list):
        raise RuntimeError("The length between feature list and label list is different")

    batch_size = 100
    hidden_layer_size = 50
    lr_rate = 1e-4

    input_layer = Affine(30, hidden_layer_size, lr_rate)
    sigmoid_layer_0 = Sigmoid()
    hidden_layer_0 = Affine(hidden_layer_size, hidden_layer_size, lr_rate)
    sigmoid_layer_1 = Sigmoid()
    hidden_layer_1 = Affine(hidden_layer_size, 2, lr_rate)
    softmax_layer = Softmax(batch_size, 2)
    loss_layer = BinaryCrossEntropy()
    layers = [input_layer, sigmoid_layer_0,  hidden_layer_0, sigmoid_layer_1, hidden_layer_1, softmax_layer]
    net = MultilayerPerceptron(layers, loss_layer)

    iters_num = 10000
    loss_thr = 1e-3
    for i in range(iters_num):
        random_idx = random.sample(range(len(normed_feature_list)), batch_size)
        batch_feature_list = []
        batch_label_list = []
        for i in range(len(normed_feature_list)):
            if i in random_idx:
                batch_feature_list.append(normed_feature_list[i])
                batch_label_list.append(label_list[i])

        if len(batch_feature_list) != len(batch_label_list) or len(batch_feature_list) != batch_size:
            raise RuntimeError("The length of batch data is wrong")

        feature_arr = np.array(batch_feature_list)
        label_arr = np.array(batch_label_list)

        loss = net.calculate_loss(feature_arr, label_arr, layers)
        if loss < loss_thr:
            break

        net.backward(label_arr, layers)

    net.save_parameters()

    print("Final Loss: ", loss)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Argument is wrong. Please pass the file path as an argument like `python3 train.py dataset/wdbc.csv`."
        )
        exit(1)

    main(sys.argv[1])
