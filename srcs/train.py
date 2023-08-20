import argparse
import csv
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
from evaluate import calculate_accuracy
from layers import Affine, BinaryCrossEntropy, Sigmoid, Softmax
from network import MultilayerPerceptron
from normalize import get_min_max, normalize_range
from preprocess import split_data_to_train_and_val


def draw_learning_history(learning_history):
    epoch = []
    train_loss = []
    val_loss = []
    for key in learning_history:
        epoch.append(key)
        train_loss.append(learning_history[key]["train_loss"])
        val_loss.append(learning_history[key]["val_loss"])

    plt.plot(epoch, train_loss, label="Training loss")
    plt.plot(epoch, val_loss, label="Validation loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.title("Learning curves for a multilayer perceptron")
    plt.legend()
    plt.show()


def main(data_path: str, output_param_path: str):
    with open(data_path) as f:
        reader = csv.reader(f, delimiter=",")
        input_data_list = [row for row in reader]

    batch_size = 100
    (
        normed_train_feature_list,
        train_label_list,
        normed_val_feature_list,
        val_label_list,
    ) = split_data_to_train_and_val(input_data_list, batch_size)

    if len(normed_train_feature_list) != len(train_label_list) or len(
        normed_val_feature_list
    ) != len(val_label_list):
        raise RuntimeError(
            "The length between feature list and label list is different"
        )

    hidden_layer_size = 50
    lr_rate = 1e-4

    input_layer = Affine(30, hidden_layer_size, lr_rate)
    sigmoid_layer_0 = Sigmoid()
    hidden_layer_0 = Affine(hidden_layer_size, hidden_layer_size, lr_rate)
    sigmoid_layer_1 = Sigmoid()
    hidden_layer_1 = Affine(hidden_layer_size, 2, lr_rate)
    softmax_layer = Softmax(batch_size, 2)
    loss_layer = BinaryCrossEntropy()
    layers = [
        input_layer,
        sigmoid_layer_0,
        hidden_layer_0,
        sigmoid_layer_1,
        hidden_layer_1,
        softmax_layer,
    ]
    net = MultilayerPerceptron(layers, loss_layer, batch_size)

    iters_num = 15000
    epoch_cnt = 0
    epoch_num = int(len(normed_train_feature_list) / batch_size + 1)
    epoch_all = int(iters_num / epoch_num)
    loss_thr = 1e-3
    learning_history = {}
    for itr in range(iters_num):
        random_idx = random.sample(range(len(normed_train_feature_list)), batch_size)
        batch_feature_list = []
        batch_label_list = []
        for i in range(len(normed_train_feature_list)):
            if i in random_idx:
                batch_feature_list.append(normed_train_feature_list[i])
                batch_label_list.append(train_label_list[i])

        if (
            len(batch_feature_list) != len(batch_label_list)
            or len(batch_feature_list) != batch_size
        ):
            raise RuntimeError("The length of batch data is wrong")

        feature_arr = np.array(batch_feature_list)
        label_arr = np.array(batch_label_list)

        loss = net.calculate_loss(feature_arr, label_arr, layers, True)
        if loss < loss_thr:
            break

        net.backward(label_arr, layers)

        if itr % epoch_num == 0:
            epoch_cnt += 1
            val_loss = net.calculate_loss(
                np.array(normed_val_feature_list),
                np.array(val_label_list),
                layers,
                False,
            )
            learning_history[epoch_cnt] = {"train_loss": loss, "val_loss": val_loss}
            print(
                "Epoch: {0}/{1} Loss: {2} Val Loss: {3}".format(
                    epoch_cnt, epoch_all, round(loss, 4), round(val_loss, 4)
                )
            )

    draw_learning_history(learning_history)

    net.save_parameters(output_param_path)
    train_acc = calculate_accuracy(
        net, layers, batch_size, normed_train_feature_list, train_label_list
    )
    val_acc = calculate_accuracy(
        net, layers, batch_size, normed_val_feature_list, val_label_list
    )
    print()
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Final Loss: ", loss)
    print("Train Accuracy: {0}%".format(round(train_acc * 100, 1)))
    print("Val Accuracy: {0}%".format(round(val_acc * 100, 1)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_path")
    parser.add_argument("--output_param_path")
    args = parser.parse_args()

    random.seed(10)
    np.random.seed(10)
    main(args.train_data_path, args.output_param_path)
