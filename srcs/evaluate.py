import numpy as np


def decide_class_by_probability(pred_res):
    class_list = []
    for pred in pred_res:
        if pred[0] > pred[1]:
            class_list.append(1)
        else:
            class_list.append(0)

    return class_list


def compare_result_and_label(class_list, batch_label_list, fill_num):
    true_num = 0
    false_num = 0
    for i in range(len(class_list) - fill_num):
        if class_list[i] == batch_label_list[i]:
            true_num += 1
        else:
            false_num += 1

    return true_num, false_num


def calculate_accuracy(net, layers, batch_size, normed_feature_list, label_list):
    all_true_num = 0
    all_false_num = 0
    while len(normed_feature_list) > 0:
        if len(normed_feature_list) != len(label_list):
            raise RuntimeError(
                "[calculate_accuracy] The length between feature list and label list is different"
            )
        fill_num = 0
        if len(normed_feature_list) >= batch_size:
            batch_feature_list = normed_feature_list[:batch_size]
            batch_label_list = label_list[:batch_size]
            del normed_feature_list[:batch_size]
            del label_list[:batch_size]
        else:
            batch_feature_list = normed_feature_list
            batch_label_list = label_list
            fill_num = batch_size - len(batch_feature_list)
            normed_feature_list = []
            label_list = []
            for i in range(batch_size - len(batch_feature_list)):
                batch_feature_list.append(batch_feature_list[0])
                batch_label_list.append(batch_label_list[0])

        pred_res = net.predict(np.array(batch_feature_list), layers, False)
        class_list = decide_class_by_probability(pred_res)
        true_num, false_num = compare_result_and_label(
            class_list, batch_label_list, fill_num
        )
        all_true_num += true_num
        all_false_num += false_num

    accuracy = all_true_num / (all_true_num + all_false_num)

    return accuracy
