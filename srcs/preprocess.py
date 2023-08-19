import random

from normalize import get_norm_feature_list


def extract_input_data(input_data_list):
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

    return feature_list, label_list


def split_data_to_train_and_val(input_data_list, val_size):
    random_idx = random.sample(range(len(input_data_list)), val_size)
    train_data = []
    val_data = []
    for i in range(len(input_data_list)):
        if i in random_idx:
            val_data.append(input_data_list[i])
        else:
            train_data.append(input_data_list[i])

    train_feature_list, train_label_list = extract_input_data(train_data)
    val_feature_list, val_label_list = extract_input_data(val_data)

    normed_train_feature_list = get_norm_feature_list(train_feature_list)
    normed_val_feature_list = get_norm_feature_list(val_feature_list)

    return (
        normed_train_feature_list,
        train_label_list,
        normed_val_feature_list,
        val_label_list,
    )
