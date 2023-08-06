import csv
import math
import sys

from tabulate import tabulate


def calculate_std(input_data_list, col_num, mean, count):
    diff_sum = 0
    for data in input_data_list:
        if data[col_num] != "":
            value = float(data[col_num])
            diff_sum += (value - mean) ** 2

    return math.sqrt(diff_sum / (count - 1))


def calculate_threshold_by_ratio(input_data_list, col_num, count, ratio):
    value_list = []
    for data in input_data_list:
        if data[col_num] != "":
            value = float(data[col_num])
            value_list.append(value)

    threshold_index = count * ratio
    value_list.sort()

    return value_list[int(threshold_index)]


def display_result(
    numerical_feature_list,
    count_list,
    mean_list,
    std_list,
    min_list,
    max_list,
    threshold_25_list,
    threshold_50_list,
    threshold_75_list,
):
    count_list.insert(0, "Count")
    mean_list.insert(0, "Mean")
    std_list.insert(0, "Std")
    min_list.insert(0, "Min")
    threshold_25_list.insert(0, "25%")
    threshold_50_list.insert(0, "50%")
    threshold_75_list.insert(0, "75%")
    max_list.insert(0, "Max")
    data = [
        count_list,
        mean_list,
        std_list,
        min_list,
        threshold_25_list,
        threshold_50_list,
        threshold_75_list,
        max_list,
    ]

    print(tabulate(data, headers=numerical_feature_list))


def main(data_path: str):
    with open(data_path) as f:
        reader = csv.reader(f, delimiter=",")
        input_data_list = [row for row in reader]

    feature_list = [
        "radius",
        "texture",
        "perimeter",
        "area",
        "smoothness",
        "compactness",
        "concavity",
        "concavePoints",
        "symmetry",
        "fractalDimension",
    ]
    feat_type = ["Mean", "Standard error", "Largest"]

    for idx in range(len(feat_type)):
        feature_start_pos = 2
        count_list = []
        mean_list = []
        std_list = []
        min_list = []
        max_list = []
        threshold_25_list = []
        threshold_50_list = []
        threshold_75_list = []
        for i in range(len(feature_list) * len(feat_type)):
            if i % len(feat_type) != idx:
                continue
            col_num = i + feature_start_pos
            count = 0
            data_sum = 0.0
            min_value = sys.float_info.max
            max_value = sys.float_info.min
            for data in input_data_list:
                if data[col_num] != "":
                    value = float(data[col_num])
                    if value < min_value:
                        min_value = value
                    if value > max_value:
                        max_value = value
                    data_sum += value
                    count += 1

            mean = data_sum / count
            std = calculate_std(input_data_list, col_num, mean, count)
            threshold_25 = calculate_threshold_by_ratio(
                input_data_list, col_num, count, 0.25
            )
            threshold_50 = calculate_threshold_by_ratio(
                input_data_list, col_num, count, 0.5
            )
            threshold_75 = calculate_threshold_by_ratio(
                input_data_list, col_num, count, 0.75
            )

            count_list.append(count)
            mean_list.append(mean)
            std_list.append(std)
            min_list.append(min_value)
            max_list.append(max_value)
            threshold_25_list.append(threshold_25)
            threshold_50_list.append(threshold_50)
            threshold_75_list.append(threshold_75)

        print("FEATURE TYPE: ", feat_type[idx])
        display_result(
            feature_list,
            count_list,
            mean_list,
            std_list,
            min_list,
            max_list,
            threshold_25_list,
            threshold_50_list,
            threshold_75_list,
        )
        print()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Argument is wrong. Please pass the file path as an argument like `python3 describe.py dataset/wdbc.csv`."
        )
        exit(1)

    main(sys.argv[1])
