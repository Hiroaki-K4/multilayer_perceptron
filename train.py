import csv
import sys


def main(data_path: str):
    with open(data_path) as f:
        reader = csv.reader(f, delimiter=",")
        input_data_list = [row for row in reader]
    print(input_data_list)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Argument is wrong. Please pass the file path as an argument like `python3 train.py dataset/wdbc.csv`."
        )
        exit(1)

    main(sys.argv[1])
