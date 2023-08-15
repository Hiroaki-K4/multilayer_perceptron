import argparse
import csv

def main(test_data_path: str, param_path: str):
    with open(test_data_path) as f:
        reader = csv.reader(f, delimiter=",")
        input_data_list = [row for row in reader]
    print(len(input_data_list))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data_path")
    parser.add_argument("--param_path")
    args = parser.parse_args()

    main(args.test_data_path, args.param_path)
