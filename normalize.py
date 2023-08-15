

def normalize_range(data_arr, min_list, max_list):
    for i in range(data_arr.shape[0]):
        for j in range(data_arr.shape[1]):
            data_arr[i, j] = (data_arr[i, j] - min_list[j]) / (
                max_list[j] - min_list[j]
            )

    return data_arr


def get_min_max(data_arr):
    min_list = []
    max_list = []
    for i in range(data_arr.shape[1]):
        col = data_arr[:, i]
        min_list.append(float(min(col)))
        max_list.append(float(max(col)))

    return min_list, max_list
