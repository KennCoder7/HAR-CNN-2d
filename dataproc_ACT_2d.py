import pandas as pd
import numpy as np
from scipy import stats

# DATA_PATH = "ACT_data"


def read_data(file_path):
    column_names = ['user-id', 'activity', 'timestamp', 'x-axis', 'y-axis', 'z-axis']
    data = pd.read_csv(file_path, header=None, names=column_names)
    return data  # return a Pandas data frame


def feature_normalize(dt):  # normalise each of the accelerometer component (i.e. x, y and z)
    mu = np.mean(dt, axis=0)
    sigma = np.std(dt, axis=0)
    return (dt - mu) / sigma


print("### Process1 --- Read data --- Started ###")
dataset = read_data('ACT_data/raw/actitracker_raw.txt')
print("### Check nan of dataset before dropna(): ###")
print(dataset.isnull().any())
dataset = dataset.dropna()
print("### Check nan of dataset after dropna(): ###")
print(dataset.isnull().any())
print("### Process1 --- Read data --- Finished ###")

print("### Process2 --- Normalization --- Started ###")
dataset['x-axis'] = round(dataset['x-axis'], 2)
dataset['y-axis'] = round(dataset['y-axis'], 2)
dataset['z-axis'] = round(dataset['z-axis'], 2)
dataset['x-axis'] = feature_normalize(dataset['x-axis'])
dataset['y-axis'] = feature_normalize(dataset['y-axis'])
dataset['z-axis'] = feature_normalize(dataset['z-axis'])
print("### Check nan of dataset_normalize before dropna(): ###")
print(dataset.isnull().any())
dataset = dataset.dropna()
print("### Check nan of dataset_normalize after dropna(): ###")
print(dataset.isnull().any())
# print(dataset)
print("### Process2 --- Normalization --- Finished ###")


def windows(data, size):
    start = 0
    while start < data.count():
        yield start, start + size
        start += (size / 2)


def segment_signal(data, window_size=90):
    segments = np.empty((0, window_size, 3))
    labels = np.empty((0))
    n = 0
    for (start, end) in windows(data["timestamp"], window_size):
        start = int(start)
        end = int(end)
        x = data["x-axis"][start:end]
        y = data["y-axis"][start:end]
        z = data["z-axis"][start:end]
        if len(dataset["timestamp"][start:end]) == window_size:
            segments = np.vstack([segments, np.dstack([x, y, z])])
            labels = np.append(labels, stats.mode(data["activity"][start:end])[0][0])
        if start-n > 0.05*data["timestamp"].count():
            n = start
            print("### Process3 --- Segment --- In progress --- [ ",
                  100*round(start/data['timestamp'].count(), 2), "% ] Finished ###")
    return segments, labels


def str_to_num(list):
    list_unique = np.unique(list)
    list_unique_len = len(list_unique)
    new_list = np.array([], dtype=int)
    for i in list:
        for j in range(list_unique_len):
            if list_unique[j] == i:
                new_list = np.append(new_list, int(j))
    return new_list


print("### Process3 --- Segment --- Started ###")
segments, labels = segment_signal(dataset)
print("### Process3 --- Segment --- Finished ###")

print("### Process4 --- Data&Labels Transform --- Started ###")
reshaped_segments = segments.reshape(len(segments), 1, 90, 3)
data_2d = reshaped_segments.transpose((0, 3, 2, 1))
unique_labels = np.unique(labels)
num_labels = str_to_num(labels)
unique_num_labels = np.unique(num_labels)
unique_labels = np.append(unique_labels, unique_num_labels)
labels_2d = np.asarray(pd.get_dummies(labels), dtype=np.int8)
print("### Process4 --- Data&Labels Transform --- Finished ###")

print("### Process5 --- Save --- Started ###")
np.save('ACT_data/np_2d/np_data_2d_v1.npy', data_2d)
print("data_2d shape: ", data_2d.shape)
np.save('ACT_data/np_2d/np_labels_1d_v1.npy', num_labels)
print("num_labels: ", num_labels)
np.save('ACT_data/np_2d/np_real_str_labels_1d_v1.npy', labels)
print("real_labels: ", labels)
np.save('ACT_data/np_2d/np_unique_labels_1d_v1.npy', unique_labels)
print("unique_labels: ", unique_labels)
print("labels number: ", labels.shape[0])
np.save('ACT_data/np_2d/np_labels_2d_v1.npy', labels_2d)
print("2d labels: ", labels_2d)
print("### Process5 --- Save --- Finished ###")