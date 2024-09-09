import numpy as np

def relu(x):
    return np.clip(x, 0, 127)

input = np.loadtxt("./map/Rec_010.txt", skiprows=70, max_rows=140)

if input.ndim == 1:
    input = np.expand_dims(input, axis=0)

g = 1024

# L0
weight = np.loadtxt("./map/dnn_weight_0.txt", delimiter=',', converters={-1: lambda s: float(0)})[:, :-1]
bias = np.loadtxt("./map/dnn_bias_0.txt", delimiter=',', converters={-1: lambda s: float(0)})[:-1]
output_expected_L0 = np.clip(np.round((np.matmul(input, weight) + bias) / g), -128, 127)
output_expected_L0R = relu(output_expected_L0)

# L1
weight = np.loadtxt("./map/dnn_weight_1.txt", delimiter=',', converters={-1: lambda s: float(0)})[:, :-1]
bias = np.loadtxt("./map/dnn_bias_1.txt", delimiter=',', converters={-1: lambda s: float(0)})[:-1]
output_expected_L1 = np.clip(np.round((np.matmul(output_expected_L0R, weight) + bias) / g), -128, 127)
output_expected_L1R = relu(output_expected_L1)

# L2
weight = np.loadtxt("./map/dnn_weight_2.txt", delimiter=',', converters={-1: lambda s: float(0)})[:, :-1]
bias = np.loadtxt("./map/dnn_bias_2.txt", delimiter=',', converters={-1: lambda s: float(0)})[:-1]
output_expected_L2 = np.clip(np.round((np.matmul(output_expected_L1R, weight) + bias) / g), -128, 127)
output_expected_L2R = relu(np.clip(output_expected_L2 * 2, -128, 127))

# L3
weight = np.loadtxt("./map/dnn_weight_3.txt", delimiter=',', converters={-1: lambda s: float(0)})[:, :-1]
bias = np.loadtxt("./map/dnn_bias_3.txt", delimiter=',', converters={-1: lambda s: float(0)})[:-1]
output_expected_L3 = np.clip(np.round((np.matmul(output_expected_L2R, weight) + 2 * bias) / g), -128, 127)
output_expected_L3R = relu(np.clip(output_expected_L3 * 2, -128, 127))

# L4
weight = np.loadtxt("./map/dnn_weight_4.txt", delimiter=',', converters={-1: lambda s: float(0)})[:, :-1]
bias = np.loadtxt("./map/dnn_bias_4.txt", delimiter=',', converters={-1: lambda s: float(0)})[:-1]
output_expected_L4 = np.clip(np.round((np.matmul(output_expected_L3R, weight) + 4 * bias) / g), -128, 127)

with open("./map/expected_in.bin", "wb") as f:
    f.write(np.array(np.shape(input)[0],  dtype=np.uint16))
    f.write(np.array(input,               dtype=np.uint16))
    f.write(np.array(output_expected_L0R, dtype=np.uint16))
    f.write(np.array(output_expected_L1R, dtype=np.uint16))
    f.write(np.array(output_expected_L2R, dtype=np.uint16))
    f.write(np.array(output_expected_L3R, dtype=np.uint16))

with open("./map/expected_out.bin", "wb") as f:
    f.write(np.array(output_expected_L0, dtype=np.int16))
    f.write(np.array(output_expected_L1, dtype=np.int16))
    f.write(np.array(output_expected_L2, dtype=np.int16))
    f.write(np.array(output_expected_L3, dtype=np.int16))
    f.write(np.array(output_expected_L4, dtype=np.int16))


