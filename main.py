import load_data
import numpy as np
import neuralN

train, y, test, y_t = load_data.load_data(1000, 100)


print("Making Tuples...")
training_data = [(train[i], y[i]) for i in range(0, len(train))]
test_data = [(test[i], y_t[i]) for i in range(0, len(test))]
print("Done")

W1, B1, W2, B2 = neural.SGD(training_data, 30, 10, 3, test_data)
