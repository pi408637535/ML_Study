import numpy as np
import math
import matplotlib.pyplot as plt


def generate_data():
    x = np.linspace(1,100,num=44)
    y = 4 * x
    return (x,y)


def generate_adjust_weight(output, target, x):
    adjust = (-1) * (target - output) * x
    return adjust

def feedback_weight(original_weight, output, target, x):
    stop = 0.01
    w = original_weight - stop * generate_adjust_weight(output, target, x)
    return w

def train(data_x, data_y, init_weight, condition):
    size = len(data_x)

    output_list =[]

    for i in range(size):
        output = init_weight * data_x[i]
        output_list.append(output)
        print(init_weight)
        if math.fabs(output - data_y[i]) < condition:
            break
        else:
            init_weight = feedback_weight(init_weight, output, data_y[i], data_x[i])

    plt.plot(data_x[0: len(output_list)], output_list, 'o-', color='g')
    plt.scatter(data_x[0: len(output_list)], data_y[0: len(output_list)], alpha=0.2)
    plt.show()


if __name__ == '__main__':
    x,y = generate_data()
    #print(x)
    #print(y)
    data_weight = np.random.randn(3)
    condition = 0.5
    #print(data_weight)

    train(x, y, data_weight[0], condition)