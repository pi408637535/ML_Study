import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivative_sigmoid(x):
    return x * (1 - x)


class NeutralNet():
    def __init__(self):
        self.__weight1 = np.random.randn(3, 4)  #3三个输入,4代表4个神经元
        self.__weight2 = np.random.randn(4, 1)

    def think(self, inputs):
        output_from_layer1 = sigmoid(np.dot(inputs, self.__weight1))
        output_from_layer2 = sigmoid(np.dot(output_from_layer1, self.__weight2))
        return output_from_layer1,output_from_layer2


    def train(self, training_inputs, set_outputs, training_iterations):
        step = 0.1
        for i in range(training_iterations):
            output_from_layer1, output_from_layer2 = self.think(training_inputs)
            weight2_error = (set_outputs - output_from_layer2)
            weight2_responsibility = weight2_error * derivative_sigmoid(output_from_layer2)

            weight1_error = (-1) * weight2_responsibility.dot(self.__weight2.T)
            weight1_responsibility = weight1_error * derivative_sigmoid(output_from_layer1)

            weight1_adjustment = step * np.dot(training_inputs.T,weight1_responsibility)
            weight2_adjustment = step * np.dot(output_from_layer1.T,weight2_responsibility)


            self.__weight1 += weight1_adjustment
            self.__weight2 += weight2_adjustment

    def synaptic_weight(self):
        return self.__weight1,self.__weight2


if __name__ == '__main__':

    training_set_inputs = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]])
    training_set_outputs = np.array([[0, 1, 1, 1, 1, 0, 0]]).T
    neutral_net = NeutralNet()
    print("New synaptic weights after training: ")
    print(neutral_net.synaptic_weight())

    neutral_net.train(training_set_inputs, training_set_outputs, 80000)
    print("New synaptic weights after training: ")
    print(neutral_net.synaptic_weight())

    # Test the neural network with a new situation.
    print("Considering new situation [1, 0, 0] -> ?: ")
    print(neutral_net.think(np.array([1, 0, 0])))
