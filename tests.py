from FFNN import NeuralNetwork

if __name__ == '__main__':

    print("TESTING")

    nn1 = NeuralNetwork(file_name="test_weights/test1")
    nn1.initialize_random(input_size=3, layer_sizes=[5, 2])
    nn1.save_weights()

    nn2 = NeuralNetwork(file_name="test_weights/test1")
    nn2.load_weights()

    assert nn1 == nn2
    assert nn1.forward_pass([1, 2, 3]) == nn2.forward_pass([1, 2, 3])

    nn3 = NeuralNetwork(file_name="test_weights/test2")
    nn3.load_weights()

    assert nn3.forward_pass([1, 0]) == [0.6629970129852887, 0.7253160725279748]
