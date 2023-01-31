from FFNN import NeuralNetwork

if __name__ == '__main__':

    nn1 = NeuralNetwork()
    nn1.initialize_random(input_size=2, layer_sizes=[1, 2])

    assert nn1.size() == [[3], [2, 2]]

    nn2 = NeuralNetwork(file_name="test_weights/test1")
    nn2.initialize_random(input_size=3, layer_sizes=[5, 2])
    nn2.save_weights()

    nn3 = NeuralNetwork(file_name="test_weights/test1")
    nn3.load_weights()

    assert nn2 == nn3
    assert nn2.forward_pass([1, 2, 3]) == nn3.forward_pass([1, 2, 3])

    nn4 = NeuralNetwork(file_name="test_weights/test2")
    nn4.load_weights()

    assert nn4.forward_pass([999, 0.2]) == [1]

    nn5 = NeuralNetwork(file_name="test_weights/test3")
    nn5.load_weights()

    assert nn5.forward_pass([1, 0]) == [0.6629970129852887, 0.7253160725279748]

    nn6 = NeuralNetwork()
    nn6.initialize_random(input_size=3, layer_sizes=[1])

    training_data = [[0, 0, 0],
                     [0, 0, 1],
                     [0, 1, 1],
                     [1, 0, 0],
                     [1, 0, 1],
                     [1, 1, 0]]

    training_data_outputs = [[0],
                             [1],
                             [1],
                             [0],
                             [1],
                             [0]]

    nn6.train(training_data=training_data,
              training_data_outputs=training_data_outputs, epochs=10, learning_rate=1)

    test_data = [[0, 1, 0],  # Output should be 0
                 [1, 1, 1]]  # Output should be 1

    assert nn6.predict(inputs=test_data[0])[0] < 0.5
    assert nn6.predict(inputs=test_data[1])[0] > 0.5
