# Golang NeuralNet

This is a rough implementation of a very simple Neural Network that showcases the basics of Machine Learning.

The network is defined in [neuralnetwork.go](neuralnetwork.go). An example for calling the network is found in [main.go](main.go).

The network is prepared to work with the MNIST dataset in CSV format (find
  it [here](https://pjreddie.com/projects/mnist-in-csv/)).

The dataset contains 60,000 greyscale images (28x28 pixels) for training and 10,000 greyscale images for testing.

[imageutil.go](imageutil.go) provides functionality to convert from CSV to PNG. It also provides a function for
  preparing custom images that are to be passed to the neural network.

The network can be stored and loaded in/from YAML files.

The implementation purposely refrains from using advanced libraries in order to give a complete look _under the hood_.
For this reason, some basic matrix operations are defined in [matrixutility.go](matrixutility.go).
Still, make sure that you have the following libraries installed:

* gonum.org/v1/gonum v0.12.0
* gopkg.in/yaml.v3 v3.0.1


__First Steps:__ Download the [datasets](https://pjreddie.com/projects/mnist-in-csv/) and have a look at the three functions in [main.go](main.go). Choose one and begin your experiments!
