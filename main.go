package main

import (
	"encoding/csv"
	"fmt"
	"os"
	"os/signal"
	"syscall"
)

func main() {
	var checkpointFile = "./neuralnettest.weights"
	TrainAndTest(checkpointFile)
	//LoadAndClassify(checkpointFile)
	//ConvertToPng()
}

/*
Converts the MNIST CSV files to PNGs
*/
func ConvertToPng() {
	CsvToPng("/mnt/data/Datasets/mnist/mnist_train.csv", "/mnt/data/Datasets/mnist/images/train")
	CsvToPng("/mnt/data/Datasets/mnist/mnist_test.csv", "/mnt/data/Datasets/mnist/images/test")
}

/*
Loads an image file and performs classifications.
*/
func LoadAndClassify(checkpointFile string) {
	var imageFile = "./images/0.png"

	net, err := LoadNeuralNet(checkpointFile)
	if err != nil {
		panic(err)
	}
	data, _, label, err := ImageToQueryInput(imageFile)
	//data, _, label, err := ImageToQueryInput("/mnt/data/Datasets/mnist/images/test/5-image-857.png")
	if err != nil {
		panic(err)
	}
	queryResult := net.Query(data)
	var classifiedIndex int = Argmax(queryResult)
	fmt.Printf("Image with label %v was classified as %v\n", label, classifiedIndex)
	fmt.Printf("%v", queryResult)
}

/*
Performs training and testing, stores weights after training
*/
func TrainAndTest(checkpointFile string) {
	//var trainFile = "/mnt/data/Datasets/dummy/train.csv"
	//var testFile = "/mnt/data/Datasets/dummy/test.csv"
	//var inputNodes = 500
	//var hiddenNodes = 200
	//var outputNodes = 2
	//var learningRate float64 = 0.005
	//var epochs = 10
	//var numValidation = 5000
	var trainFile = "/mnt/data/Datasets/mnist/mnist_train.csv"
	var testFile = "/mnt/data/Datasets/mnist/mnist_test.csv"
	var inputNodes = 784
	var hiddenNodes = 200
	var outputNodes = 10
	var learningRate float64 = 0.005
	var epochs = 50
	var numValidation = 5000
	var n *NeuralNetwork = NewNeuralNetwork(inputNodes, hiddenNodes, outputNodes, learningRate)
	file, err := os.Open(trainFile)
	if err != nil {
		panic(err)
	}
	reader := csv.NewReader(file)
	records, _ := reader.ReadAll()
	trainData, err := PrepareDataset(records)
	trainLabels, err := PrepareTrainLabels(records, 9)
	var validationData [][]float64 = make([][]float64, len(trainData))
	for i, _ := range trainData {
		validationData[i] = trainData[i]
	}
	validationLabels, err := PrepareTestLabels(records, 9)
	file, err = os.Open(testFile)
	if err != nil {
		panic(err)
	}
	reader = csv.NewReader(file)
	records, _ = reader.ReadAll()
	testData, err := PrepareDataset(records)
	testLabels, err := PrepareTestLabels(records, 1)

	// channel c listens for keyboard interrupts and closes the channel cancel. This in return tells n.TrainEpochs to stop training.
	c := make(chan os.Signal, 1)
	cancel := make(chan any, 1)
	signal.Notify(c, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-c
		cancel <- nil
	}()

	fmt.Println("Beginning with Training")
	n = n.TrainEpochs(trainData, trainLabels, validationData[:numValidation], validationLabels[:numValidation], epochs, cancel, true)

	fmt.Printf("Storing Weights under %s.\n", checkpointFile)
	n.StoreNeuralNet(checkpointFile)
	fmt.Println("Beginning with Testing")
	correct, length := n.Validate(testData, testLabels)
	accuracy := float64(correct) / float64(length)
	fmt.Println("Accuracy: ", accuracy)

}
