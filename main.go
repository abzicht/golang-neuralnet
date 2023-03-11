package main

import (
	"encoding/csv"
	"fmt"
	"os"
	"os/signal"
	"syscall"
)

func main() {
	LoadAndClassify()
	//ConvertToPng()
}

func ConvertToPng() {
	CsvToPng("/mnt/data/Datasets/mnist/mnist_train.csv", "/mnt/data/Datasets/mnist/images/train")
	CsvToPng("/mnt/data/Datasets/mnist/mnist_test.csv", "/mnt/data/Datasets/mnist/images/test")
}

func LoadAndClassify() {

	net, err := LoadNeuralNet("./neuralnet.weights")
	if err != nil {
		panic(err)
	}
	data, _, label, err := ImageToQueryInput("./images/04.png")
	if err != nil {
		panic(err)
	}
	queryResult := net.Query(data)
	var classifiedIndex int = Argmax(queryResult)
	fmt.Printf("Image with label %v was classified as %v\n", label, classifiedIndex)
	fmt.Printf("%v", queryResult)
}
func TrainAndTest() {
	var inputNodes = 784
	var hiddenNodes = 200
	var outputNodes = 10
	var learningRate float64 = 0.005
	var epochs = 40
	var numValidation = 5000
	var checkpointPath string = "./"
	var n *NeuralNetwork = NewNeuralNetwork(inputNodes, hiddenNodes, outputNodes, learningRate)
	file, err := os.Open("/mnt/data/Datasets/mnist/mnist_train.csv")
	if err != nil {
		panic(err)
	}
	reader := csv.NewReader(file)
	records, _ := reader.ReadAll()
	trainData, err := PrepareDataset(records)
	trainLabels, err := PrepareTrainLabels(records)
	validationData := trainData
	validationLabels, err := PrepareTestLabels(records)
	file, err = os.Open("/mnt/data/Datasets/mnist/mnist_test.csv")
	if err != nil {
		panic(err)
	}
	reader = csv.NewReader(file)
	records, _ = reader.ReadAll()
	testData, err := PrepareDataset(records)
	testLabels, err := PrepareTestLabels(records)

	c := make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-c
		fmt.Printf("Interrupt detected. Storing Weights under %s", checkpointPath)
		n.StoreNeuralNet(checkpointPath + "neuralnet.weights")
		fmt.Println(" Exiting.")
		os.Exit(1)
	}()

	println("Beginning with Training")
	for epoch := 0; epoch < epochs; epoch++ {
		print("Epoch ", epoch+1)
		// Randomly Shuffle (Fisher–Yates shuffle):
		// TODO: Currently, shuffling destroys the training effect. Reason unknown
		//for i := len(trainData) - 1; i >= 0; i-- {
		//	j := rand.Intn(i + 1)
		//	trainData[i], trainData[j] = trainData[j], trainData[i]
		//	trainLabels[i], trainLabels[j] = trainLabels[j], trainLabels[i]
		//	validationData[i], validationData[j] = validationData[j], validationData[i]
		//	validationLabels[i], validationLabels[j] = validationLabels[j], validationLabels[i]
		//}
		for i, _ := range trainData {
			n = n.Train(trainData[i], trainLabels[i])
		}
		print(" done. Validation ")
		correct, length := n.Validate(validationData[:numValidation], validationLabels[:numValidation])
		accuracy := float64(correct) / float64(length)
		fmt.Println("accuracy:", accuracy)
	}

	n.StoreNeuralNet(fmt.Sprintf("%sneuralnet.weights", checkpointPath))
	println("Beginning with Testing")
	correct, length := n.Validate(testData, testLabels)
	accuracy := float64(correct) / float64(length)
	println("Accuracy: ", accuracy)

}
