package main

import (
	"encoding/csv"
	"fmt"
	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/stat/distuv"
	"gopkg.in/yaml.v3"
	"math"
	"os"
	"strconv"
)

type NeuralNetwork struct {
	Inodes       int
	Hnodes       int
	Onodes       int
	Learningrate float64
	TrainingStep uint64
	Wih          *Weights `yaml:"Wih"`
	Who          *Weights `yaml:"Who"`
}

type Weights struct {
	WeightArray [][]float64 `yaml:"WeightArray,flow"`
}

func NewNeuralNetwork(inodes, hnodes, onodes int, learningrate float64) *NeuralNetwork {
	wihNormal := distuv.Normal{0, math.Pow(float64(hnodes), -0.5), rand.NewSource(0)}
	whoNormal := distuv.Normal{0, math.Pow(float64(onodes), -0.5), rand.NewSource(0)}
	wih := CreateWeights(hnodes, inodes, wihNormal.Rand)
	who := CreateWeights(onodes, hnodes, whoNormal.Rand)
	return &NeuralNetwork{inodes, hnodes, onodes, learningrate, 0, &wih, &who}
}

func NewWeights(weightArray [][]float64) *Weights {
	return &Weights{weightArray}
}

/*
Creates a 2D matrix with shape "rows" x "columns". Fills default values using the fill function.
If no fill function is provided, matrix is initialized without values.
*/
func CreateWeights(rows, columns int, fill func() float64) Weights {
	var data [][]float64 = make([][]float64, rows)
	for i, _ := range data {
		data[i] = make([]float64, columns)
		if fill != nil {
			for j, _ := range data[i] {
				data[i][j] = fill()
			}
		}
	}
	return Weights{data}
}

/*
Returns the number of training steps done so far
*/
func (n NeuralNetwork) GetTrainingSteps() uint64 {
	return n.TrainingStep
}

/*
Stores all neuralnet parameters in a YAML file.
*/
func (n NeuralNetwork) StoreWeights(filePath string) error {
	yamlData, err := yaml.Marshal(&n)
	if err != nil {
		return err
	}
	//fmt.Println(string(yamlData))

	err = os.WriteFile(filePath, yamlData, 0644)
	return err
}

func LoadWeights(filePath string) (*NeuralNetwork, error) {
	var n NeuralNetwork
	f, err := os.Open(filePath)
	if err != nil {
		return &n, err
	}
	defer f.Close()
	decoder := yaml.NewDecoder(f)
	err = decoder.Decode(&n)
	return &n, err
}

/*
Logistic Sigmoid for a float
*/
func expit(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func (n NeuralNetwork) ActivationFunction(array [][]float64) [][]float64 {
	var outArray = make([][]float64, len(array))
	for i, _ := range outArray {
		outArray[i] = make([]float64, len(array[i]))
		for j, _ := range outArray[i] {
			outArray[i][j] = expit(array[i][j])
		}
	}
	return outArray
}

func TransposeArray(array []float64) [][]float64 {
	var arrayTransposed [][]float64 = make([][]float64, len(array))
	for i, _ := range arrayTransposed {
		arrayTransposed[i] = []float64{array[i]}
	}
	return arrayTransposed
}

func TransposeMatrix(array [][]float64) [][]float64 {
	if len(array) < 1 {
		return array
	}
	var arrayTransposed [][]float64 = make([][]float64, len(array[0]))
	for i, _ := range arrayTransposed {
		arrayTransposed[i] = make([]float64, len(array))
		for j, _ := range arrayTransposed[i] {
			arrayTransposed[i][j] = array[j][i]
		}
	}
	return arrayTransposed
}

func DotProduct(matrixA, matrixB [][]float64) [][]float64 {
	var outMatrix [][]float64 = make([][]float64, len(matrixA))
	for i, _ := range matrixA {
		outMatrix[i] = make([]float64, len(matrixB[0]))
		for j, _ := range matrixB[0] {
			var sum float64 = 0
			for k, _ := range matrixB {
				sum += matrixA[i][k] * matrixB[k][j]
			}
			outMatrix[i][j] = sum
		}
	}
	return outMatrix
}

func MatrixOperation(matrixA, matrixB [][]float64, operator func(float64, float64) float64) [][]float64 {
	var outMatrix [][]float64 = make([][]float64, len(matrixA))
	for i, _ := range matrixA {
		outMatrix[i] = make([]float64, len(matrixA[i]))
		for j, _ := range matrixB[i] {
			outMatrix[i][j] = operator(matrixA[i][j], matrixB[i][j])
		}
	}
	return outMatrix
}

func (n NeuralNetwork) Train(inputs []float64, targets []float64) *NeuralNetwork {
	var inputsTransposed = TransposeArray(inputs)
	var targetsTransposed = TransposeArray(targets)

	var hiddenInputs = DotProduct(n.Wih.WeightArray, inputsTransposed)
	var hiddenOutputs = n.ActivationFunction(hiddenInputs)

	var finalInputs = DotProduct(n.Who.WeightArray, hiddenOutputs)
	var finalOutputs = n.ActivationFunction(finalInputs)

	var outputErrors [][]float64 = MatrixOperation(targetsTransposed, finalOutputs, func(a float64, b float64) float64 {
		return a - b
	})
	var hiddenErrors = DotProduct(TransposeMatrix(n.Who.WeightArray), outputErrors)
	// who = who + learningrate * Dot(output*finalOutput*(1.0-finalOutput), hiddenOutput.Tranpose)
	who := &Weights{MatrixOperation(n.Who.WeightArray, DotProduct(MatrixOperation(outputErrors, finalOutputs, func(a float64, b float64) float64 {
		return a * b * (1.0 - b)
	}), TransposeMatrix(hiddenOutputs)),
		func(a float64, b float64) float64 {
			return a + (n.Learningrate)*b
		})}

	wih := &Weights{MatrixOperation(n.Wih.WeightArray, DotProduct(MatrixOperation(hiddenErrors, hiddenOutputs, func(a float64, b float64) float64 {
		return a * b * (1.0 - b)
	}), TransposeMatrix(inputsTransposed)),
		func(a float64, b float64) float64 {
			return a + n.Learningrate*b
		})}
	return &NeuralNetwork{Inodes: n.Inodes, Hnodes: n.Hnodes, Onodes: n.Onodes, Learningrate: n.Learningrate, TrainingStep: n.TrainingStep + 1, Wih: wih, Who: who}
}

func (n NeuralNetwork) Query(inputs []float64) [][]float64 {
	var inputsTransposed = TransposeArray(inputs)
	var hiddenInputs = DotProduct(n.Wih.WeightArray, inputsTransposed)
	var hiddenOutputs = n.ActivationFunction(hiddenInputs)
	var finalInputs = DotProduct(n.Who.WeightArray, hiddenOutputs)
	var finalOutputs = n.ActivationFunction(finalInputs)
	return finalOutputs
}

/*
Returns the number of correctly classified samples and the number of provided samples
*/
func (n NeuralNetwork) Validate(inputs [][]float64, labels []int) (int, int) {
	classifications := []float64{}
	var correct int = 0
	for i, _ := range inputs {
		outputs := n.Query(inputs[i])
		if labels[i] == Argmax(outputs) {
			classifications = append(classifications, 1)
			correct++
		} else {
			classifications = append(classifications, 0)
		}
	}
	return correct, len(classifications)
}

/*
Returns the top level index of the subarray with the highest value
*/
func Argmax(values [][]float64) int {
	curMax := -1e100
	index := 0
	for i, _ := range values {
		localMax := -1e100
		for j, _ := range values[i] {
			if values[i][j] > localMax {
				localMax = values[i][j]
			}
		}
		if localMax > curMax {
			curMax = localMax
			index = i
		}
	}
	return index
}

func prepareTrainLabels(rawData [][]string) [][]float64 {
	labels := make([][]float64, len(rawData))
	for i, _ := range rawData {
		labels[i] = make([]float64, 10)
		for k := 0; k < 10; k++ {
			labels[i][k] = 0
		}
		value, err := strconv.Atoi(rawData[i][0])
		if err != nil {
			panic(err)
		}
		if value < 0 || value > 9 {
			panic("Label is not in allowed space: " + strconv.Itoa(value))
		}
		labels[i][value] = 0.999
	}
	return labels
}
func prepareTestLabels(rawData [][]string) []int {
	labels := make([]int, len(rawData))
	for i, _ := range rawData {
		value, err := strconv.Atoi(rawData[i][0])
		if err != nil {
			panic(err)
		}
		if value < 0 || value > 9 {
			panic("Label is not in allowed space: " + strconv.Itoa(value))
		}
		labels[i] = value
	}
	return labels
}

func prepareDataset(rawData [][]string) [][]float64 {
	outputData := make([][]float64, len(rawData))
	for i, _ := range rawData {
		outputData[i] = make([]float64, len(rawData[i])-1)
		for j, _ := range rawData[i] {
			if j == 0 {
				continue
			} else {
				value, err := strconv.Atoi(rawData[i][j])
				if err != nil {
					panic(err)
				}
				outputData[i][j-1] = ((float64(value) / 255.0) * 0.999) + 0.001
			}
		}
	}
	return outputData
}

func main2() {
	array := []int{1, 2, 3, 4, 5, 6, 7}
	for i := len(array) - 1; i >= 0; i-- {
		//for i, _ := range array {
		j := rand.Intn(i + 1)
		array[i], array[j] = array[j], array[i]
	}
	fmt.Println(array)
}

func main() {
	var inputNodes = 784
	var hiddenNodes = 200
	var outputNodes = 10
	var learningRate float64 = 0.005
	var epochs = 0
	var numValidation = 5000
	var n *NeuralNetwork = NewNeuralNetwork(inputNodes, hiddenNodes, outputNodes, learningRate)
	file, err := os.Open("/mnt/data/Datasets/mnist/mnist_train.csv")
	if err != nil {
		panic(err)
	}
	reader := csv.NewReader(file)
	records, _ := reader.ReadAll()
	trainData := prepareDataset(records)
	trainLabels := prepareTrainLabels(records)
	validationData := trainData
	validationLabels := prepareTestLabels(records)
	file, err = os.Open("/mnt/data/Datasets/mnist/mnist_test.csv")
	if err != nil {
		panic(err)
	}
	reader = csv.NewReader(file)
	records, _ = reader.ReadAll()
	//testData := prepareDataset(records)
	//testLabels := prepareTestLabels(records)

	println("Beginning with Training")
	for epoch := 0; epoch < epochs; epoch++ {
		print("Epoch ", epoch+1)
		// Randomly Shuffle (Fisher–Yates shuffle):
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

	println("Beginning with Testing")
	//correct, length := n.Validate(testData, testLabels)
	//accuracy := float64(correct) / float64(length)
	//println("Accuracy: ", accuracy)

	n.StoreWeights(fmt.Sprintf("weights_epoch_%b_accuracy_%e.nn", epochs, 0.0))       //accuracy))
	n, err = LoadWeights(fmt.Sprintf("weights_epoch_%b_accuracy_%e.nn", epochs, 0.0)) //accuracy))
	if err != nil {
		panic(err)
	}
	fmt.Println(n.Who)
}
