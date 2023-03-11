package main

import (
	"errors"
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
	Wih          [][]float64 `yaml:"Wih,flow"` // "flow" ensures that yaml stores the weights in a single line
	Who          [][]float64 `yaml:"Who,flow"`
}

/*
Initializes a new neural network with the specified layer sizes and learning rate. TrainingStep is initialized with 0,
Wih and Who are initialized with a random gaussian distribution
*/
func NewNeuralNetwork(inodes, hnodes, onodes int, learningrate float64) *NeuralNetwork {
	//Preparation of gaussian distribution function for Wih and Who
	wihNormal := distuv.Normal{0, math.Pow(float64(hnodes), -0.5), rand.NewSource(0)}
	whoNormal := distuv.Normal{0, math.Pow(float64(onodes), -0.5), rand.NewSource(0)}
	wih := CreateWeights(hnodes, inodes, wihNormal.Rand)
	who := CreateWeights(onodes, hnodes, whoNormal.Rand)
	return &NeuralNetwork{inodes, hnodes, onodes, learningrate, 0, wih, who}
}

/*
Creates a 2D matrix with shape "rows" x "columns". Fills default values using the fill function.
If no fill function is provided, matrix is initialized without values.
*/
func CreateWeights(rows, columns int, fill func() float64) [][]float64 {
	var data [][]float64 = make([][]float64, rows)
	for i, _ := range data {
		data[i] = make([]float64, columns)
		if fill != nil {
			for j, _ := range data[i] {
				data[i][j] = fill()
			}
		}
	}
	return data
}

/*
Returns the number of training steps done so far.
*/
func (n NeuralNetwork) GetTrainingSteps() uint64 {
	return n.TrainingStep
}

/*
Stores all neuralnet parameters in a YAML file.
Compatible with LoadNeuralNet.
*/
func (n NeuralNetwork) StoreNeuralNet(filePath string) error {
	yamlData, err := yaml.Marshal(&n)
	if err != nil {
		return err
	}
	err = os.WriteFile(filePath, yamlData, 0644)
	return err
}

/*
Loads a neural net stored in a yaml file located in `filePath`. Should have been created with StoreNeuralNet.
*/
func LoadNeuralNet(filePath string) (*NeuralNetwork, error) {
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
Logistic Sigmoid for a float64
*/
func expit(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func ActivationFunction(array [][]float64) [][]float64 {
	var outArray = make([][]float64, len(array))
	for i, _ := range outArray {
		outArray[i] = make([]float64, len(array[i]))
		for j, _ := range outArray[i] {
			outArray[i][j] = expit(array[i][j])
		}
	}
	return outArray
}

/*
Performs a training step for a single sample. Returns a new neural network object that contains the new training progress

inputs: image, in the form of a single array
targets: one-hot encoded target value (should be in [0.0001,0.9999]
*/
func (n NeuralNetwork) Train(inputs []float64, targets []float64) *NeuralNetwork {
	var inputsTransposed = TransposeArray(inputs)
	var targetsTransposed = TransposeArray(targets)

	var hiddenInputs = DotProduct(n.Wih, inputsTransposed)
	var hiddenOutputs = ActivationFunction(hiddenInputs)

	var finalInputs = DotProduct(n.Who, hiddenOutputs)
	var finalOutputs = ActivationFunction(finalInputs)

	var outputErrors [][]float64 = MatrixOperation(targetsTransposed, finalOutputs, func(a float64, b float64) float64 {
		return a - b
	})
	var hiddenErrors = DotProduct(TransposeMatrix(n.Who), outputErrors)
	/*
		Read the lines below like this:
			who = who + learningrate * Dot(output*finalOutput*(1.0-finalOutput), hiddenOutput.Tranpose)
	*/
	who := MatrixOperation(n.Who, DotProduct(MatrixOperation(outputErrors, finalOutputs, func(a float64, b float64) float64 {
		return a * b * (1.0 - b)
	}), TransposeMatrix(hiddenOutputs)),
		func(a float64, b float64) float64 {
			return a + (n.Learningrate)*b
		})

	wih := MatrixOperation(n.Wih, DotProduct(MatrixOperation(hiddenErrors, hiddenOutputs, func(a float64, b float64) float64 {
		return a * b * (1.0 - b)
	}), TransposeMatrix(inputsTransposed)),
		func(a float64, b float64) float64 {
			return a + n.Learningrate*b
		})
	return &NeuralNetwork{Inodes: n.Inodes, Hnodes: n.Hnodes, Onodes: n.Onodes, Learningrate: n.Learningrate, TrainingStep: n.TrainingStep + 1, Wih: wih, Who: who}
}

/*
Returns the model's output layer values after processing the provided input
*/
func (n NeuralNetwork) Query(inputs []float64) [][]float64 {
	var inputsTransposed = TransposeArray(inputs)
	var hiddenInputs = DotProduct(n.Wih, inputsTransposed)
	var hiddenOutputs = ActivationFunction(hiddenInputs)
	var finalInputs = DotProduct(n.Who, hiddenOutputs)
	var finalOutputs = ActivationFunction(finalInputs)
	return finalOutputs
}

/*
Returns the number of correctly classified samples and the number of provided samples
TODO: Parallelize
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
	curMax := values[0][0]
	index := 0
	for i, _ := range values {
		localMax := values[i][0]
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

/*
Returns a two-dimensional array that contains One-Hot encoded labels
Label values are restricted to int values from 0 to 9.
*/
func PrepareTrainLabels(rawData [][]string) ([][]float64, error) {
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
			return nil, errors.New("Label is not in allowed space (0-9): " + strconv.Itoa(value))
		}
		labels[i][value] = 0.999
	}
	return labels, nil
}

/*
Returns a one-dimensional array that contains the label values.
Label values are restricted to int values from 0 to 9.
*/
func PrepareTestLabels(rawData [][]string) ([]int, error) {
	labels := make([]int, len(rawData))
	for i, _ := range rawData {
		value, err := strconv.Atoi(rawData[i][0])
		if err != nil {
			panic(err)
		}
		if value < 0 || value > 9 {
			return nil, errors.New("Label is not in allowed space (0-9): " + strconv.Itoa(value))
		}
		labels[i] = value
	}
	return labels, nil
}

/*
Normalizes pixel values to the range (0,1).
*/
func PrepareDataset(rawData [][]string) ([][]float64, error) {
	outputData := make([][]float64, len(rawData))
	for i, _ := range rawData {
		outputData[i] = make([]float64, len(rawData[i])-1)
		for j, _ := range rawData[i] {
			if j == 0 {
				continue
			} else {
				value, err := strconv.Atoi(rawData[i][j])
				if err != nil {
					return nil, err
				}
				outputData[i][j-1] = ((float64(value) / 255.0) * 0.999) + 0.001
			}
		}
	}
	return outputData, nil
}