package gonlpclassifier

import (
	"fmt"
	"strings"

	"gopkg.in/yaml.v3"
)

func TestModel(naiveModel NaiveBayesModel,
	dataset []string,
	actions []string,
) float64 {

	fmt.Println("=====================")
	fmt.Println("Testing training set")
	fmt.Println("=====================")

	success := 0
	for i, sentence := range dataset {
		fmt.Println("")
		fmt.Printf("Sentence: %v\n", sentence)
		predictionResult := naiveModel.Predict(sentence)
		class := predictionResult.ClassesScore[0].Class
		expected := actions[i]
		if expected == class {
			success++
		}

		m, _ := yaml.Marshal(predictionResult)

		splitYaml := strings.Split(string(m), "\n")
		for _, t := range splitYaml {
			fmt.Printf("    %v\n", t)
		}

		fmt.Printf("    Expected : %v\n", expected)
		fmt.Printf("    Predicted: %v\n", class)
		if expected == class {
			fmt.Println("    Correct  : Yes")
		} else {
			fmt.Println("    Correct  : No")
		}
	}

	trainingAccuracy := float64(success) / float64(len(dataset)) * 100
	fmt.Println("=====================")
	fmt.Println("")
	fmt.Printf("Test summary: %0.f%% successful\n", trainingAccuracy)

	return trainingAccuracy
}
