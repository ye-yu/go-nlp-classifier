package gonlpclassifier

import (
	"encoding/csv"
	"fmt"
	"os"
	"testing"

	"gopkg.in/yaml.v3"
)

func TestPredict(t *testing.T) {
	b, err := os.Open("test/test.csv")
	if err != nil {
		t.Errorf("Cannot read file test/test.csv")
		return
	}
	defer b.Close()
	lines, err := csv.NewReader(b).ReadAll()
	if err != nil {
		t.Errorf("CSV error on test/test.csv: %v", err.Error())
		return
	}

	lines = lines[1:]
	dataset := make([]string, len(lines))
	actions := make([]string, len(lines))

	for i, x := range lines {
		dataset[i] = x[0]
		actions[i] = x[1]
	}
	synonyms := make(map[string][]string)

	naiveDataset := BuildNaiveDataset(dataset, 0.2, "<action>", actions)
	naiveModel := BuildNLPNaiveBayes(naiveDataset, synonyms)

	m, _ := yaml.Marshal(naiveModel.Predict("is there a promotion for wira project"))
	s := string(m)
	fmt.Printf("%v", s)
}

func TestModelAccuracy(t *testing.T) {
	b, err := os.Open("test/test.csv")
	if err != nil {
		t.Errorf("Cannot read file test/test.csv")
		return
	}
	defer b.Close()
	lines, err := csv.NewReader(b).ReadAll()
	if err != nil {
		t.Errorf("CSV error on test/test.csv: %v", err.Error())
		return
	}

	lines = lines[1:]
	dataset := make([]string, len(lines))
	actions := make([]string, len(lines))

	for i, x := range lines {
		dataset[i] = x[0]
		actions[i] = x[1]
	}
	synonyms := make(map[string][]string)

	naiveDataset := BuildNaiveDataset(dataset, 0.2, "<action>", actions)
	naiveModel := BuildNLPNaiveBayes(naiveDataset, synonyms)
	accuracy := TestModel(naiveModel, dataset, actions)

	if accuracy < 70 {
		t.Errorf("Accuracy is less than 70%%! - %0.f%%\n", accuracy)
	}
}
