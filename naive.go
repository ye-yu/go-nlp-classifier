package gonlpclassifier

import (
	"fmt"
	"sort"

	"github.com/kniren/gota/dataframe"
)

func BuildNLPNaiveBayes(naiveDataset NaiveBayesDataset, synonyms map[string][]string) NaiveBayesModel {
	df := naiveDataset.Dataframe
	classColumn := naiveDataset.ClassName
	classes := make(map[interface{}]bool)
	classesSet := make([]interface{}, 0)
	featuresSet := make([]string, 0)
	featuresMap := make(map[string]bool)

	for i := 0; i < df.Col(classColumn).Len(); i++ {
		classes[df.Col(classColumn).Val(i)] = true
	}

	naiveInstances := make([]NaiveBayesInstance, 0)
	for class := range classes {
		classesSet = append(classesSet, class)
		classInstances := df.Filter(dataframe.F{
			Colname:    classColumn,
			Comparator: "==",
			Comparando: fmt.Sprintf("%v", class),
		})
		instancesCountFloat := float64(1 * classInstances.Nrow())

		var naiveInstance NaiveBayesInstance
		naiveInstance.Class = class
		naiveInstance.Prob = 1 / float64(len(classes))
		naiveInstance.Features = make(map[interface{}]float64)

		for _, feature := range df.Names() {
			if feature == classColumn {
				continue
			}
			featuresSet = append(featuresSet, feature)
			featuresMap[feature] = true
			sum := SumFloat(classInstances.Col(feature).Float()...)
			sum = Bound(sum, 0, instancesCountFloat)
			naiveInstance.Features[feature] = sum / instancesCountFloat
		}

		naiveInstances = append(naiveInstances, naiveInstance)
	}

	synonymLookup := make(map[string]string)

	for key, values := range synonyms {
		for _, value := range values {
			synonymLookup[value] = key
		}
	}

	return NaiveBayesModel{
		Weights:          naiveInstances,
		Features:         featuresSet,
		FeaturesMap:      featuresMap,
		Classes:          classesSet,
		UnsignificantMap: naiveDataset.Unsignificant,
		Synonyms:         synonyms,
		SynonymsLookup:   synonymLookup,
	}
}

func verboseNaiveBayes(naiveModel NaiveBayesModel, test string) NaiveBayesResult {
	naiveInstances := naiveModel.Weights
	frequencyMap := CreateFrequencyMap(test)
	predictions := make([]NaiveBayesScore, 0)
	standOutFeature := make(map[interface{}][]string)

	for _, naiveInstance := range naiveInstances {
		prob := naiveInstance.Prob
		for key, amp := range frequencyMap {
			if _, exists := naiveModel.SynonymsLookup[key]; exists {
				key = naiveModel.SynonymsLookup[key]
			}
			if naiveInstance.Features[key] > 0 {
				prob *= naiveInstance.Features[key] * float64(amp)
				standOutFeature[naiveInstance.Class] = append(standOutFeature[naiveInstance.Class], key)
			} else {
				// mismatch penalty
				prob *= 0.02
			}
		}
		if len(standOutFeature[naiveInstance.Class]) == 0 {
			prob = 0
		}

		stringClass := fmt.Sprintf("%v", naiveInstance.Class)
		prob = Bound(prob*float64(len(naiveInstances)), 0, 1)
		predictions = append(predictions, NaiveBayesScore{
			Class: stringClass,
			Score: prob,
		})

	}

	sort.Slice(predictions, func(i, j int) bool {
		return predictions[i].Score > predictions[j].Score
	})

	return NaiveBayesResult{
		Candidate:             test,
		StandOutFeatures:      standOutFeature,
		ClassesScore:          predictions,
		SynonymCheck:          naiveModel.CheckSynonym(test),
		UnsignificantFeatures: naiveModel.CheckUnsignificant(test),
	}
}

func (naiveModel NaiveBayesModel) PredictSimple(test string) (string, float64) {
	predictions := verboseNaiveBayes(naiveModel, test)
	class := predictions.ClassesScore[0].Class
	score := predictions.ClassesScore[0].Score
	return class, score
}

func (naiveModel NaiveBayesModel) Predict(test string) NaiveBayesResult {
	return verboseNaiveBayes(naiveModel, test)
}

func (naiveModel NaiveBayesModel) CheckSynonym(text string) []string {
	freqMap := CreateFrequencyMap(text)
	requiresSynonym := make(map[string]bool)

	for key := range freqMap {
		if naiveModel.FeaturesMap[key] || naiveModel.UnsignificantMap[key] {
			continue
		}

		if _, hasSynonym := naiveModel.SynonymsLookup[key]; hasSynonym {
			continue
		}
		requiresSynonym[key] = true
	}

	requiresSynonymSet := make([]string, 0)

	for key := range requiresSynonym {
		requiresSynonymSet = append(requiresSynonymSet, key)
	}

	return requiresSynonymSet
}

func (naiveModel NaiveBayesModel) CheckUnsignificant(text string) []string {
	freqMap := CreateFrequencyMap(text)
	requiresSynonym := make(map[string]bool)

	for key := range freqMap {
		if naiveModel.UnsignificantMap[key] {
			requiresSynonym[key] = true
		}
	}

	requiresSynonymSet := make([]string, 0)

	for key := range requiresSynonym {
		requiresSynonymSet = append(requiresSynonymSet, key)
	}

	return requiresSynonymSet
}
