package gonlpclassifier

import "github.com/kniren/gota/dataframe"

type RawTraining struct {
	Text   string
	Action string
}

type NaiveBayesInstance struct {
	Class    interface{}
	Prob     float64
	Features map[interface{}]float64
}

type NaiveBayesModel struct {
	Weights          []NaiveBayesInstance
	Features         []string
	FeaturesMap      map[string]bool
	UnsignificantMap map[string]bool
	Synonyms         map[string][]string
	SynonymsLookup   map[string]string
	Classes          []interface{}
}

type NaiveBayesScore struct {
	Class string
	Score float64
}

type NaiveBayesResult struct {
	ClassesScore          []NaiveBayesScore
	StandOutFeatures      map[interface{}][]string
	Candidate             string
	SynonymCheck          []string
	UnsignificantFeatures []string
}

type NaiveBayesDataset struct {
	TFIDF             []map[string]interface{}
	Columns           []string
	ClassName         string
	Unsignificant     map[string]bool
	SignificanceScore float64
	Dataframe         dataframe.DataFrame
}
