package gonlpclassifier

import (
	"math"
	"regexp"
	"sort"
	"strings"

	"github.com/kniren/gota/dataframe"
)

var re = regexp.MustCompile("(?m)[^a-zA-Z ]+")

func CreateFrequencyMap(text string) map[string]int {
	trim := strings.Trim(text, " ")
	asciiOnly := re.ReplaceAllString(trim, "")
	lower := strings.ToLower(asciiOnly)
	split := strings.Split(lower, " ")
	frequency := make(map[string]int)

	for _, w := range split {
		frequency[w]++
	}
	return frequency
}

func BuildNaiveDataset(dataset []string, idfSignificance float64, className string, classes []string) NaiveBayesDataset {
	tfDoc := make([]map[string]int, len(dataset))
	wordSetMap := make(map[string]int)
	tfGlobal := make(map[string]int)
	unsignificantMap := make(map[string]bool)

	for i, x := range dataset {
		trim := strings.Trim(x, " ")
		asciiOnly := re.ReplaceAllString(trim, "")
		lower := strings.ToLower(asciiOnly)
		split := strings.Split(lower, " ")
		frequency := make(map[string]int)

		for _, w := range split {
			frequency[w]++
			wordSetMap[w] = 1
		}

		tfDoc[i] = frequency
	}

	for _, doc := range tfDoc {
		for word := range doc {
			tfGlobal[word]++
		}
	}

	idf := make(map[string]float64)
	columns := make([]string, 0)

	for word := range wordSetMap {
		idfScore := math.Log10(float64(len(dataset)) / float64(1.0+tfGlobal[word]))
		if idfScore < idfSignificance {
			unsignificantMap[word] = true
			continue
		}
		idf[word] = idfScore
		columns = append(columns, word)
	}

	sort.Strings(columns)

	tfIdfDoc := make([]map[string]interface{}, len(dataset))

	for i := range tfDoc {
		tfIdf := make(map[string]interface{})
		for _, word := range columns {
			tfIdf[word] = float64(tfDoc[i][word]) * idf[word]
		}
		tfIdfDoc[i] = tfIdf
	}

	for i := range tfIdfDoc {
		tfIdfDoc[i][className] = classes[i]
	}

	return NaiveBayesDataset{
		TFIDF:             tfIdfDoc,
		Columns:           columns,
		ClassName:         className,
		Unsignificant:     unsignificantMap,
		SignificanceScore: idfSignificance,
		Dataframe:         dataframe.LoadMaps(tfIdfDoc),
	}
}
