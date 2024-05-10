package main

import (
	"fmt"
	"io"
	"math"
	"os"
	"regexp"
	"slices"
	"strconv"
	"strings"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/combin"
)

func main() {
	inputFile, err := os.Open(os.Args[1])
	if err != nil {
		panic(err)
	}

	inputData, err := io.ReadAll(inputFile)
	if err != nil {
		panic(err)
	}
	err = inputFile.Close()
	if err != nil {
		panic(err)
	}

	inputLines := strings.Split(string(inputData), "\n")
	var problemSense string
	var varsNames []string
	var varsIndexes []int
	var objCoefsVec []float64
	var consCoefsVec []float64
	var consRhsVec []float64
	numCons := 0
	needSlack := false
	var slackInCons []int
	for i, str := range inputLines {
		if i == 0 {
			problemSense = str
			fmt.Println(problemSense)
			continue
		}

		if i == 1 {
			inputCoefs := strings.Split(str, " ")
			multiplier := float64(1)
			idx := 0
			for _, coefStr := range inputCoefs {
				var coef float64
				if coefStr == "-" {
					multiplier = -1
					continue
				} else if coefStr == "+" {
					multiplier = 1
					continue
				}
				re := regexp.MustCompile("[a-z]+")
				coefStr2 := re.Split(coefStr, -1)
				fmt.Println(coefStr2)
				coef, err = strconv.ParseFloat(coefStr2[0], 64)
				if err != nil {
					panic(err)
				}
				varsNames = append(varsNames, "x"+coefStr2[1])
				objMultiplier := 1
				if problemSense == "max" {
					objMultiplier = -1
				}
				objCoefsVec = append(objCoefsVec, coef*multiplier*float64(objMultiplier))
				varsIndexes = append(varsIndexes, idx)
				idx++
			}
			continue
		}

		if i == 2 {
			continue
		}

		if i == len(inputLines)-1 {
			continue
		}

		numCons++

		var consStrs []string
		consRe := regexp.MustCompile("(<=)+")
		if consRe.Match([]byte(str)) {
			consStrs = strings.Split(str, " <= ")
			slackInCons = append(slackInCons, 1)

			fmt.Println(str, slackInCons)
		}

		consRe1 := regexp.MustCompile("(>=)+")
		if consRe1.Match([]byte(str)) {
			consStrs = strings.Split(str, " >= ")
			slackInCons = append(slackInCons, -1)

			fmt.Println(str, needSlack)
		}

		consRe2 := regexp.MustCompile("( =)+")
		if consRe2.Match([]byte(str)) {
			consStrs = strings.Split(str, " = ")
			slackInCons = append(slackInCons, 0)

			fmt.Println(str, needSlack)
		}
		//TODO: if rhs is negative, multiply row by -1
		consRhsValue, err := strconv.ParseFloat(consStrs[1], 64)
		if err != nil {
			panic(err)
		}
		consRhsVec = append(consRhsVec, consRhsValue)

		consCoefStrs := strings.Split(consStrs[0], " ")
		coefMultiplier := float64(1)
		for _, coefStr := range consCoefStrs {
			var coef float64
			if coefStr == "-" {
				coefMultiplier = -1
				continue
			} else if coefStr == "+" {
				coefMultiplier = 1
				continue
			}
			re := regexp.MustCompile("[a-z]+")
			coefStr2 := re.Split(coefStr, -1)
			coef, err = strconv.ParseFloat(coefStr2[0], 64)
			if err != nil {
				panic(err)
			}
			consCoefsVec = append(consCoefsVec, coef*coefMultiplier)
		}
	}

	//print var names
	fmt.Printf("x = %v\n", varsNames)
	//define c vector
	objCoefs := mat.NewDense(1, len(objCoefsVec), objCoefsVec)
	caux := mat.Formatted(objCoefs, mat.Prefix("    "), mat.Squeeze())
	fmt.Printf("c = %v\n", caux)
	//define A matrix
	consCoefs := mat.NewDense(numCons, len(objCoefsVec), consCoefsVec)
	aaux := mat.Formatted(consCoefs, mat.Prefix("    "), mat.Squeeze())
	fmt.Printf("A = %v\n", aaux)
	//define b vector
	consRhs := mat.NewDense(len(consRhsVec), 1, consRhsVec)
	baux := mat.Formatted(consRhs, mat.Prefix("    "), mat.Squeeze())
	fmt.Printf("b = %v\n", baux)

	dualMatrix := mat.DenseCopyOf(consCoefs.T())

	//add slack variables to A and define B, and indexes in base
	var indexesInBase []int
	consCoefs = mat.DenseCopyOf(consCoefs.Grow(0, numCons))
	numVars := len(objCoefsVec)
	for c := range numCons {
		consCoefs.Set(c, numVars+c, float64(slackInCons[c]))
		objCoefsVec = append(objCoefsVec, 0)
		varsNames = append(varsNames, fmt.Sprintf("x_%v", numVars+c+1))
		varsIndexes = append(varsIndexes, numVars+c)
	}
	aaux = mat.Formatted(consCoefs, mat.Prefix("    "), mat.Squeeze())
	fmt.Printf("A = %v\n", aaux)
	fmt.Printf("c = %v\n", objCoefsVec)

	//define initial base
	baseCoefs := mat.NewDense(1, numCons, nil)
	initialBase := mat.NewDense(numCons, numCons, nil)
	indexesInBase = append(indexesInBase, numVars+1)
	possiblePermutations := combin.Combinations(len(varsIndexes), numCons)
	for _, p := range possiblePermutations {
		initialIdexesInBase := []int{}
		for i, v := range p {
			initialBase.SetCol(i, mat.DenseCopyOf(consCoefs.ColView(v)).RawMatrix().Data)
			initialIdexesInBase = append(initialIdexesInBase, v)
		}

		//solve Bx=b to discover an initial feasible basis
		var xb mat.Dense
		err := xb.Solve(initialBase, consRhs)
		if err != nil {
			continue
		}

		allPositve := true
		for _, x := range xb.RawMatrix().Data {
			if x < 0 {
				allPositve = false
				break
			}
		}
		fmt.Println(p)
		if allPositve {
			indexesInBase = initialIdexesInBase
			for i, v := range indexesInBase {
				baseCoefs.Set(0, i, objCoefsVec[v])
			}
			break
		}
	}

	fmt.Printf("x in base = %v\n", indexesInBase)
	baseaux := mat.Formatted(initialBase, mat.Prefix("    "), mat.Squeeze())
	fmt.Printf("B = %v\n", baseaux)

	//define initial solution xB=b
	var initialSolution mat.Dense
	initialSolution.Solve(initialBase, consRhs)
	solaux := mat.Formatted(&initialSolution, mat.Prefix("     "), mat.Squeeze())
	fmt.Printf("xb = %v\n", solaux)

	//start the algorithm main loop
	currentBase := mat.DenseCopyOf(initialBase)
	currentSolution := mat.DenseCopyOf(&initialSolution)
	var csolaux fmt.Formatter
	var dualSolution mat.Dense
	for {
		//compute basic solution
		currentSolution.Solve(currentBase, consRhs)
		csolaux = mat.Formatted(currentSolution, mat.Prefix("     "), mat.Squeeze())
		fmt.Printf("xB = %v\n", csolaux)

		//compute dual solution values for pricing (p')
		var dual mat.Dense
		dual.Solve(currentBase.T(), baseCoefs.T())
		dual = *mat.DenseCopyOf(dual.T())

		dualaux := mat.Formatted(&dual, mat.Prefix("     "), mat.Squeeze())
		fmt.Printf("p' = %v\n", dualaux)
		dualSolution = *mat.DenseCopyOf(&dual)

		//calculate reduced costs (pricing)
		reducedCosts := make([]float64, len(objCoefsVec))
		choosedI := -1
		for i, val := range objCoefsVec {
			if slices.Contains(indexesInBase, i) {
				continue
			}
			pa := mat.Dot(dual.RowView(0), consCoefs.ColView(i))
			laux := mat.Formatted(consCoefs.ColView(i), mat.Prefix("     "), mat.Squeeze())
			fmt.Printf("aj = %v\n", laux)

			reducedCosts[i] = val - pa
			fmt.Printf("c_%v = %v\n", i, reducedCosts[i])
			if reducedCosts[i] < 0 {
				choosedI = i
				break
			}
		}

		//optimality condition
		if choosedI == -1 {
			break
		}

		//compute u = Bˆ-1A_j for pricing
		var u mat.Dense
		u.Solve(currentBase, consCoefs.ColView(choosedI))
		uaux := mat.Formatted(&u, mat.Prefix("    "), mat.Squeeze())
		fmt.Printf("u = %v\n", uaux)

		uNegative := true
		for item := range u.RawMatrix().Data {
			if item > 0 {
				uNegative = false
				break
			}
		}

		// problem is unlimited
		if uNegative {
			break
		}

		//minimal ratio test
		minimalRatio := math.Inf(1)
		leaveBaseIndex := -1
		for i, item := range u.RawMatrix().Data {
			if item <= 0 {
				continue
			}

			iRatio := currentSolution.At(i, 0) / item
			if iRatio < minimalRatio {
				minimalRatio = iRatio
				leaveBaseIndex = i
			}
		}

		//form new basis by replacing leaveBaseIndex with choosedI
		indexesInBase[leaveBaseIndex] = choosedI
		baseCoefs.Set(0, leaveBaseIndex, objCoefsVec[choosedI])
		fmt.Println(indexesInBase)
		for k := range currentBase.RawMatrix().Rows {
			currentBase.Set(k, leaveBaseIndex, consCoefs.At(k, choosedI))
		}
		baseaux = mat.Formatted(currentBase, mat.Prefix("    "), mat.Squeeze())
		fmt.Printf("B = %v\n", baseaux)
	}

	fmt.Println(indexesInBase)
	fmt.Printf("xB = %v\n", csolaux)

	//vars in solution
	var varsInSolution []string
	solutionValue := float64(0)
	for i, v := range indexesInBase {
		varsInSolution = append(varsInSolution, varsNames[v])
		solutionValue += currentSolution.At(i, 0) * objCoefsVec[v]
	}

	fmt.Println("Vars in solution =", varsInSolution, "Z =", solutionValue)

	//print dual problem
	fodual := ""
	for i, c := range consRhsVec {
		fodual += fmt.Sprintf("%vy%v ", c, i+1)
		if i < len(consRhsVec)-1 {
			fodual += "+ "
		}
	}

	var matrixDualString []string
	for i := range dualMatrix.RawMatrix().Rows {
		consString := ""
		for j := range dualMatrix.RawMatrix().Cols {
			consString += fmt.Sprintf("%vy%v ", dualMatrix.At(i, j), j+1)
			if j < dualMatrix.RawMatrix().Cols-1 {
				consString += "+ "
			}
		}
		if problemSense == "max" {
			consString += fmt.Sprintf(">= %v", -1*objCoefsVec[i])
		} else {
			consString += fmt.Sprintf("<= %v", objCoefsVec[i])
		}
		matrixDualString = append(matrixDualString, consString)
	}
	fmt.Println("---------- DUAL ---------")
	if problemSense == "max" {
		fmt.Println("min")
	} else {
		fmt.Println("max")
	}
	fmt.Println(fodual)
	fmt.Println("st")
	for _, m := range matrixDualString {
		fmt.Println(m)
	}
	dualVarsCons := ""
	for c := range dualMatrix.RawMatrix().Cols {
		dualVarsCons += fmt.Sprintf("y%v", c+1)
		if problemSense == "min" {
			if slackInCons[c] > 0 {
				dualVarsCons += " <= 0\n"
			} else {
				dualVarsCons += " >= 0\n"
			}
		} else {
			if slackInCons[c] > 0 {
				dualVarsCons += " >= 0\n"
			} else {
				dualVarsCons += " <= 0\n"
			}
		}

	}
	fmt.Println(dualVarsCons)
	fmt.Println("Dual solution: ")
	dualaux := mat.Formatted(&dualSolution, mat.Prefix("     "), mat.Squeeze())
	fmt.Printf("p' = %v\n", dualaux)

	fmt.Println("\n--------- Sensitivity analysis ---------")
	//sensitivity analysis
	var beta mat.Dense
	beta.Inverse(currentBase)
	bounds := [][]float64{}
	for i := range beta.RawMatrix().Cols {
		minBound := math.Inf(1)
		maxBound := math.Inf(-1)
		for j := range beta.RawMatrix().Rows {
			delta := -currentSolution.At(j, 0) / beta.At(j, i)
			if beta.At(j, i) < 0 && delta < minBound {
				minBound = delta
			}

			if beta.At(j, i) > 0 && delta > maxBound {
				maxBound = delta
			}
		}
		bounds = append(bounds, []float64{minBound, maxBound})
	}

	for i, b := range bounds {
		fmt.Println(b[0], ">= delta b", i+1, ">=", b[1])
	}
}
