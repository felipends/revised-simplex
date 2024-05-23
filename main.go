package main

import (
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"regexp"
	"slices"
	"strconv"
	"strings"

	"github.com/lukpank/go-glpk/glpk"
	"gonum.org/v1/gonum/mat"
)

const (
	//options
	SOLVE    = "1"
	GENERATE = "2"

	//constants for simplex
	BIGM = float64(1)

	//epsilon for precision reasons
	epsilon  = float64(1e-5)
	epsilon2 = float64(1e-6)
)

// generate instance from an mps file using go-mps package
func generateInstanceFromMPS(filename string) {
	lp := glpk.New()
	defer lp.Delete()
	lp.ReadMPS(glpk.MPS_FILE, nil, filename)
	var newInputStr string

	newInputStr += fmt.Sprintln("min")
	//populate obj function
	for c := range lp.NumCols() {
		newInputStr += fmt.Sprintf("%vx%v", lp.ObjCoef(c+1), c+1)
		if c < lp.NumCols()-1 {
			newInputStr += " + "
		}
	}
	newInputStr += "\nst\n"

	//populate constraints
	rowSignal := []string{}
	rowsRhs := []float64{}
	rowsVec := [][]float64{}
	for r := range lp.NumRows() + 1 {
		rowVec := make([]float64, lp.NumCols())
		if r == 0 {
			continue
		}

		idxs, row := lp.MatRow(r)
		for i, v := range idxs {
			if v == 0 {
				continue
			}
			rowVec[v-1] = row[i]
		}
		if lp.RowLB(r) == -math.MaxFloat64 {
			rowSignal = append(rowSignal, "<=")
			rowsRhs = append(rowsRhs, lp.RowUB(r))
		} else if lp.RowUB(r) == math.MaxFloat64 {
			rowsRhs = append(rowsRhs, lp.RowLB(r))
			rowSignal = append(rowSignal, ">=")
		} else {
			rowsRhs = append(rowsRhs, lp.RowLB(r))
			rowSignal = append(rowSignal, "=")
		}
		rowsVec = append(rowsVec, rowVec)
	}

	for i, r := range rowsVec {
		for l := range r {
			newInputStr += fmt.Sprintf("%vx%v", r[l], l+1)
			if l < len(r) {
				newInputStr += " + "
			}
		}
		newInputStr += fmt.Sprintf("%v %v\n", rowSignal[i], rowsRhs[i])
	}

	bdSigns := []string{">=", "<="}
	for c := range lp.NumCols() {
		for idx, s := range bdSigns {
			if idx == 0 && lp.ColLB(c+1) == -math.MaxFloat64 {
				continue
			} else if idx == 1 && lp.ColUB(c+1) == math.MaxFloat64 {
				continue
			}

			for k := range lp.NumCols() {
				if c == k {
					newInputStr += fmt.Sprintf("1x%v", c+1)
				} else {
					newInputStr += fmt.Sprintf("0x%v", c+1)
				}
				if k < lp.NumCols()-1 {
					newInputStr += " + "
				}
			}
			newInputStr += " " + s
			if idx == 0 {
				newInputStr += fmt.Sprintf(" %v\n", lp.ColLB(c+1))
			} else {
				newInputStr += fmt.Sprintf(" %v\n", lp.ColUB(c+1))
			}
		}
	}

	for c := range lp.NumCols() {
		newInputStr += fmt.Sprintf("x%v", c+1)
		if c < lp.NumCols()-1 {
			newInputStr += ","
		}
	}
	newInputStr += " >= 0"

	charBuff := []byte(newInputStr)

	f, err := os.Create("testInput.txt")
	if err != nil {
		panic(err)
	}
	defer f.Close()

	_, err = f.Write(charBuff)
	if err != nil {
		panic(err)
	}

	os.Exit(0)
}

func simplex(initialBase, initialSolution, consCoefs, consRhs, baseCoefs *mat.Dense, objCoefsVec []float64, indexesInBase []int) ([]*mat.Dense, []int) {
	currentSolution := mat.DenseCopyOf(initialSolution)
	currentBase := mat.DenseCopyOf(initialBase)
	var dualSolution *mat.Dense
	iter := 0
	for {
		if iter == 10000 {
			//break
		}
		iter++
		//compute basic solution
		currentSolution.Solve(currentBase, consRhs)
		// csolaux = mat.Formatted(currentSolution, mat.Prefix("     "), mat.Squeeze())
		// fmt.Printf("xB = %v\n", csolaux)
		solutionValue := float64(0)
		for i, v := range indexesInBase {
			solutionValue += currentSolution.At(i, 0) * objCoefsVec[v]
		}
		fmt.Printf("Z = %v\n", solutionValue)
		//compute dual solution values for pricing (p')
		inverseBases := mat.DenseCopyOf(currentBase)
		inverseBases.Inverse(currentBase)

		//cT*B^-1
		var dual mat.Dense
		//baseCoefs == cT
		dual.Mul(baseCoefs, inverseBases)

		// dualaux := mat.Formatted(&dual, mat.Prefix("     "), mat.Squeeze())
		// fmt.Printf("p' = %v\n", dualaux)
		dualSolution = mat.DenseCopyOf(&dual)

		//calculate reduced costs (pricing)
		reducedCosts := make([]float64, len(objCoefsVec))
		choosedI := -1
		flagC := false
		for i, val := range objCoefsVec {
			if slices.Contains(indexesInBase, i) {
				continue
			}
			//pT*Aj
			pa := mat.Dot(dual.RowView(0), consCoefs.ColView(i))
			// laux := mat.Formatted(consCoefs.ColView(i), mat.Prefix("     "), mat.Squeeze())
			// fmt.Printf("aj = %v\n", laux)

			reducedCosts[i] = val - pa
			// fmt.Printf("c_%v = %v\n", i, reducedCosts[i])
			if reducedCosts[i] < -epsilon && !flagC {
				// fmt.Printf("CHOOSED -> c_%v = %v\n", i, reducedCosts[i])
				flagC = true
				choosedI = i
				break
			}
		}

		//optimality condition
		if choosedI == -1 {
			break
		}

		//compute u = Bˆ-1*A_j for pricing
		var u mat.Dense
		u.Mul(inverseBases, consCoefs.ColView(choosedI))
		// uaux := mat.Formatted(&u, mat.Prefix("    "), mat.Squeeze())
		// fmt.Printf("u = %v\n", uaux)

		uNegative := true
		for _, item := range u.RawMatrix().Data {
			if item >= epsilon {
				uNegative = false
				break
			}
		}

		// problem is unbounded
		if uNegative {
			uaux := mat.Formatted(&u, mat.Prefix("    "), mat.Squeeze())
			fmt.Printf("u = %v\n", uaux)
			panic("unbounded")
		}

		//minimal ratio test
		minimalRatio := math.MaxFloat64
		leaveBaseIndex := -1
		for i, item := range u.RawMatrix().Data {
			if item < epsilon2 {
				continue
			}
			if math.Abs(currentSolution.At(i, 0)) < epsilon {
				currentSolution.Set(i, 0, 0)
			}
			iRatio := currentSolution.At(i, 0) / item
			// fmt.Printf("x/u = %v\n%v\n", iRatio, i)
			if iRatio < minimalRatio-epsilon || (iRatio == minimalRatio && indexesInBase[leaveBaseIndex] > indexesInBase[i]) {
				minimalRatio = iRatio
				leaveBaseIndex = i

				// fmt.Printf("LEAVING -> x/u = %v %v\n", iRatio, indexesInBase[i])
			}
		}

		//form new basis by replacing leaveBaseIndex with choosedI
		indexesInBase[leaveBaseIndex] = choosedI
		baseCoefs.Set(0, leaveBaseIndex, objCoefsVec[choosedI])
		// fmt.Println(indexesInBase, objCoefsVec)
		for k := range currentBase.RawMatrix().Rows {
			currentBase.Set(k, leaveBaseIndex, consCoefs.At(k, choosedI))
		}
		// baseaux = mat.Formatted(currentBase, mat.Prefix("    "), mat.Squeeze())
		// fmt.Printf("B = %v\n", baseaux)
		fmt.Printf("-------------------- ITERATION %v ----------------------\n", iter)
		fmt.Printf("-------------------- BASE CHANGE %v -> %v ----------------------\n", indexesInBase[leaveBaseIndex], choosedI)
	}

	return []*mat.Dense{currentSolution, currentBase, dualSolution, baseCoefs}, indexesInBase
}

func main() {
	// option 1 solve instance from file <filename>
	// option 2 generate instance from mps file <filename>
	if len(os.Args) != 3 {
		panic(errors.New("wrong number of arguments. Usage: go run main.go <option (1/2)> <filename>"))
	}

	switch option := os.Args[1]; option {
	case SOLVE:
	case GENERATE:
		generateInstanceFromMPS(os.Args[2])
	}

	fmt.Println(epsilon)
	inputFile, err := os.Open(os.Args[2])
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

	//read instance
	inputLines := strings.Split(string(inputData), "\n")
	var problemSense string
	var varsNames []string
	var varsIndexes []int
	var objCoefsVec []float64
	var consCoefsVec []float64
	var consRhsVec []float64
	numCons := 0
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
		consSlack := 1
		if consRe.Match([]byte(str)) {
			consStrs = strings.Split(str, " <= ")
			fmt.Println(str, slackInCons)
		}

		consRe1 := regexp.MustCompile("(>=)+")
		if consRe1.Match([]byte(str)) {
			consStrs = strings.Split(str, " >= ")
			consSlack = -1
		}

		consRe2 := regexp.MustCompile("( =)+")
		if consRe2.Match([]byte(str)) {
			consStrs = strings.Split(str, " = ")
			consSlack = 0
		}
		consRhsValue, err := strconv.ParseFloat(consStrs[1], 64)
		if err != nil {
			panic(err)
		}

		multicons := float64(1)
		if consRhsValue < 0 {
			multicons = -1
			consRhsValue = -consRhsValue
		}

		slackInCons = append(slackInCons, consSlack*int(multicons))

		consRhsVec = append(consRhsVec, consRhsValue)

		consCoefStrs := strings.Split(consStrs[0], " ")
		coefMultiplier := float64(1)
		for _, coefStr := range consCoefStrs {
			var coef float64
			if coefStr == "-" {
				coefMultiplier = -1 * multicons
				continue
			} else if coefStr == "+" {
				coefMultiplier = 1 * multicons
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

	// dualMatrix := mat.DenseCopyOf(consCoefs.T())

	numVars := len(objCoefsVec)
	var initialBase *mat.Dense
	indexesInBase := []int{}
	//add slack variables to A if necessary to ensure that the problem is in standard form
	var consNoNeedAux []int
	column := 0
	baseIdx := 0
	var baseCoefs *mat.Dense
	idxAux := 0
	for c := range numCons {
		if slackInCons[c] == 0 {
			continue
		} else if slackInCons[c] == 1 {
			consNoNeedAux = append(consNoNeedAux, c)

			baseColumn := make([]float64, numCons)
			baseColumn[c] = 1

			if initialBase == nil {
				initialBase = mat.NewDense(numCons, 1, nil)
				baseCoefs = mat.NewDense(1, 1, nil)
			} else {
				initialBase = mat.DenseCopyOf(initialBase.Grow(0, 1))
				baseCoefs = mat.DenseCopyOf(baseCoefs.Grow(0, 1))
			}

			initialBase.SetCol(baseIdx, baseColumn)
			indexesInBase = append(indexesInBase, numVars+column)
			baseIdx++
		}

		consCoefs = mat.DenseCopyOf(consCoefs.Grow(0, 1))
		consCoefs.Set(c, numVars+column, float64(slackInCons[c]))
		column++
		idxAux++
		objCoefsVec = append(objCoefsVec, 0)
		varsNames = append(varsNames, fmt.Sprintf("x_%v", numVars+idxAux))
		varsIndexes = append(varsIndexes, numVars+c)
	}
	numVars += column
	// originalProblemNumVars := numVars
	originalProblemObjVec := objCoefsVec
	fmt.Println(originalProblemObjVec)
	originalProblemConsCoefs := mat.DenseCopyOf(consCoefs)
	artificialIndexes := []int{}

	if initialBase == nil || !(initialBase.RawMatrix().Cols == initialBase.RawMatrix().Rows && initialBase.RawMatrix().Cols == numCons) {
		//add auxiliary variables
		multi := 1
		if problemSense == "max" {
			multi = -1
		}
		column = 0
		for c := range numCons {
			if slices.Contains(consNoNeedAux, c) {
				continue
			}

			//add aux var to constraint
			consCoefs = mat.DenseCopyOf(consCoefs.Grow(0, 1))
			consCoefs.Set(c, numVars+column, 1)

			//add coeficient to obj function
			objCoefsVec = append(objCoefsVec, float64(multi)*BIGM)
			artificialIndexes = append(artificialIndexes, numVars+column)

			//add aux var to initialBase
			if initialBase == nil {
				initialBase = mat.NewDense(numCons, 1, nil)
				baseCoefs = mat.NewDense(1, 1, nil)
			} else {
				initialBase = mat.DenseCopyOf(initialBase.Grow(0, 1))
				baseCoefs = mat.DenseCopyOf(baseCoefs.Grow(0, 1))
			}

			initialBase.Set(c, baseIdx+column, 1)
			baseCoefs.Set(0, baseIdx+column, float64(multi)*BIGM)
			indexesInBase = append(indexesInBase, numVars+column)
			column++
			varsIndexes = append(varsIndexes, numVars+c)
			varsNames = append(varsNames, fmt.Sprintf("x_%v", numVars+c+1))
		}
		numVars += column
	}

	aaux = mat.Formatted(consCoefs, mat.Prefix("    "), mat.Squeeze())
	fmt.Printf("A = %v\n", aaux)
	fmt.Printf("c = %v\n", objCoefsVec)
	baseaux := mat.Formatted(initialBase, mat.Prefix("    "), mat.Squeeze())
	fmt.Printf("B = %v\n", baseaux)

	//define initial solution xB=b
	var initialSolution mat.Dense
	initialSolution.Solve(initialBase, consRhs)
	solaux := mat.Formatted(&initialSolution, mat.Prefix("     "), mat.Squeeze())
	fmt.Printf("xb = %v\n", solaux)

	//start the two phases simplex method
	//set the obj coeffs for original variables to 0 and maintain the auxiliary variables the same
	newObjCoeffsVec := make([]float64, numVars)
	for i := range newObjCoeffsVec {
		if slices.Contains(artificialIndexes, i) {
			newObjCoeffsVec[i] = objCoefsVec[i]
		}
	}
	fmt.Println(newObjCoeffsVec, objCoefsVec, originalProblemObjVec)
	//solve auxiliary problem
	resultMatrices, indexesInBase := simplex(initialBase, &initialSolution, consCoefs, consRhs, baseCoefs, newObjCoeffsVec, indexesInBase)
	currentSolution := mat.DenseCopyOf(resultMatrices[0])
	currentBase := mat.DenseCopyOf(resultMatrices[1])
	dualSolution := mat.DenseCopyOf(resultMatrices[2])
	baseCoefs = mat.DenseCopyOf(resultMatrices[3])

	//check if the value of auxiliar problem is 0
	originalProblmSolutionValue := float64(0)
	for i, _ := range indexesInBase {
		originalProblmSolutionValue += currentSolution.At(i, 0) * baseCoefs.At(0, i)
		fmt.Println(currentSolution.At(i, 0))
	}
	fmt.Println(originalProblmSolutionValue)

	if math.Abs(originalProblmSolutionValue) > epsilon {
		panic("infeasible!")
	}

	//check if basis has auxiliary variables
	hasAux := false
	for _, item := range indexesInBase {
		if slices.Contains(artificialIndexes, item) {
			hasAux = true
		}
	}
	fmt.Println("has aux", hasAux)
	//drive auxiliary variables from basis by pivoting them
	if hasAux {
		for i, item := range indexesInBase {
			if !slices.Contains(artificialIndexes, item) {
				continue
			}

			inverseBasis := mat.DenseCopyOf(currentBase)
			inverseBasis.Inverse(currentBase)
			pivotMatrix := mat.DenseCopyOf(originalProblemConsCoefs)
			pivotMatrix.Mul(inverseBasis, originalProblemConsCoefs)
			//iterate over pivotMatrix, remove row and aux var if all at(i,item) = 0, replace otherwise
			allZero := true
			for k := range pivotMatrix.RawMatrix().Cols {
				if math.Abs(pivotMatrix.At(i, k)) > epsilon && !slices.Contains(indexesInBase, k) {
					allZero = false
					//replace Bi with Al
					indexesInBase[i] = k
					baseCoefs.Set(0, i, originalProblemObjVec[k])
					currentBase.SetCol(i, mat.DenseCopyOf(originalProblemConsCoefs.ColView(k)).RawMatrix().Data)
					break
				}
			}
			if allZero {
				//remove item from basis
				indexesInBase = append(indexesInBase[:i], indexesInBase[i+1:]...)
				newBasis := *mat.NewDense(currentBase.RawMatrix().Cols-1, currentBase.RawMatrix().Cols-1, nil)
				newBaseCoefs := *mat.NewDense(1, newBasis.RawMatrix().Cols, nil)
				for c := range currentBase.RawMatrix().Cols {
					for r := range currentBase.RawMatrix().Rows {
						if c != i && r != i {
							newBasis.Set(r, c, currentBase.At(r, c))
						}
					}
					if c != i {
						newBaseCoefs.Set(0, c, baseCoefs.At(0, c))
					}
				}
				currentBase = mat.DenseCopyOf(&newBasis)
				baseCoefs = mat.DenseCopyOf(&newBaseCoefs)

				//remove item from A
				newConsCoefs := *mat.NewDense(originalProblemConsCoefs.RawMatrix().Rows-1, originalProblemConsCoefs.RawMatrix().Cols, nil)
				newConsRhs := *mat.NewDense(originalProblemConsCoefs.RawMatrix().Rows-1, 1, nil)
				for r := range originalProblemConsCoefs.RawMatrix().Rows {
					if r == item {
						continue
					}
					for c := range originalProblemConsCoefs.RawMatrix().Cols {
						newConsCoefs.Set(r, c, originalProblemConsCoefs.At(r, c))
					}
					newConsRhs.Set(r, 0, consRhs.At(r, 0))
				}
				originalProblemConsCoefs = mat.DenseCopyOf(&newConsCoefs)
				consRhs = &newConsRhs
			}
		}
	}
	hasAux = false
	//check basis again
	for _, item := range indexesInBase {
		if slices.Contains(artificialIndexes, item) {
			hasAux = true
		}
	}
	fmt.Println("has aux", hasAux)
	//update baseCoefs with the original obj coefs
	for n := range numCons {
		baseCoefs.Set(0, n, objCoefsVec[indexesInBase[n]])
	}
	//apply simplex to the original problem with the found basis
	resultMatrices, indexesInBase = simplex(currentBase, &initialSolution, originalProblemConsCoefs, consRhs, baseCoefs, originalProblemObjVec, indexesInBase)
	currentSolution = resultMatrices[0]
	currentBase = resultMatrices[1]
	dualSolution = resultMatrices[2]

	// indexesInBaseSorted := indexesInBase
	// slices.SortFunc(indexesInBaseSorted, func(i, j int) int {
	// 	return i - j
	// })

	// newBase := mat.DenseCopyOf(currentBase)
	// for _, val := range indexesInBase {
	// 	sortedIndex := slices.Index(indexesInBaseSorted, val)
	// 	newBase.SetCol(sortedIndex, mat.DenseCopyOf(currentBase.ColView(sortedIndex)).RawMatrix().Data)
	// }
	// fmt.Println(indexesInBase)
	// fmt.Println(indexesInBaseSorted)
	// sanityTest := mat.DenseCopyOf(currentSolution)
	// sanityTest.Mul(newBase, consRhs)
	// sanityaux := mat.Formatted(sanityTest, mat.Prefix("     "), mat.Squeeze())
	// fmt.Printf("BX = %v\n", sanityaux)
	// sanityaux2 := mat.Formatted(consRhs, mat.Prefix("    "), mat.Squeeze())
	// fmt.Printf("b = %v\n", sanityaux2)

	fmt.Println(indexesInBase)
	csolaux := mat.Formatted(currentSolution, mat.Prefix("     "), mat.Squeeze())
	fmt.Printf("xB = %v\n", csolaux)

	//vars in solution
	var varsInSolution []string
	solutionValue := float64(0)
	for i, v := range indexesInBase {
		varsInSolution = append(varsInSolution, varsNames[v])
		solutionValue += currentSolution.At(i, 0) * baseCoefs.At(0, i)
	}

	fmt.Println("-------------------- SOLUTION ---------------\nVars in solution =", varsInSolution, "\nZ =", solutionValue)

	// unsortedIndexes := make([]int, len(indexesInBase))
	// copy(unsortedIndexes, indexesInBase)
	// slices.SortFunc(indexesInBase, func(i, j int) int {
	// 	return i - j
	// })
	// fmt.Println(indexesInBase, unsortedIndexes)

	// auxiliarBase := mat.DenseCopyOf(currentBase)
	// auxiliarSolution := mat.DenseCopyOf(currentSolution)
	// for i, val := range indexesInBase {
	// 	auxiliarBase.SetCol(i, mat.DenseCopyOf(consCoefs.ColView(val)).RawMatrix().Data)
	// 	auxiliarSolution.Set(i, 0, currentSolution.At(slices.Index(unsortedIndexes, val), 0))
	// }

	// sanityTest := mat.DenseCopyOf(currentSolution)
	// sanityTest.Mul(auxiliarBase, auxiliarSolution)
	// sanityaux := mat.Formatted(sanityTest, mat.Prefix("     "), mat.Squeeze())
	// fmt.Printf("BX = %v\n", sanityaux)

	// sanityaux2 := mat.Formatted(consRhs, mat.Prefix("    "), mat.Squeeze())
	// fmt.Printf("b = %v\n", sanityaux2)
	//print dual problem
	// fodual := ""
	// for i, c := range consRhsVec {
	// 	fodual += fmt.Sprintf("%vy%v ", c, i+1)
	// 	if i < len(consRhsVec)-1 {
	// 		fodual += "+ "
	// 	}
	// }

	// var matrixDualString []string
	// for i := range dualMatrix.RawMatrix().Rows {
	// 	consString := ""
	// 	for j := range dualMatrix.RawMatrix().Cols {
	// 		consString += fmt.Sprintf("%vy%v ", dualMatrix.At(i, j), j+1)
	// 		if j < dualMatrix.RawMatrix().Cols-1 {
	// 			consString += "+ "
	// 		}
	// 	}
	// 	if problemSense == "max" {
	// 		consString += fmt.Sprintf(">= %v", -1*objCoefsVec[i])
	// 	} else {
	// 		consString += fmt.Sprintf("<= %v", objCoefsVec[i])
	// 	}
	// 	matrixDualString = append(matrixDualString, consString)
	// }
	// fmt.Println("---------- DUAL ---------")
	// if problemSense == "max" {
	// 	fmt.Println("min")
	// } else {
	// 	fmt.Println("max")
	// }
	// fmt.Println(fodual)
	// fmt.Println("st")
	// for _, m := range matrixDualString {
	// 	fmt.Println(m)
	// }
	// dualVarsCons := ""
	// for c := range dualMatrix.RawMatrix().Cols {
	// 	dualVarsCons += fmt.Sprintf("y%v", c+1)
	// 	if problemSense == "min" {
	// 		if slackInCons[c] > 0 {
	// 			dualVarsCons += " <= 0\n"
	// 		} else {
	// 			dualVarsCons += " >= 0\n"
	// 		}
	// 	} else {
	// 		if slackInCons[c] > 0 {
	// 			dualVarsCons += " >= 0\n"
	// 		} else {
	// 			dualVarsCons += " <= 0\n"
	// 		}
	// 	}

	// }
	// fmt.Println(dualVarsCons)
	fmt.Println("Dual solution: ")
	dualaux := mat.Formatted(dualSolution, mat.Prefix("     "), mat.Squeeze())
	fmt.Printf("p' = %v\n", dualaux)

	// fmt.Println("\n--------- Sensitivity analysis ---------")
	// //sensitivity analysis
	// var beta mat.Dense
	// beta.Inverse(currentBase)
	// bounds := [][]float64{}
	// for i := range beta.RawMatrix().Cols {
	// 	minBound := math.Inf(1)
	// 	maxBound := math.Inf(-1)
	// 	for j := range beta.RawMatrix().Rows {
	// 		delta := -currentSolution.At(j, 0) / beta.At(j, i)
	// 		if beta.At(j, i) < 0 && delta < minBound {
	// 			minBound = delta
	// 		}

	// 		if beta.At(j, i) > 0 && delta > maxBound {
	// 			maxBound = delta
	// 		}
	// 	}
	// 	bounds = append(bounds, []float64{minBound, maxBound})
	// }

	// for i, b := range bounds {
	// 	fmt.Println(b[0], ">= delta b", i+1, ">=", b[1])
	// }
}
