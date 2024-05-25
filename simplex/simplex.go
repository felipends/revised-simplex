package simplex

import (
	"fmt"
	"math"
	"slices"

	"gonum.org/v1/gonum/mat"
	"q.log/simplex/model"
)

const (
	epsilon1 = 1e-5
	epsilon2 = 1e-5
)

func AddArtificialVariables(m *model.Model) *mat.Dense {
	basis := mat.NewDense(m.NumRows, m.NumRows, nil)
	for r := range m.NumRows {
		bRowVec := make([]float64, m.NumRows)
		bRowVec[r] = 1
		m.AddCol(bRowVec, 1e5)
		m.AddArtificalVariable()
		basis.SetCol(r, bRowVec)
	}

	return basis
}

func PrintBasis(basis *mat.Dense) {
	caux := mat.Formatted(basis, mat.Prefix("    "), mat.Squeeze())
	fmt.Printf("B = %v\n", caux)
	r, c := basis.Dims()
	fmt.Println(r, c)
}

// Solve solves the model by the revised simplex method
// and returns the last basis
func Solve(m *model.Model, basis *mat.Dense) *mat.Dense {
	currentBasis := mat.DenseCopyOf(basis)
	basisCoefsVec := []float64{}
	basisIndexes := []int{}
	for i, v := range m.V {
		if v.IsBasic {
			basisCoefsVec = append(basisCoefsVec, m.C.At(0, i))
			basisIndexes = append(basisIndexes, i)
		}
	}
	basisCoefs := mat.NewDense(1, m.NumRows, basisCoefsVec)
	iter := 0
	currentSolution := mat.NewDense(m.NumRows, 1, nil)
	for {
		if iter == 10000 {
			//break
		}
		iter++
		//compute B^-1
		inverseBases := mat.NewDense(m.NumRows, m.NumRows, nil)
		inverseBases.Inverse(currentBasis)

		//compute basic solution by solving Bx=b
		currentSolution.Mul(inverseBases, m.B)
		for c := range m.NumCols {
			if m.V[c].IsBasic {
				m.X.Set(c, 0, currentSolution.At(slices.Index(basisIndexes, c), 0))
			}
		}

		m.UpdateVariablesValues()
		fmt.Println()
		solutionValue := float64(0)
		for i, v := range m.V {
			if v.IsBasic {
				solutionValue += v.Value * m.C.RawMatrix().Data[i]
			}
		}
		fmt.Printf("Z = %v\n", solutionValue)
		//compute dual solution values for pricing (p')

		//pT = cbT*B^-1
		var dual mat.Dense
		//baseCoefs is already cbT
		dual.Mul(basisCoefs, inverseBases)

		//calculate reduced costs (pricing)
		reducedCosts := make([]float64, m.NumCols)
		chosedJ := -1
		for j := range m.NumCols {
			if m.V[j].IsBasic {
				continue
			}
			//pT*Aj
			pa := mat.Dot(dual.RowView(0), m.A.ColView(j))
			//c'j = cj - pT*Aj
			reducedCosts[j] = m.C.RawMatrix().Data[j] - pa
			if reducedCosts[j] < -epsilon1 {
				chosedJ = j
				break
			}
		}

		//optimality condition
		if chosedJ == -1 {
			break
		}

		//compute u = Bˆ-1*A_j for pricing
		var u mat.Dense
		u.Mul(inverseBases, m.A.ColView(chosedJ))

		uNegative := true
		for _, item := range u.RawMatrix().Data {
			if item > -epsilon1 {
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
		uVec := u.RawMatrix().Data
		for i := range m.NumRows {
			if uVec[i] < epsilon2 {
				continue
			}
			if currentSolution.At(i, 0) < epsilon2 {
				currentSolution.Set(i, 0, 0)
			}
			iRatio := currentSolution.At(i, 0) / uVec[i]
			if iRatio < minimalRatio-epsilon1 || (leaveBaseIndex != -1 && (math.Abs(math.Abs(iRatio)-math.Abs(minimalRatio)) <= epsilon2 && basisIndexes[leaveBaseIndex] > basisIndexes[i])) {
				minimalRatio = iRatio
				leaveBaseIndex = i
			}
		}

		fmt.Printf("-------------------- BASE CHANGE %v -> %v ----------------------\n", basisIndexes[leaveBaseIndex], chosedJ)
		//form new basis by replacing leaveBaseIndex with chosedJ
		m.V[basisIndexes[leaveBaseIndex]].IsBasic = false
		m.V[basisIndexes[leaveBaseIndex]].Value = 0
		basisIndexes[leaveBaseIndex] = chosedJ
		m.V[chosedJ].IsBasic = true
		basisCoefs.Set(0, leaveBaseIndex, m.C.RawMatrix().Data[chosedJ])
		for k := range currentBasis.RawMatrix().Rows {
			currentBasis.Set(k, leaveBaseIndex, m.A.At(k, chosedJ))
		}
		fmt.Printf("-------------------- ITERATION %v ----------------------\n", iter)
	}

	return currentBasis
}
