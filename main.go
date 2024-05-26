package main

import (
	"os"

	"gonum.org/v1/gonum/mat"
	"q.log/simplex/instance"
	"q.log/simplex/simplex"
)

func main() {
	filename := os.Args[1]
	r := instance.NewReader(filename)
	m := r.ConstructModelFromFile()

	//originalProblem
	om := r.ConstructModelFromFile()

	m.PrintC()
	m.PrintA()
	m.PrintB()

	initialBasis := simplex.AddArtificialVariables(m)
	simplex.PrintBasis(initialBasis)

	m.ModifyOriginalPorblem()
	auxiliaryBasis := simplex.Solve(m, initialBasis)
	hasArtificial := false
	for _, v := range m.V {
		if v.IsArtificial && v.IsBasic {
			hasArtificial = true
			break
		}
	}

	newBasis := mat.DenseCopyOf(auxiliaryBasis)
	if hasArtificial {
		newBasis = simplex.DriveOutArtificialVars(m, auxiliaryBasis)
	}
	index := 0
	for i, v := range m.V {
		if !v.IsArtificial {
			om.V[i].IsBasic = v.IsBasic
			if v.IsBasic {
				colData := mat.DenseCopyOf(om.A.ColView(i)).RawMatrix().Data
				newBasis.SetCol(index, colData)
				index++
			}
		}
	}
	_ = simplex.Solve(om, newBasis)
}
