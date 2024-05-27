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

	m.PrintC()
	m.PrintA()
	m.PrintB()

	initialBasis := simplex.AddArtificialVariables(m)
	simplex.PrintBasis(initialBasis)

	//m.ModifyOriginalPorblem(1)
	auxiliaryBasis := simplex.Solve(m, initialBasis)
	os.Exit(0)

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
	m.ModifyOriginalPorblem(2)

	index := 0
	for i, v := range m.V {
		if !v.IsArtificial {
			if v.IsBasic {
				colData := mat.DenseCopyOf(m.A.ColView(i)).RawMatrix().Data
				newBasis.SetCol(index, colData)
				index++
			}
		}
	}
	_ = simplex.Solve(m, newBasis)
}
