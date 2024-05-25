package main

import (
	"os"

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
	_ = simplex.Solve(m, initialBasis)

	m.PrintSolution()
}
