package model

import (
	"errors"
	"fmt"

	"gonum.org/v1/gonum/mat"
)

type Variable struct {
	Value        float64
	IsBasic      bool
	IsArtificial bool
}

type Model struct {
	//V variables
	V []*Variable

	//C objective function coefficients
	C *mat.Dense

	//A constraints matrix
	A *mat.Dense

	//B constraints rhs
	B *mat.Dense

	//X values of the variables, i.e. the solution
	X *mat.Dense

	NeedArtificial []int
	SlackIndexes   []int

	NumRows int
	NumCols int
}

func NewModel(numRows, numCols int) *Model {
	return &Model{
		C:       mat.NewDense(1, numCols, nil),
		A:       mat.NewDense(numRows, numCols, nil),
		B:       mat.NewDense(numRows, 1, nil),
		X:       mat.NewDense(numCols, 1, nil),
		NumRows: numRows,
		NumCols: numCols,
	}
}

func (m *Model) SetC(cVec []float64) error {
	if len(cVec) != m.NumCols {
		return errors.New("mismatch number of variables")
	}

	m.C = mat.NewDense(1, m.NumCols, cVec)

	return nil
}

func (m *Model) SetA(aVec []float64) error {
	if len(aVec) != m.NumCols*m.NumRows {
		return errors.New("mismatch number of variables and/or constraints")
	}

	m.A = mat.NewDense(m.NumRows, m.NumCols, aVec)

	return nil
}

func (m *Model) SetB(bVec []float64) error {
	if len(bVec) != m.NumRows {
		return errors.New("mismatch number of constraints")
	}

	m.B = mat.NewDense(m.NumRows, 1, bVec)

	return nil
}

func (m *Model) SetX(xVec []float64) error {
	if len(xVec) != m.NumCols*m.NumRows {
		return errors.New("mismatch number of variables")
	}

	m.X = mat.NewDense(m.NumCols, 1, xVec)

	return nil
}

func (m *Model) AddCol(cVec []float64, coef float64) error {
	if len(cVec) != m.NumRows {
		return errors.New("mismatch number of rows, i.e. wrong len of cVec")
	}

	m.A = mat.DenseCopyOf(m.A.Grow(0, 1))
	m.A.SetCol(m.NumCols, cVec)

	m.C = mat.DenseCopyOf(m.C.Grow(0, 1))
	m.C.Set(0, m.NumCols, coef)

	m.X = mat.DenseCopyOf(m.X.Grow(1, 0))

	m.NumCols++
	return nil
}

func (m *Model) AddRow(rVec []float64, rhs float64) error {
	if len(rVec) != m.NumCols {
		return errors.New("mismatch number of columns, i.e. wrong len of rVec")
	}

	m.A = mat.DenseCopyOf(m.A.Grow(1, 0))
	m.A.SetRow(m.NumRows, rVec)

	m.B = mat.DenseCopyOf(m.B.Grow(1, 0))
	m.A.Set(m.NumRows, 0, rhs)

	m.NumRows++
	return nil
}

func (m *Model) RemoveCol(c int) error {
	if c < 0 || c >= m.NumCols {
		return errors.New("column does not exists")
	}

	auxA := mat.NewDense(m.NumRows, m.NumCols-1, nil)
	auxC := mat.NewDense(1, m.NumCols-1, nil)
	auxX := mat.NewDense(m.NumCols-1, 1, nil)
	for col := range m.NumCols {
		if col == c {
			continue
		}
		for row := range m.NumRows {
			auxA.Set(row, col, m.A.At(row, col))
		}
		auxC.Set(0, col, m.C.At(0, col))
		auxX.Set(col, 0, m.X.At(col, 0))
	}

	m.A = mat.DenseCopyOf(auxA)
	m.C = mat.DenseCopyOf(auxC)
	m.X = mat.DenseCopyOf(auxX)
	m.NumCols--

	return nil
}

func (m *Model) RemoveRow(r int) error {
	if r < 0 || r >= m.NumRows {
		return errors.New("row does not exists")
	}

	auxA := mat.NewDense(m.NumRows, m.NumCols-1, nil)
	auxB := mat.NewDense(m.NumRows-1, 1, nil)
	for row := range m.NumRows {
		if row == r {
			continue
		}
		for col := range m.NumCols {
			auxA.Set(row, col, m.A.At(row, col))
		}
		auxB.Set(row, 0, m.B.At(row, 0))
	}

	m.A = mat.DenseCopyOf(auxA)
	m.B = mat.DenseCopyOf(auxB)
	m.NumRows--

	return nil
}

func (m *Model) MultiplyConstraint(row int, mul float64) error {
	if row < 0 || row >= m.NumRows {
		return errors.New("row does not exists")
	}

	for col := range m.NumCols {
		m.A.Set(row, col, m.A.At(row, col)*mul)
	}
	m.B.Set(row, 0, m.B.At(row, 0)*mul)
	return nil
}

func (m *Model) CreateVariables() {
	m.V = make([]*Variable, m.NumCols)
	for c := range m.NumCols {
		m.V[c] = &Variable{
			Value:        0,
			IsBasic:      false,
			IsArtificial: false,
		}
	}
}

func (m *Model) AddArtificalVariable() {
	v := &Variable{
		Value:        0,
		IsBasic:      true,
		IsArtificial: true,
	}

	m.V = append(m.V, v)
}

func (m *Model) UpdateVariablesValues() {
	for i, v := range m.V {
		v.Value = m.X.At(i, 0)
	}
}

func (m *Model) ModifyOriginalPorblem() {
	for c := range m.NumCols {
		if m.V[c].IsArtificial {
			continue
		}
		m.C.Set(0, c, 0)
	}
}

func (m *Model) PrintC() {
	caux := mat.Formatted(m.C, mat.Prefix("    "), mat.Squeeze())
	fmt.Printf("c = %v\n", caux)
	_, c := m.C.Dims()
	fmt.Println(c)
}

func (m *Model) PrintB() {
	caux := mat.Formatted(m.B, mat.Prefix("    "), mat.Squeeze())
	fmt.Printf("b = %v\n", caux)
	r, _ := m.B.Dims()
	fmt.Println(r)
}

func (m *Model) PrintA() {
	caux := mat.Formatted(m.A, mat.Prefix("    "), mat.Squeeze())
	fmt.Printf("A = %v\n", caux)
	r, c := m.A.Dims()
	fmt.Println(r, c)
}

func (m *Model) PrintSolution() {
	z := float64(0)
	for c := range m.NumCols {
		z += m.V[c].Value * m.C.At(0, c)
	}

	fmt.Printf("Z = %v\n", z)
}
