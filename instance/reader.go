package instance

import (
	"fmt"
	"math"
	"runtime"

	"github.com/lukpank/go-glpk/glpk"
	"q.log/simplex/model"
)

// Reader reads a mps file to construct a model
type Reader struct {
	filename string
}

func NewReader(filename string) *Reader {
	return &Reader{
		filename: filename,
	}
}

// ConstructModelFromFile returns a *Model in standard form
func (r *Reader) ConstructModelFromFile() *model.Model {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	lp := glpk.New()
	defer lp.Delete()
	lp.ReadMPS(glpk.MPS_FILE, nil, r.filename)

	m := model.NewModel(lp.NumRows(), lp.NumCols())
	fmt.Println(lp.NumCols())

	//populate obj function
	var cVec []float64
	for c := range lp.NumCols() + 1 {
		if c == 0 {
			continue
		}
		cVec = append(cVec, lp.ObjCoef(c))
	}
	m.SetC(cVec)

	//populate constraints
	aVec := []float64{}
	rowsRhs := []float64{}
	rowSignal := []string{}
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
		aVec = append(aVec, rowVec...)
	}

	err := m.SetA(aVec)
	if err != nil {
		panic(err)
	}

	err = m.SetB(rowsRhs)
	if err != nil {
		panic(err)
	}

	bdSigns := []string{">=", "<="}
	for c := range lp.NumCols() {
		for idx, s := range bdSigns {
			if idx == 0 && (lp.ColLB(c+1) == -math.MaxFloat64 || lp.ColLB(c+1) == 0) {
				continue
			} else if idx == 1 && (lp.ColUB(c+1) == math.MaxFloat64 || lp.ColUB(c+1) == 0) {
				continue
			}

			rowVec := make([]float64, lp.NumCols())
			rowRhs := float64(0)

			rowVec[c] = 1
			rowSignal = append(rowSignal, s)
			if idx == 0 {
				rowRhs = lp.ColLB(c + 1)
			} else {
				rowRhs = lp.ColUB(c + 1)
			}
			m.AddRow(rowVec, rowRhs)
		}
	}

	//pass the model to standard form
	for r := range m.NumRows {
		//adds slack and surplus variables
		if rowSignal[r] == "=" {
			m.NeedArtificial = append(m.NeedArtificial, r)
			continue
		} else if rowSignal[r] == "<=" {
			colVec := make([]float64, m.NumRows)
			colVec[r] = 1
			m.SlackIndexes = append(m.SlackIndexes, m.NumCols)
			m.AddCol(colVec, 0)
		} else if rowSignal[r] == ">=" {
			m.SlackIndexes = append(m.SlackIndexes, m.NumCols)
			colVec := make([]float64, m.NumRows)
			colVec[r] = -1
			m.AddCol(colVec, 0)
		}
	}

	for r := range m.NumRows {
		if m.B.At(r, 0) < 0 {
			m.MultiplyConstraint(r, -1)
		}
	}

	m.CreateVariables()

	return m
}
