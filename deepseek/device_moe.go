package deepseek

import (
	"fmt"
	"log"
	"math"

	mlx "github.com/moncho/mlxgo"
	"github.com/moncho/mlxgo/deepseek/quant"
)

// DeviceMoE owns a homogeneous routed-expert bank and one shared expert.
// Routing indices remain MLX arrays throughout Forward. This opt-in inference
// component can be used directly or through SessionOptions.DeviceDecode.
// Float32 activations are used with either float32 or weight-only MXFP4 weights.
// Calls are serialized on the MLX worker. Close invalidates copies, while
// previously returned lazy outputs retain their dependencies.
type DeviceMoE struct {
	gate, up, down                   mlx.Array
	shared                           ExpertWeights
	packedGate, packedUp, packedDown *quant.FP4LinearBank
	packedShared                     *FP4Expert
	dim, inter, count                int
	closed                           bool
}

// NewDeviceMoE stacks equally shaped float32 routed experts once and retains
// independent handles for the shared expert. Inputs remain caller-owned and may
// be closed after construction. Source and stacked buffers coexist while the
// bank is prepared; only bank/shared handles are retained by the result.
func NewDeviceMoE(experts []ExpertWeights, shared ExpertWeights) (*DeviceMoE, error) {
	if len(experts) == 0 {
		return nil, fmt.Errorf("deepseek: empty device expert bank")
	}
	m := &DeviceMoE{count: len(experts)}
	err := mlx.Batch(func() error {
		s := &scope{}
		s.check(experts[0].Gate, "expert gate", -1, -1)
		if s.err != nil {
			return s.err
		}
		shape := experts[0].Gate.Shape()
		m.inter, m.dim = shape[0], shape[1]
		parts := [3][]mlx.Array{}
		for _, e := range experts {
			s.check(e.Gate, "expert gate", m.inter, m.dim)
			s.check(e.Up, "expert up", m.inter, m.dim)
			s.check(e.Down, "expert down", m.dim, m.inter)
			parts[0] = append(parts[0], e.Gate)
			parts[1] = append(parts[1], e.Up)
			parts[2] = append(parts[2], e.Down)
		}
		s.check(shared.Gate, "shared gate", m.inter, m.dim)
		s.check(shared.Up, "shared up", m.inter, m.dim)
		s.check(shared.Down, "shared down", m.dim, m.inter)
		if s.err != nil {
			return s.err
		}
		for i, dst := range []*mlx.Array{&m.gate, &m.up, &m.down} {
			stacked, err := mlx.Stack(parts[i])
			if err != nil {
				return err
			}
			*dst, err = mlx.TransposeAxes(stacked, []int{0, 2, 1})
			stacked.Close()
			if err != nil {
				return err
			}
		}
		for i, dst := range []*mlx.Array{&m.shared.Gate, &m.shared.Up, &m.shared.Down} {
			a := []mlx.Array{shared.Gate, shared.Up, shared.Down}[i]
			var err error
			*dst, err = mlx.Reshape(a, a.Shape())
			if err != nil {
				return err
			}
		}
		return mlx.Eval(m.gate, m.up, m.down, m.shared.Gate, m.shared.Up, m.shared.Down)
	})
	if err != nil {
		m.Close()
		return nil, deviceMoEError("prepare float32 expert bank", err)
	}
	return m, nil
}

// NewFP4DeviceMoE copies packed row-scaled E2M1/E8M0 expert weights into a bank.
// dim and inter must be positive multiples of 32. No full decoded weight matrix
// is retained. Constructors validate every weight group for nonfinite values.
func NewFP4DeviceMoE(experts []FP4ExpertWeights, shared FP4ExpertWeights, dim, inter int) (*DeviceMoE, error) {
	if len(experts) == 0 || dim <= 0 || inter <= 0 || dim%32 != 0 || inter%32 != 0 {
		return nil, fmt.Errorf("deepseek: invalid FP4 device expert dimensions")
	}
	m := &DeviceMoE{count: len(experts), dim: dim, inter: inter}
	err := mlx.Batch(func() error {
		// Prepare one projection at a time to bound temporary packed storage.
		for part, dst := range []**quant.FP4LinearBank{&m.packedGate, &m.packedUp, &m.packedDown} {
			bank, err := stackFP4Projection(experts, part, dim, inter)
			if err != nil {
				return err
			}
			*dst = bank
		}
		var err error
		m.packedShared, err = NewFP4Expert(shared, dim, inter)
		return err
	})
	if err != nil {
		m.Close()
		return nil, deviceMoEError("prepare FP4 expert bank", err)
	}
	return m, nil
}

func stackFP4Projection(experts []FP4ExpertWeights, part, dim, inter int) (*quant.FP4LinearBank, error) {
	linears := make([]*quant.FP4Linear, 0, len(experts))
	defer func() {
		for _, l := range linears {
			l.Close()
		}
	}()
	rows, cols := inter, dim
	if part == 2 {
		rows, cols = dim, inter
	}
	for _, expert := range experts {
		w := []FP4Weight{expert.Gate, expert.Up, expert.Down}[part]
		l, err := quant.NewFP4Linear(w.Data, w.Scales, rows, cols)
		if err != nil {
			return nil, err
		}
		linears = append(linears, l)
	}
	return quant.StackFP4Linears(linears)
}

// Forward returns caller-owned lazy [tokens,dim] activations from x [tokens,dim].
// gateWeight [experts,dim] and gateBias [experts] are borrowed. Router semantics
// match MoE, including tie ordering and correction bias. Each selected expert
// sees only its assigned token rows; the shared expert always sees every token.
// Positive limit clips the gate above and up projection on both sides, and
// routing amplitudes are applied before the down projection. No tensor values
// are read by Go. This path is intended for inference, not gradient training.
func (m *DeviceMoE) Forward(x, gateWeight, gateBias mlx.Array, c RouterConfig, limit float32) (mlx.Array, error) {
	out, err := one(func(s *scope) mlx.Array {
		if m == nil || m.closed {
			s.err = fmt.Errorf("closed device expert bank")
			return mlx.Array{}
		}
		if limit < 0 || math.IsNaN(float64(limit)) || math.IsInf(float64(limit), 0) {
			s.err = fmt.Errorf("invalid SwiGLU limit")
			return mlx.Array{}
		}
		s.check(x, "device expert input", -1, m.dim)
		s.check(gateWeight, "device router weight", m.count, m.dim)
		weights, indices := route(s, x, gateWeight, gateBias, c)
		if s.err != nil {
			return mlx.Array{}
		}
		tokens, k := x.Shape()[0], c.TopK
		input := s.add(mlx.Reshape(x, []int{tokens, 1, m.dim}))
		rows := s.add(mlx.ArangeDType(0, float64(tokens), 1, mlx.Int32))
		rows = s.add(mlx.Reshape(rows, []int{tokens, 1}))
		var gate, up mlx.Array
		if m.packedGate != nil {
			gate = s.add(m.packedGate.Gather(input, rows, indices))
			up = s.add(m.packedUp.Gather(input, rows, indices))
		} else {
			gate = s.add(mlx.GatherMM(input, m.gate, rows, indices, false))
			up = s.add(mlx.GatherMM(input, m.up, rows, indices, false))
		}
		amplitudes := s.add(mlx.Reshape(weights, []int{tokens, k, 1, 1}))
		h := expertActivation(s, gate, up, amplitudes, limit)
		rows = s.add(mlx.ArangeDType(0, float64(tokens*k), 1, mlx.Int32))
		rows = s.add(mlx.Reshape(rows, []int{tokens, k}))
		var selected mlx.Array
		if m.packedDown != nil {
			selected = s.add(m.packedDown.Gather(h, rows, indices))
		} else {
			selected = s.add(mlx.GatherMM(h, m.down, rows, indices, false))
		}
		routed := s.add(mlx.SumAxis(selected, 1, false))
		routed = s.add(mlx.Reshape(routed, []int{tokens, m.dim}))
		var shared mlx.Array
		if m.packedShared != nil {
			shared = s.add(m.packedShared.Forward(x, s.add(mlx.Ones([]int{tokens, 1}, mlx.Float32)), limit))
		} else {
			shared = expert(s, x, m.shared, s.scalar(1), limit)
		}
		return s.add(mlx.Add(routed, shared))
	})
	if err != nil {
		return mlx.Array{}, deviceMoEError("evaluate device experts", err)
	}
	return out, nil
}

func deviceMoEError(action string, err error) error {
	log.Printf("deepseek: %s: %v", action, err)
	return fmt.Errorf("deepseek: %s failed", action)
}

// Close releases owned weights and is idempotent.
func (m *DeviceMoE) Close() error {
	if m == nil {
		return nil
	}
	return mlx.Batch(func() error {
		if m.closed {
			return nil
		}
		m.closed = true
		err := mlx.CloseArrays([]mlx.Array{m.gate, m.up, m.down, m.shared.Gate, m.shared.Up, m.shared.Down})
		for _, p := range []*quant.FP4LinearBank{m.packedGate, m.packedUp, m.packedDown} {
			if e := p.Close(); err == nil {
				err = e
			}
		}
		if e := m.packedShared.Close(); err == nil {
			err = e
		}
		if err != nil {
			return deviceMoEError("close device experts", err)
		}
		return nil
	})
}
