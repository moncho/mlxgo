package deepseek

import (
	"fmt"

	mlx "github.com/moncho/mlxgo"
	"github.com/moncho/mlxgo/deepseek/quant"
)

// Called on the worker. Publish all banks and replacement views together so a
// failed preparation leaves the model and any existing sessions usable.
func (m *Model) prepareDeviceExperts() (err error) {
	if len(m.deviceExperts) == m.config.Layers {
		return nil
	}
	c := m.config
	packed := make([]bool, c.Layers)
	for layer := range packed {
		count := 0
		for id := -1; id < c.Experts; id++ {
			if m.fp4Experts[expertPrefix(ExpertID{Layer: layer, Index: id})] != nil {
				count++
			}
		}
		if count != 0 && count != c.Experts+1 {
			return fmt.Errorf("deepseek: device decode requires homogeneous expert storage in layer %d", layer)
		}
		packed[layer] = count > 0
	}
	banks := make(map[int]*DeviceMoE, c.Layers)
	floatViews := make(map[string]mlx.Array)
	packedViews := make(map[string]*FP4Expert)
	defer func() {
		if err == nil {
			return
		}
		for _, bank := range banks {
			bank.Close()
		}
		for _, a := range floatViews {
			a.Close()
		}
		for _, p := range packedViews {
			p.Close()
		}
	}()
	for layer := 0; layer < c.Layers; layer++ {
		if packed[layer] {
			bank, e := m.packedDeviceBank(layer)
			if e != nil {
				return deviceMoEError("prepare decode bank", e)
			}
			banks[layer] = bank
			for id := 0; id < c.Experts; id++ {
				p := &FP4Expert{dim: c.Dim, inter: c.InterDim}
				packedViews[expertPrefix(ExpertID{Layer: layer, Index: id})] = p
				for part, dst := range []**quant.FP4Linear{&p.gate, &p.up, &p.down} {
					*dst, e = []*quant.FP4LinearBank{bank.packedGate, bank.packedUp, bank.packedDown}[part].Linear(id)
					if e != nil {
						return deviceMoEError("prepare packed expert view", e)
					}
				}
			}
		} else {
			expert := func(id int) ExpertWeights {
				p := expertPrefix(ExpertID{Layer: layer, Index: id})
				return ExpertWeights{Gate: m.weights[p+"w1.weight"], Up: m.weights[p+"w3.weight"], Down: m.weights[p+"w2.weight"]}
			}
			experts := make([]ExpertWeights, c.Experts)
			for id := range experts {
				experts[id] = expert(id)
			}
			bank, e := NewDeviceMoE(experts, expert(-1))
			if e != nil {
				return e
			}
			banks[layer] = bank
			for id := 0; id < c.Experts; id++ {
				prefix := expertPrefix(ExpertID{Layer: layer, Index: id})
				for part, a := range []mlx.Array{bank.gate, bank.up, bank.down} {
					view, e := floatExpertView(a, id)
					if e != nil {
						return deviceMoEError("prepare float32 expert view", e)
					}
					floatViews[prefix+[]string{"w1.weight", "w3.weight", "w2.weight"}[part]] = view
				}
			}
		}
	}
	// Views reference the evaluated bank buffers. Drop the old routed weights;
	// shared weights already use independent handles to the same allocation.
	for name, a := range floatViews {
		old := m.weights[name]
		old.Close()
		m.weights[name] = a
	}
	for name, p := range packedViews {
		m.fp4Experts[name].Close()
		m.fp4Experts[name] = p
	}
	m.deviceExperts = banks
	return nil
}

func floatExpertView(bank mlx.Array, index int) (mlx.Array, error) {
	return one(func(s *scope) mlx.Array {
		shape := bank.Shape()
		a := s.add(mlx.Slice(bank, []int{index, 0, 0}, []int{index + 1, shape[1], shape[2]}, []int{1, 1, 1}))
		a = s.add(mlx.SqueezeAxis(a, 0))
		return s.add(mlx.Transpose(a))
	})
}

func (m *Model) packedDeviceBank(layer int) (*DeviceMoE, error) {
	c := m.config
	bank := &DeviceMoE{dim: c.Dim, inter: c.InterDim, count: c.Experts}
	err := func() error {
		for part, dst := range []**quant.FP4LinearBank{&bank.packedGate, &bank.packedUp, &bank.packedDown} {
			linears := make([]*quant.FP4Linear, c.Experts)
			for id := range linears {
				p := m.fp4Experts[expertPrefix(ExpertID{Layer: layer, Index: id})]
				linears[id] = []*quant.FP4Linear{p.gate, p.up, p.down}[part]
			}
			var err error
			*dst, err = quant.StackFP4Linears(linears)
			if err != nil {
				return err
			}
		}
		shared := m.fp4Experts[expertPrefix(ExpertID{Layer: layer, Index: -1})]
		bank.packedShared = &FP4Expert{dim: c.Dim, inter: c.InterDim}
		for i, dst := range []**quant.FP4Linear{&bank.packedShared.gate, &bank.packedShared.up, &bank.packedShared.down} {
			var err error
			*dst, err = []*quant.FP4Linear{shared.gate, shared.up, shared.down}[i].Retain()
			if err != nil {
				return err
			}
		}
		return nil
	}()
	if err != nil {
		bank.Close()
		return nil, err
	}
	return bank, nil
}
