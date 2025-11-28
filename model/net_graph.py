import torch.nn as nn

from dataclasses import dataclass
from collections import deque, defaultdict



@dataclass
class Layer:
    id: int
    op: nn.Module        # nn.Module or callable
    name: str
    is_act : bool        
    shape_out = None

    def __str__(self):
        if self.shape_out is not None:
            return f"{self.name}_{self.id} {tuple(self.shape_out)}"
        return f"{self.name}_{self.id}"
    
class NetGraph:
    """
    Maintains DAG of ops (nodes) + connections (edges).
    """
    def __init__(self):
        self.layers = {}                 # id -> Layer
        self.ins = defaultdict(list)     # id -> [parent ids]
        self.outs = defaultdict(list)    # id -> [child ids]
        self.next_id = 0
        self.decision_nodes = []         # Nodes with outgoing degree = 2
        self._inputs = []                # ids explicitly marked as graph inputs
        self._outputs = []               # ids explicitly marked as graph outputs

    def __repr__(self):
        return f"NetGraph(num_nodes={len(self.layers)})"

    def add_node(self, op, name=None, is_act=False):
        nid = self.next_id
        self.next_id += 1
        if name is None:
            if isinstance(op, nn.Module):
                name = op.__class__.__name__
            else:
                name = getattr(op, "__name__", "lambda")
        self.layers[nid] = Layer(nid, op, name, is_act)
        return nid

    def add_connection(self, i, j):
        assert i in self.layers and j in self.layers and i != j
        if j not in self.outs[i]:
            self.outs[i].append(j)
            self.ins[j].append(i)

    def remove_connection(self, i, j):
        assert i in self.layers and j in self.layers and i != j
        self.outs[i].remove(j)
        self.ins[j].remove(i)

    def mark_decision_nodes(self):
        decision_size = None
        for nid, outs in self.outs.items():
            if len(outs) >= 2:
                output_shape = getattr(self.layers[nid], "shape_out")
                if output_shape is not None:
                    if decision_size is not None:
                        assert output_shape == decision_size, 'Inconsistent dimensions along decision nodes'
                    else:
                        decision_size = output_shape
                elif hasattr(self.layers[nid].op, "output_shape"):
                    output_shape = getattr(self.layers[nid].op, "output_shape")
                    if decision_size is not None:
                        assert output_shape == decision_size, 'Inconsistent dimensions along decision nodes'
                    else:
                        decision_size = output_shape

                self.decision_nodes.append(nid)

    def mark_inputs(self, ids):
        self._inputs = list(ids)

    def mark_outputs(self, ids):
        self._outputs = list(ids)

    def inputs(self):
        if self._inputs:
            return self._inputs
        return [i for i in self.layers.keys() if len(self.ins[i]) == 0]

    def outputs(self):
        if self._outputs:
            return self._outputs
        return [i for i in self.layers.keys() if len(self.outs[i]) == 0]

    def topological_order(self):
        indeg = {i: len(self.ins[i]) for i in self.layers}
        q = deque(sorted([i for i, d in indeg.items() if d == 0]))
        order = []
        while q:
            u = q.popleft()
            order.append(u)
            for v in self.outs[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)
        if len(order) != len(self.layers):
            raise ValueError("Graph contains a cycle")
        return order
    
    def parameters(self):
        for layer in self.layers.values():
            op = layer.op
            if hasattr(op, 'parameters'):
                yield from op.parameters()

    def to(self, device, dtype=None):
        for layer in self.layers.values():
            op = layer.op
            if isinstance(op, nn.Module):
                op.to(device=device, dtype=dtype)
        return self