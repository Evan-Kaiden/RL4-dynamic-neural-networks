
import torch
import torch.nn as nn

from model.net_graph import NetGraph
from rl.a2c_agent import A2CAgent

class DAGExecutor(nn.Module):
    def __init__(self, graph:NetGraph):
        super().__init__()
        self.graph = graph
        self._decision_nodes = graph.decision_nodes
        self._inputs = graph.inputs()
        self._order = graph.topological_order()
        self._module_map = {}
        self._stateless = {}

        self.temps = {}
        self.path_decisions = {}
        self.prev_nid = None
        self.cur_nid = self._order[0]
        self.cur_idx = 0

        self.target = None

        self.graph_size = len(graph.layers)

        for nid, node in graph.layers.items():
            if isinstance(node.op, nn.Module):
                attr_name = str(node)
                setattr(self, attr_name, node.op)
                self._module_map[nid] = attr_name
            else:
                self._stateless[nid] = node.op

    

    def _run_until_decision_point(self):
        first_run = True
        while self.cur_idx < len(self._order):
            nid = self.cur_nid
            

            if nid in self._inputs:
                self.cur_idx += 1
                if self.cur_idx >= len(self._order): return
                self.prev_nid = self._order[self.cur_idx - 1]
                self.cur_nid  = self._order[self.cur_idx]
                continue

            parents = [p for p in self.graph.ins[nid] if p in self.temps]
            if not parents:
                self.cur_idx += 1
                if self.cur_idx >= len(self._order): return
                self.prev_nid = self._order[self.cur_idx - 1]
                self.cur_nid  = self._order[self.cur_idx]
                continue

            all_parents = self.graph.ins[nid]
            decision_parent = [p for p in all_parents if p in self.path_decisions]
            blocked = (decision_parent == parents) and (nid not in self.path_decisions[decision_parent[0]])
            if blocked:
                self.cur_idx += 1
                if self.cur_idx >= len(self._order): return
                self.prev_nid = self._order[self.cur_idx - 1]
                self.cur_nid  = self._order[self.cur_idx]
                continue

            args = [self.temps[p] for p in parents]

            if nid in self._stateless:
                op = self._stateless[nid]
                if len(args) > 1:
                    y = op(*args)
                elif self.graph.layers[nid].is_act:
                    y = op(args[0])
                else:
                    y = args[0]
            elif nid in self._module_map:
                mod = getattr(self, self._module_map[nid])
              
                y = mod(*args) if len(args) > 1 else mod(args[0])
            else:
                raise KeyError(f"No op found for node {nid}")
    
            self.temps[nid] = y
            layer = self.graph.layers[nid]
            if hasattr(y, "shape"):
                layer.shape_out = tuple(y.shape)

            if self.cur_nid in self._decision_nodes and not first_run:
                return

            self.cur_idx += 1
            if self.cur_idx >= len(self._order): return
            self.prev_nid = self._order[self.cur_idx - 1]
            self.cur_nid  = self._order[self.cur_idx]

            first_run = False

    def reset_target(self, target : torch.tensor):
        self.target = target
    

    def reset_input(self, *inputs):
        self.temps = {}
        self.path_decisions = {}
        self.prev_nid = None
        self.cur_nid = self._order[0]
        self.cur_idx = 0

        gins = self._inputs
        if len(inputs) != len(gins):
            raise ValueError(f"Expected {len(gins)} inputs, got {len(inputs)}")
        for nid, x in zip(gins, inputs):
            self.temps[nid] = x

    
        self._run_until_decision_point()
        context = self.temps[self.cur_nid]
        n_actions = len(self.graph.outs[self.cur_nid])
        done = self._order[-1] in self.temps.keys()
        return (context, n_actions), done

    def step(self, action : int):
        chosen_child = self.graph.outs[self.cur_nid][action]
        self.path_decisions[self.cur_nid] = [chosen_child]

        temps_start_size = len(self.temps)
        self._run_until_decision_point()
        temps_end_size = len(self.temps)
        step_nodes = temps_end_size - temps_start_size
        done = self._order[-1] in self.temps.keys()
        if done:
            preds = [self.temps[nid] for nid in self.graph.outputs()][0]
            pred = torch.argmax(preds)
            correct = pred == self.target
            reward = torch.tensor(-float(step_nodes / self.graph_size) + (1.0 if correct else 0.0),
                      dtype=torch.float32, device=preds.device)
            return (None, 0), reward, True
        context = self.temps[self.cur_nid]
        n_actions = len(self.graph.outs[self.cur_nid])
        reward = torch.tensor(-float(step_nodes / self.graph_size), dtype=torch.float32, device=context.device)
        return (context, n_actions), reward, False

    def forward(self, agent : A2CAgent, *inputs):
        temps = {}
        path_decisions = {}
        gins = self._inputs
        logps = []
        values = []

        if len(inputs) != len(gins):
            raise ValueError(f"Expected {len(gins)} inputs, got {len(inputs)}")
        for nid, x in zip(gins, inputs):
            temps[nid] = x


        for nid in self._order:
            if nid in gins: continue
        
            parents = [pid for pid in self.graph.ins[nid] if pid in temps]

            if not parents: continue

            all_parents = self.graph.ins[nid]
            decision_parent = [p for p in all_parents if p in path_decisions]

            if decision_parent == parents and nid not in path_decisions[decision_parent[0]]:
                continue

            args = [temps[parent] for parent in parents]

            if nid in self._stateless:
                op = self._stateless[nid]
                if len(args) > 1: y = op(*args)
                elif self.graph.layers[nid].is_act: y = op(args[0])
                else: y = args[0]

            elif nid in self._module_map:
                mod = getattr(self, self._module_map[nid])
                y = mod(*args) if len(args) > 1 else mod(args[0])
            else:
                raise KeyError(f"No op found for node {nid}")

            temps[nid] = y
            layer = self.graph.layers[nid]
            if hasattr(y, "shape"):
                layer.shape_out = tuple(y.shape)

            if nid in self._decision_nodes:
                n_actions = len(self.graph.outs[nid])
    
                action_logits, value = agent.act(y, n_actions)

                dist = torch.distributions.Categorical(logits=action_logits)
                action = dist.sample()
                logp = dist.log_prob(action)

                # Log values
                logps.append(logp)
                values.append(value)
                
                chosen_child = self.graph.outs[nid][action.item()]
                path_decisions[nid] = [chosen_child]
        
        outs = [temps[nid] for nid in self.graph.outputs()]

        return (outs[0], path_decisions, temps, logps, values) if len(outs) == 1 else (tuple(outs), path_decisions, temps, logps, values)