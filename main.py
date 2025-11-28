# visualize_dag.py
import copy
from graphviz import Digraph

from utils.viz import make_gif
from dataloader import trainloader, testloader
from rl.classification_trainer import NetworkTrainer
from model.net_graph import NetGraph
from model.dag_executor import DAGExecutor
from rl.a2c_agent import A2CAgent
from rl.a2c_trainer import A2CTrainer
import torch
import torch.nn as nn
import torch.nn.functional as F


g = NetGraph()
# --- Nodes ---
D = 32

# ---- Nodes ----
inp    = g.add_node(lambda x: x,           name="Input")
flat   = g.add_node(nn.Flatten(),          name="Flatten")
toD    = g.add_node(nn.Linear(28*28, D),   name="InProj")
actIn  = g.add_node(F.relu,                name="ReLU_In", is_act=True)

# --- Simple path (1 layer) ---
s1     = g.add_node(nn.Linear(D, D),       name="S1")
s1a    = g.add_node(F.relu,                name="S1_ReLU", is_act=True)

# --- Medium path (2 layers) ---
m1     = g.add_node(nn.Linear(D, D),       name="M1")
m1a    = g.add_node(F.relu,                name="M1_ReLU", is_act=True)
m2     = g.add_node(nn.Linear(D, D),       name="M2")
m2a    = g.add_node(F.relu,                name="M2_ReLU", is_act=True)

# --- Large path (5 layers) ---
l1     = g.add_node(nn.Linear(D, D),       name="L1")
l1a    = g.add_node(F.relu,                name="L1_ReLU", is_act=True)
l2     = g.add_node(nn.Linear(D, D),       name="L2")
l2a    = g.add_node(F.relu,                name="L2_ReLU", is_act=True)
l3     = g.add_node(nn.Linear(D, D),       name="L3")
l3a    = g.add_node(F.relu,                name="L3_ReLU", is_act=True)
l4     = g.add_node(nn.Linear(D, D),       name="L4")
l4a    = g.add_node(F.relu,                name="L4_ReLU", is_act=True)
l5     = g.add_node(nn.Linear(D, D),       name="L5")
l5a    = g.add_node(F.relu,                name="L5_ReLU", is_act=True)

# --- Merge (all D-dim) ---
add_sm = g.add_node(torch.add,             name="Add_Simple_Med")
add_all= g.add_node(torch.add,             name="Add_All")

# (Optional) Head that keeps D-dim (remove if you want the merged D as output)
head   = g.add_node(nn.Identity(),         name="HeadIdent")

# ---- Connections ----
g.add_connection(inp, flat)
g.add_connection(flat, toD)
g.add_connection(toD, actIn)

# Fan-out to the three paths from actIn
# Simple
g.add_connection(actIn, s1)
g.add_connection(s1, s1a)

# Medium (2 layers)
g.add_connection(actIn, m1)
g.add_connection(m1, m1a)
g.add_connection(m1a, m2)
g.add_connection(m2, m2a)

# Large (5 layers)
g.add_connection(actIn, l1)
g.add_connection(l1, l1a)
g.add_connection(l1a, l2)
g.add_connection(l2, l2a)
g.add_connection(l2a, l3)
g.add_connection(l3, l3a)
g.add_connection(l3a, l4)
g.add_connection(l4, l4a)
g.add_connection(l4a, l5)
g.add_connection(l5, l5a)

# Merge: (Simple + Medium) + Large
g.add_connection(s1a, add_sm)
g.add_connection(m2a, add_sm)
g.add_connection(add_sm, add_all)
g.add_connection(l5a, add_all)

# Final head (still D-dim)
g.add_connection(add_all, head)

# IO marks
g.mark_inputs([inp])
g.mark_outputs([head])


g.mark_decision_nodes()



non_rl_g = copy.deepcopy(g)
rl_g = copy.deepcopy(g)

# run the non-rl -- get acc
non_rl_exe = DAGExecutor(non_rl_g)
agent = A2CAgent(D, 64, 3) 
trainer = NetworkTrainer(trainloader, testloader, agent, non_rl_g, non_rl_exe)
exe = DAGExecutor(g)
trainer.fit(5)

print()
# run the best-path of non-rl -- get acc
# run the rl -- get acc
agent = A2CAgent(D, 64, 3)
exe = DAGExecutor(g)
trainer = A2CTrainer(trainloader, testloader, agent, g, exe, entropy_coef=0.0)
trainer.fit(5)


