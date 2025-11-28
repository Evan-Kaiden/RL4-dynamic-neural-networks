import os
import torch
from PIL import Image
from io import BytesIO
from graphviz import Digraph


def make_gif(agent, exe, graph, input, filename='dag.gif'):
    
    def act(obs):
        action_logits, *_ = agent.forward(*obs)
        dist = torch.distributions.Categorical(logits=action_logits)
        action = dist.sample()
        return action.item()
    
    frames = []
    
    initial = True
    done = False
    while not done:
        if initial:
            obs, done = exe.reset_input(input)
            initial = False
        else:
            obs, _, done = exe.step(act(obs))
        nodes_ran = exe.temps.keys()
        f = Digraph('DAG', filename='dag.gv', format='png')

        for nid, layer in graph.layers.items():
            name = str(layer) 
            ran = nid in nodes_ran
            if ran:
                f.node(name, shape='box', style='bold') 
            else:
                f.node(name, shape='box', style='dotted')
            for other in graph.outs[nid]:
                if ran and other in nodes_ran:
                    if nid in exe.path_decisions and other in exe.path_decisions[nid]:
                        f.edge(name, str(graph.layers[other]), color='black')
                    elif nid not in exe.path_decisions:
                        f.edge(name, str(graph.layers[other]), color='black')
                    else:
                        f.edge(name, str(graph.layers[other]), color='grey', style='dotted')
                else:
                    f.edge(name, str(graph.layers[other]), color='grey', style='dotted')
        png_bytes = f.pipe()
        pil_image = Image.open(BytesIO(png_bytes))
        frames.append(pil_image)

        output_path = os.path.join('dags', filename)
        output_dir, _ = os.path.split(output_path)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],  
            duration=1000,  
            loop=0
        )

def make_graph(graph, filename='dag.png'):
    f = Digraph('DAG', filename='dag.gv', format='png')

    for nid, layer in graph.layers.items():
        name = str(layer) 
        f.node(name, shape='box', style='bold') 
        for other in graph.outs[nid]:
                f.edge(name, str(graph.layers[other]))

    png_bytes = f.pipe()
    pil_image = Image.open(BytesIO(png_bytes))

    output_path = os.path.join('dags', filename)
    output_dir, _ = os.path.split(output_path)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    pil_image.save(output_path)



def make_path_graph(agent, exe, graph, input, filename='dag.png'):
    
    def act(obs):
        action_logits, *_ = agent.forward(*obs)
        dist = torch.distributions.Categorical(logits=action_logits)
        action = dist.sample()
        return action.item()
        
    initial = True
    done = False
    while not done:
        if initial:
            obs, done = exe.reset_input(input)
            initial = False
        else:
            obs, _, done = exe.step(act(obs))

    nodes_ran = exe.temps.keys()
    f = Digraph('DAG', filename='dag.gv', format='png')

    for nid, layer in graph.layers.items():
        name = str(layer) 
        ran = nid in nodes_ran
        if ran:
            f.node(name, shape='box', style='bold') 
        else:
            f.node(name, shape='box', style='dotted')
        for other in graph.outs[nid]:
            if ran and other in nodes_ran:
                if nid in exe.path_decisions and other in exe.path_decisions[nid]:
                    f.edge(name, str(graph.layers[other]), color='black')
                elif nid not in exe.path_decisions:
                    f.edge(name, str(graph.layers[other]), color='black')
                else:
                    f.edge(name, str(graph.layers[other]), color='grey', style='dotted')
            else:
                f.edge(name, str(graph.layers[other]), color='grey', style='dotted')
    png_bytes = f.pipe()
    pil_image = Image.open(BytesIO(png_bytes))

    output_path = os.path.join('dags', filename)
    output_dir, _ = os.path.split(output_path)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    pil_image.save(output_path)