import graphviz
from . import trace


def generate_graph(root):
    nodes, edges = trace.trace(root)

    def node_label(node):
        return f'{node.label} | {str(node)} | grad: {node.grad}'

    def node_ops_id(node):
        return f'{node.id()}_{node.ops}'

    dot = graphviz.Digraph()
    dot.attr(rankdir='BT')
    for node in nodes:
        dot.node(node.id(), node_label(node), shape='record')
        if node.ops != '':
            dot.node(node_ops_id(node), node.ops, shape='oval')
            dot.edge(node_ops_id(node), node.id())

    for edge in edges:
        dot.edge(edge[0].id(), edge[1].id() if edge[1].ops ==
                 '' else node_ops_id(edge[1]))

    return dot
