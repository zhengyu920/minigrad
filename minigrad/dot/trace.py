def trace(root):
    nodes = []
    edges = []

    def DFS(node):
        if node in nodes:
            return
        nodes.append(node)
        for prev in node.prevs:
            edges.append((prev, node))
            DFS(prev)
    DFS(root)
    return nodes, edges
