from graphviz import Digraph

def trace(root):
  # create sets of graph nodes and edges
  nodes, edges = set(), set()
  def build(vertex):
    if vertex not in nodes:
      nodes.add(vertex)
      for child in vertex._prev:
        edges.add((child, vertex))
        build(child)
  build(root)
  return nodes, edges

def draw_dot(root):
  # LR -> left to right
  dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})
  
  nodes, edges = trace(root)
  for node in nodes:
    uid = str(id(node))
    # create rectangular nodes for values
    dot.name(name = uid, label = "{ %s | data %.4f }" % (node.label, node.data), shape='record')
    if node._op:
      dot.node(name = uid + node._op, label = node._op)
      dot.edge(uid + node._op, uid)
      
  for node1, node2 in edges:
    dot.edge(str(id(node1)), str(id(node2)) + node2._op)
    
  return dot