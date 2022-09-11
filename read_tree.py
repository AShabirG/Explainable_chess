import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

g_leela = nx.read_gml('tree_10.gml', label='id')
g_maia = nx.read_gml('tree_9.gml', label='id')
g_test = nx.read_gml('tree_1.gml', label='id')


# Find the move most visited in the tree. This does not mean it is the best move sequence (although it could be).
# Not a very useful tool but could be handy when seeking explanations as to why it may have missed a better move.
def most_visited_move(graph):
    number_visited = []
    for i in range(len(graph) - 1):
        number_visited.append(graph.nodes[i + 1]['N'])
    most_visited_index = max(range(len(number_visited)), key=number_visited.__getitem__)
    return graph.nodes[most_visited_index + 1]['move'], most_visited_index + 1


def out_nodes_from_source(graph):
    out_nodes = [a[1] for a in list(graph.out_edges(0))]
    return out_nodes


def suggested_move(graph, move_index):
    """

    :param graph: Graph to use
    :param move: Move index to find the follow up for
    :return: Suggested follow-up. Both move and index of move
    """
    response_to_move = list(graph.out_edges(move_index))
    if not response_to_move:
        return 0, 0
    eval_of_responses = []
    for i in range(len(response_to_move)):
        eval_of_responses.append(float(graph.nodes[response_to_move[i][1]]['Q']))
    best_q = max(range(len(eval_of_responses)), key=eval_of_responses.__getitem__)
    best_move_index = response_to_move[best_q][1]
    best_move = graph.nodes[best_move_index]['move']
    return best_move, best_move_index


def continuation(graph, move_index):
    move_1, move_1_index = suggested_move(graph, move_index)
    move_2, move_2_index = suggested_move(graph, move_1_index)
    move_3, _ = suggested_move(graph, move_2_index)
    return move_1, move_2, move_3


# Generators python
def weak_engines_move_for_strong_move(graph, strong_graph, move):
    """
    Function that takes in the move of the strong engine and figures out what the weaker engine thinks 
    is the best continuation.
    :param graph: Graph of weaker engine
    :param move: Move in leela form. eg. "e4e5"
    :return: Nothing yet
    """
    for i in range(len(graph) - 1):
        if graph.nodes[i]['move'] == move:
            parent_node = list(graph.in_edges(i))[0][0]
            connections_of_parent = list(graph.out_edges(parent_node))
            max_Q = -1
            for j in range(len(connections_of_parent)):
                if float(graph.nodes[connections_of_parent[j][1]]['Q']) >= max_Q:
                    max_Q = float(graph.nodes[connections_of_parent[j][1]]['Q'])
                    max_Q_index = connections_of_parent[j][1]
            if graph.nodes[max_Q_index]['move'] == move:
                print(f"The suggested move from both engines are identical. Best move:{move}")
                move_1, move_2, move_3 = continuation(strong_graph, max_Q_index)
                print(f"Suggested continuation by leela:{move} {move_1} {move_2} {move_3}")
                move_1, move_2, move_3 = continuation(graph, max_Q_index)
                print(f"Suggested continuation by weaker engine:{move} {move_1} {move_2} {move_3}")

            else:
                weak_move = graph.nodes[max_Q_index]['move']
                print(f"Leela suggests playing {move} in this position whereas the weaker engine suggests {weak_move}.")
                move_1, move_2, move_3 = continuation(graph, i)
                print(
                    f"The weak engines continuation of the actual best move ({move}) is: {move_1}, {move_2}, {move_3}")
                move_1, move_2, move_3 = continuation(strong_graph, 1)
                print(
                    f"The strong engines continuation of the best move ({move}) is: {move_1}, {move_2}, {move_3}")
                print("Insert explanation here")
            return 0
            # weak_engine_suggestions = list(graph.out_edges(max_Q_index))


move_t, move_index = most_visited_move(g_leela)
# print(move_t)
# response, best_move_index = suggested_move(g_test, move_index)
# response, best_move_index = suggested_move(g_test, best_move_index)
# print(most_visited_move(g_test))
weak_engines_move_for_strong_move(g_maia, g_leela, 'a2a1')


def number_of_leaves(graph):
    breadth = 0
    for i in range(len(graph)):
        if graph.out_degree(i) == 0:
            breadth += 1
    return breadth


def eval_of_first_moves(graph):
    first_layer_nodes = out_nodes_from_source(graph)
    total_Q = 0
    for i in first_layer_nodes:
        total_Q += float(graph.nodes[i]['Q']) + 1
    return total_Q / len(first_layer_nodes)


p = 12
leaves_maia = []
num_of_nodes_from_source_maia = []
Q_val_maia = []
while p < 31:
    g_maia = nx.read_gml(f'tree_{p}.gml', label='id')
    num_of_nodes_from_source_maia.append(len(out_nodes_from_source(g_maia)))
    leaves_maia.append(number_of_leaves(g_maia))
    Q_val_maia.append(eval_of_first_moves(g_maia))
    p += 1

l = 31
leaves_leela = []
num_of_nodes_from_source_leela = []
Q_val_leela = []
while l < 50:
    g_leela = nx.read_gml(f"tree_{l}.gml", label='id')
    num_of_nodes_from_source_leela.append(len(out_nodes_from_source(g_leela)))
    leaves_leela.append(number_of_leaves(g_leela))
    Q_val_leela.append(eval_of_first_moves(g_leela))
    l += 1


def average(list):
    return sum(list) / len(list)


print(f'Leela nodes: {average(num_of_nodes_from_source_leela)}')
print(f'Maia nodes: {average(num_of_nodes_from_source_maia)}')
print(f'Leaves maia: {average((leaves_maia))}')
print(f'leaves Leela: {average(leaves_leela)}')
print(f'Average Q-values of first layer of leela: {average(Q_val_leela) / 19}')
print(f'Average Q-values of first layer of maia: {average(Q_val_maia) / 19}')

X = np.arange(1, 20, 1)
Y = num_of_nodes_from_source_maia
Z = num_of_nodes_from_source_leela

X_axis = np.arange(len(X))

plt.bar(X_axis - 0.2, Y, 0.4, label='Maia')
plt.bar(X_axis + 0.2, Z, 0.4, label='Leela')

plt.xticks(X_axis, X)
plt.xlabel("Graph")
plt.ylabel("Number of nodes in first layer")
plt.title("Number of moves explored from original position")
plt.legend()
plt.show()

X = np.arange(1, 20, 1)
Y = leaves_maia
Z = leaves_leela

X_axis = np.arange(len(X))

plt.bar(X_axis - 0.2, Y, 0.4, label='Maia')
plt.bar(X_axis + 0.2, Z, 0.4, label='Leela')

plt.xticks(X_axis, X)
plt.xlabel("Graph")
plt.ylabel("Number of leaves")
plt.title("Leaves for each graph")
plt.legend()
plt.show()

X = np.arange(1, 20, 1)
Y = Q_val_maia
Z = Q_val_leela

X_axis = np.arange(len(X))

# plt.bar(X_axis - 0.2, Y, 0.4, label='Maia')
# plt.bar(X_axis + 0.2, Z, 0.4, label='Leela')
plt.plot(X_axis, Y, label='Maia')
plt.plot(X_axis, Z, label='Leela')
plt.xticks(X_axis, X)
plt.xlabel("Graph")
plt.ylabel("Average evaluation range of 0 to 2")
plt.title("Average evaluation of moves in the first layer (+1)")
plt.legend()
plt.show()
