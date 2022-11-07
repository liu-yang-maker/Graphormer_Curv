
import networkx as nx

def no_surgery(G_origin, *args, **kwargs):
    return G_origin

def surgery(G_origin, weight='weight', cut_proportion=0.03):
    G = G_origin.copy()
    w = nx.get_edge_attributes(G, weight)

    assert cut_proportion >= 0 and cut_proportion <= 1, "Cut proportion should be in [0, 1]"

    sorted_edges = sorted(w.items(), key=lambda x:x[1])
    #to_cut = []
    #for e, e_w in sorted_edges[int(len(sorted_edges) * (1 - cut_proportion)):]:
    #    if e_w < 0:
    #        to_cut.append(e)
    to_cut = [e for (e, e_w) in sorted_edges[int(len(sorted_edges) * (1 - cut_proportion)):]]
    print("*************** Surgery time ****************")
    print("* Cut %d edges." % len(to_cut))
    G.remove_edges_from(to_cut)
    print("* Number of nodes now: %d" % G.number_of_nodes())
    print("* Number of edges now: %d" % G.number_of_edges())

    return G

def surgery_n(G_origin, weight='weight', cut_n=1):
    G = G_origin.copy()
    w = nx.get_edge_attributes(G, weight)

    assert cut_n >= 0 and cut_n <= G.number_of_edges(), "Cut proportion should be in [0, 1]"

    sorted_edges = sorted(w.items(), key=lambda x:x[1])
    # to_cut = []
    # for e, e_w in sorted_edges[-cut_n:]:
    #     if e_w < 0:
    #         to_cut.append(e)
    to_cut = [e for (e, e_w) in sorted_edges[-cut_n:]]
    G.remove_edges_from(to_cut)
    return G