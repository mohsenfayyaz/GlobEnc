import networkx as nx
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
import datasets
from transformers import BertTokenizer
import pickle
from itertools import *
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow
from multiprocessing import Pool, Manager
from p_tqdm import p_map
from src.attention_flow_abstract import AttentionFlow


class AttentionMaxFlow(AttentionFlow):
    def __init__(self):
        self.output_hidden_states = False  # Return flow up to all layers (True) or just up to the last layer (False)

    def compute_flows(self, attentions_list, desc="", output_hidden_states=False, num_cpus=4):
        self.output_hidden_states = output_hidden_states
        if num_cpus < 2:
            print("Using 1 core...")
            attentions_max_flows = []
            for i in tqdm(range(len(attentions_list)), desc=desc):
                attentions_max_flows.append(self.compute_one_max_flow(attentions_list[i]))
        else:
            print(f"Using p_map {num_cpus} cores...")
            attentions_max_flows = p_map(self.compute_one_max_flow, attentions_list,
                                         **{"num_cpus": num_cpus})
        return attentions_max_flows

    def compute_one_max_flow(self, att_mat):
        """
        :param att_mat: (#layers, sentence_len, sentence_len)
        :return: max flow matrix (sentence_len, sentence_len)
        """
        res_att_mat = self.pre_process(att_mat)

        res_adj_mat = AttentionMaxFlow.mats_to_adjmat(res_att_mat)
        flow_values = self.compute_max_flows(res_adj_mat, att_mat.shape[1])
        flow_attentions = AttentionMaxFlow.adjmat_to_mats(flow_values, n_layers=att_mat.shape[0], l=att_mat.shape[-1])

        # nx_graph = mats_to_graph(res_att_mat)
        # flow_attentions = compute_max_flow_nx(nx_graph, layer=len(att_mat), num_nodes=att_mat.shape[1])
        # print(att_mat.shape)
        if self.output_hidden_states:
            return flow_attentions
        else:
            return flow_attentions[[-1]]

    def compute_max_flows(self, nodes, length, do_normalize=True, approx=10000000):
        graph = csr_matrix(np.round(nodes * approx).astype(np.int64))
        number_of_nodes = len(nodes)
        flow_values = np.zeros((number_of_nodes, number_of_nodes))

        if self.output_hidden_states:
            layers = 12
            for layer, i, j in product(range(layers), range(length), range(length)):
                flow_values[(layer + 1) * length + i][layer * length + j] = maximum_flow(graph,
                                                                                         (layer + 1) * length + i,
                                                                                         j).flow_value / approx
        else:
            layer = 11
            for i, j in product(range(length), range(length)):
                flow_values[(layer + 1) * length + i][layer * length + j] = maximum_flow(graph, (layer + 1) * length + i,
                                                                                         j).flow_value / approx
        if do_normalize:
            flow_values[length:] /= flow_values[length:].sum(axis=-1, keepdims=True)
        return flow_values

    @staticmethod
    def mats_to_adjmat(mat):
        layers, length, _ = mat.shape
        adjmat = np.zeros(((layers + 1) * length, (layers + 1) * length))
        for layer in range(layers):
            for j in range(length):
                for k in range(length):
                    adjmat[(layer + 1) * length + k,
                           layer * length + j] = mat[layer][k][j]
        return adjmat

    @staticmethod
    def adjmat_to_mats(adjmat, n_layers, l):
        mats = np.zeros((n_layers, l, l))
        for i in np.arange(n_layers):
            mats[i] = adjmat[(i + 1) * l:(i + 2) * l, i * l:(i + 1) * l]
        return np.array(mats)

    @staticmethod
    def mats_to_graph(mat):
        """
        :param mat: [#layers, sentence_len, sentence_len]
        :return: nx graph
        """
        g = nx.DiGraph()
        for layer in range(mat.shape[0]):
            for attender in range(mat.shape[1]):
                for attendee in range(mat.shape[2]):
                    g.add_edge(f"{layer}_{attendee}", f"{layer + 1}_{attender}",
                               capacity=mat[layer][attender][attendee])
        return g

    @staticmethod
    def compute_max_flow_nx(graph: nx.Graph, layer, num_nodes):
        flow_values = np.zeros((num_nodes, num_nodes))
        for attender in range(num_nodes):
            for attendee in range(num_nodes):
                flow_value = nx.maximum_flow_value(graph, f"0_{attendee}", f"{layer}_{attender}")
                flow_values[attender][attendee] = flow_value
        return flow_values
