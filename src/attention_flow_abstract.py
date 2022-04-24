import abc
import numpy as np


class AttentionFlow(abc.ABC):
    @abc.abstractmethod
    def compute_flows(self, attentions_list, desc="", output_hidden_states=False, num_cpus=4):
        raise NotImplementedError()

    def pre_process(self, att_mat):
        # if att_mat.sum(axis=-1)[..., None] != 1:
        #     att_mat = att_mat / np.max(att_mat, axis=(1, 2), keepdims=True)
        return att_mat
