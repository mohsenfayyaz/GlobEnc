from abc import ABC
import numpy as np
from tqdm.auto import tqdm
try:
    from src.attention_flow_abstract import AttentionFlow
except Exception:
    from ..src.attention_flow_abstract import AttentionFlow


class AttentionRollout(AttentionFlow, ABC):
    def compute_flows(self, attentions_list, desc="", output_hidden_states=False, num_cpus=0, disable_tqdm=False):
        """
        :param attentions_list: list of attention maps (#examples, #layers, #sent_len, #sent_len)
        :param desc:
        :param output_hidden_states:
        :param num_cpus:
        :return:
        """
        attentions_rollouts = []
        for i in tqdm(range(len(attentions_list)), desc=desc, disable=disable_tqdm):
            if output_hidden_states:
                attentions_rollouts.append(self.compute_joint_attention(attentions_list[i]))
            else:
                attentions_rollouts.append(self.compute_joint_attention(attentions_list[i])[[-1]])
        return attentions_rollouts

    def compute_joint_attention(self, att_mat):
        res_att_mat = self.pre_process(att_mat)
        # res_att_mat = res_att_mat[4:10, :, :]
        joint_attentions = np.zeros(res_att_mat.shape)
        layers = joint_attentions.shape[0]
        joint_attentions[0] = res_att_mat[0]
        for i in np.arange(1, layers):
            joint_attentions[i] = res_att_mat[i].dot(joint_attentions[i - 1])

        return joint_attentions
