import torch
from tqdm.auto import tqdm
import numpy as np


def extract_attentions(model, encoder_func, dataset_len, device="cpu", delete_sep=False):
    """
    Extract raw attentions and norms
    :param model:
    :param encoder_func:
    :param dataset_len:
    :param device:
    :return:
    """
    raw_attentions = []
    norms_list = [[] for i in range(9)]
    # head_attn_n, attn_n, attnres_n, attnresln_n, (+attn_enc),
    # attn_n_ratio, attnres_n_ratio, attnresln_n_ratio, (+attn_enc_ratio)
    model.to(device)
    model.eval()
    for id in tqdm(range(dataset_len)):
        encoded = encoder_func(id).to(device)
        # encoded = tokenizer.encode_plus(data["text"], return_tensors="pt").to(device)        
        with torch.no_grad():
            logits, attentions, norms = model(**encoded, output_attentions=True, output_norms=True, return_dict=False)
            # logits:     [1, 2],
            # attentions: 12(layer) * [1, 12(heads), 24(sentence_len), 24(sentence_len)],
            # norms:      12(layer) * 7+2(type)

        last_token = attentions[0].shape[-1]
        if delete_sep:
            last_token -= 1

        num_layers = len(attentions)
        for attention_type in range(9):
            norm = torch.stack([norms[i][attention_type] for i in range(num_layers)]).squeeze().cpu().numpy()
            if 0 < attention_type < 5:  # N: 1, N-Res: 2, N-ResLN: 3, N-Enc: 4
                norm = norm[:, :last_token, :last_token]
            elif attention_type >= 5:
                norm = norm[:, :last_token]
            norms_list[attention_type].append(norm)

        raw_attention = torch.mean(torch.stack(attentions).squeeze(), axis=1).cpu().numpy()  # Mean of heads
        raw_attentions.append(raw_attention[:, :last_token, :last_token])  # (12, sentence_len, sentence_len)
    return raw_attentions, norms_list


def build_ratio_residual_attentions(raw_attentions_list, norms_list):
    r_ratio_attentions = {
        "W-FixedRes": [],
        "W-Res": [],
        "N-FixedRes": [],
        # "Uniform-Res": []
    }
    for idx in tqdm(range(len(raw_attentions_list))):
        raw_attention = raw_attentions_list[idx]
        r_half_matrix = np.ones(norms_list[5][idx].shape) * 0.5

        r_ratio_attentions["W-FixedRes"].append(__build_ratio_residual_attention(raw_attention, r_half_matrix))

        # normalized_attn_n = norms_list[1][idx] / np.max(norms_list[1][idx], axis=(1, 2), keepdims=True)
        normalized_attn_n = norms_list[1][idx] / np.sum(norms_list[1][idx], axis=2, keepdims=True)
        r_ratio_attentions["N-FixedRes"].append(__build_ratio_residual_attention(normalized_attn_n, r_half_matrix))

        # norms_list[8]: N-Enc_ratio
        r_ratio_attentions["W-Res"].append(__build_ratio_residual_attention(raw_attention, norms_list[8][idx], wres=True))

        # r_ratio_attentions["Uniform-Res"].append(
        #     __build_ratio_residual_attention(np.ones_like(raw_attention) / len(raw_attention), norms_list[8][idx]))

    return r_ratio_attentions


def __build_ratio_residual_attention(raw_attention, ratio_matrix, wres=False):
    """
    :param raw_attention: (layers, sentence_len, sentence_len)
    :param ratio_matrix: (layers, sentence_len)
    :return:
    """
    result_attention = np.zeros(raw_attention.shape)
    for layer in range(raw_attention.shape[0]):
        result_attention[layer] = __add_residual(raw_attention[layer], ratio_matrix[layer], wres)
    return result_attention


def __add_residual(att_mat, ratios, wres=False):
    """
    :param att_mat: (sentence_len, sentence_len)
    :param ratios: (sentence_len)
    :return:
    """
    att_mat_cp = np.copy(att_mat)
    for token_idx in range(att_mat_cp.shape[0]):
        r = ratios[token_idx]
        if wres:
            att_mat_cp[token_idx][token_idx] = 0
            att_mat_cp[token_idx] /= np.sum(att_mat_cp[token_idx])
        att_mat_cp[token_idx] *= r
        att_mat_cp[token_idx][token_idx] += (1 - r)
    return att_mat_cp
