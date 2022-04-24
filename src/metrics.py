from scipy.stats import spearmanr, pearsonr
import numpy as np
from tqdm.auto import tqdm


def compute_spearman_correlation(attentions_list, saliency_file_address, desc="", aggregation="CLS", max_length=512):
    """
    :param attentions_list: (#batch, #layers, sentence_len, sentence_len)
    :param saliency_file_address:
    :param desc: tqdm desc
    :param aggregation: CLS (Based on what affects CLS) | SUM (Based on the effect on all tokens)
    :return: spearmans (#batch, #layers, attender)
    """
    saliencies = np.load(saliency_file_address)
    # pearsons = []
    spearmans = []

    if len(attentions_list[0].shape) == 2:  # No layers
        attentions_list = [a.reshape(1, a.shape[0], a.shape[1]) for a in attentions_list]

    for i in tqdm(range(len(attentions_list)), desc=desc):
        i_spearmans = []
        for layer in range(attentions_list[i].shape[0]):
            length = min(len(attentions_list[i][0]), max_length)
            # pearsons.append(pearsonr(attentions[i].sum(axis=0), saliencies[i][:length])[0])
            if aggregation == "CLS":
                i_spearmans.append(
                    spearmanr(attentions_list[i][layer][0][:length], saliencies[i][:length]).correlation)  # CLS
            elif aggregation == "SUM":
                i_spearmans.append(
                    spearmanr(attentions_list[i][layer].sum(axis=0)[:length], saliencies[i][:length]).correlation)
            else:
                raise Exception("Undefined aggregation method. Possible values: CLS, SUM")
        spearmans.append(np.array(i_spearmans))
    return spearmans


def compute_spearman_correlation_hta(attentions_list, hta_file_address, desc="", max_length=512):
    """
    :param attentions_list: (256, 12, seq_len, seq_len)
    :param hta_file_address: (12, 256, 64, 64)
    :param desc:
    :param max_length:
    :return: (256, 12, seq_len) = (batch, layers, attender)
    """
    hta = np.load(hta_file_address)
    spearmans = []

    if len(attentions_list[0].shape) == 2:  # No layers
        attentions_list = [a.reshape(1, a.shape[0], a.shape[1]) for a in attentions_list]

    # len(attentions_list)
    for i in tqdm(range(len(attentions_list)), desc=desc):
        i_spearmans = []
        length = min(len(attentions_list[i][0]), max_length)
        for layer in range(attentions_list[i].shape[0]):
            i_layer_spearmans = []
            for attender in range(length):
                i_layer_spearmans.append(spearmanr(attentions_list[i][layer][attender][:length],
                                             hta[layer][i][attender][:length]).correlation)
            i_spearmans.append(np.array(i_layer_spearmans))
        spearmans.append(np.array(i_spearmans))
    return spearmans
