import numpy as np
from gensim import matutils
import gensim.downloader
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, precision_score, recall_score

fasttext = None


def get_fasttext_model():
    global fasttext
    if fasttext is None:
        fasttext = gensim.downloader.load('fasttext-wiki-news-subwords-300')
    return fasttext


def cosine_similarity_for_word_pair(predicted: str, gold: str) -> float:
    model = get_fasttext_model()
    try:
        if len(predicted.split()) > 1:
            predicted_vec = matutils.unitvec(
                model.get_mean_vector(
                    keys=predicted.split()))
        else:
            predicted_vec = model.get_vector(predicted, norm=True)
        if len(gold.split()) > 1:
            gold_vec = matutils.unitvec(
                model.get_mean_vector(
                    keys=gold.split()))
        else:
            gold_vec = model.get_vector(gold, norm=True)
        return np.dot(predicted_vec, gold_vec)
    except BaseException:
        print(
            f"(fasttext) - Error in computing cosine similarity: predicted '",
            predicted, "', but wanted '",
            gold, "'")
        return 0


# compute spearman correlation between fasttext score and final annotation
def spearmanr_correlation(annotated_test_df):
    return spearmanr(
        annotated_test_df["embedding_fasttext_sim"],
        annotated_test_df["Annotation"])


def compute_scores(y_true: list[int], y_pred: list[int]) -> tuple[float, float, float]:
    return f1_score(y_true, y_pred), precision_score(y_true, y_pred), recall_score(y_true, y_pred)
