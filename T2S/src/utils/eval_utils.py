# ROUGE metric
from rouge_score import rouge_scorer

NER_EVAL_MEASURES = ["COR", "INC", "PAR", "MIS", "SPU"]


# https://stackoverflow.com/questions/55488945/nernamed-entity-recognition-similarity-between-sentences-in-documents
def compute_jaccard_index(topic_ents, tweets_ents, decimal_figures=3):
    """
    Compute the jaccard index between two sets of entities.
    :param topic_ents: The topic entities (reference)
    :param tweets_ents: The tweets entities for the topic (hypothesis)
    :param decimal_figures: Number of decimal figures to round the jaccard index
    :return: The rounded jaccard index between topic_ents and tweets_ents
    """
    jaccard = len(topic_ents & tweets_ents) / len(topic_ents | tweets_ents)
    return round(jaccard * 100, decimal_figures)


def semeval_confusion_matrix(topic_ner, tweets_ner, baseline="ner"):
    """
    Based in - http://www.davidsbatista.net/blog/2018/05/09/Named_Entity_Evaluation/
    This blog post shows how the International Workshop on Semantic Evaluation (SemEval) evaluates NER tasks.
    Has 4 different measures and builds a confusion matrix using them:
    - Strict -
    - Exact -
    - Partial -
    - Type -
    :param topic_ner: List of (entity, label) tuples for topic entities
    :param tweets_ner: List of (entity, label) tuples for tweets entities
    :param baseline: String that describes the baseline used ("ner", "tt", "kw")
    :return: Dict schema containing the confusion matrix for the evaluated entities
    """
    if baseline == "ner":
        topic_ents = [ent for (ent, label) in topic_ner]
        topic_labels = [label for (ent, label) in topic_ner]
        tweet_ents = [ent for (ent, label) in tweets_ner]
        tweet_labels = [label for (ent, label) in tweets_ner]
    else:
        # tt
        topic_ents = [ent for (label, ent) in topic_ner]
        topic_labels = [label for (label, ent) in topic_ner]
        tweet_ents = [ent for (label, ent) in tweets_ner]
        tweet_labels = [label for (label, ent) in tweets_ner]

    measures = ["strict", "exact", "partial", "type"]

    eval_schema = {
        "metrics": {
            measure: {
                "COR": 0, "INC": 0, "PAR": 0, "MIS": 0, "SPU": 0
            } for measure in measures
        }
    }

    for ent, label in tweets_ner:
        # scenario I
        if (ent, label) in topic_ner:
            eval_schema = increment_schema(eval_schema, "COR", "COR", "COR", "COR")
        # scenario II
        if ent not in topic_ents:
            eval_schema = increment_schema(eval_schema, "SPU", "SPU", "SPU", "SPU")
        # scenario IV -> same entity - different label
        if ent in topic_ents:
            indices = [idx for idx, entity in enumerate(topic_ents) if entity == ent]
            for idx in indices:
                if topic_labels[idx] != label:
                    eval_schema = increment_schema(eval_schema, strict="INC", partial="COR", exact="COR", type_="INC")
        # scenario V and VI
        # Retrieves indexes where it finds the tweet entity is represented in part of the topic entity
        indices = [idx for idx, entity in enumerate(topic_ents) if ent in entity if entity != ent]
        if len(indices) > 0:
            for idx in indices:
                # scenario V -> same label - partial entity
                if topic_labels[idx] == label:
                    eval_schema = increment_schema(eval_schema, strict="INC", partial="PAR", exact="INC", type_="COR")
                # scenario VI -> partial entity - different label
                elif topic_labels[idx] != label:
                    eval_schema = increment_schema(eval_schema, strict="INC", partial="PAR", exact="INC", type_="INC")

    for ent, label in topic_ner:
        # Entity partial in topic and complete in tweet
        indices = [idx for idx, entity in enumerate(tweet_ents) if ent in entity if entity != ent]
        if len(indices) > 0:
            for idx in indices:
                # scenario V -> same label - partial entity
                if tweet_labels[idx] == label:
                    eval_schema = increment_schema(eval_schema, strict="INC", partial="PAR", exact="INC", type_="COR")
                # scenario VI -> partial entity - different label
                elif topic_labels[idx] != label:
                    eval_schema = increment_schema(eval_schema, strict="INC", partial="PAR", exact="INC", type_="INC")
        # scenario III -> topic entity not predicted as tweet entity
        elif ent not in tweet_ents:
            eval_schema = increment_schema(eval_schema, strict="MIS", partial="MIS", exact="MIS", type_="MIS")

    return eval_schema


def increment_schema(schema, strict, partial, exact, type_):
    rules = [strict not in NER_EVAL_MEASURES, partial not in NER_EVAL_MEASURES,
             exact not in NER_EVAL_MEASURES, type_ not in NER_EVAL_MEASURES]

    # Any stands for multiple or statements
    if any(rules):
        raise ValueError(f"Parameters 'strict', 'partial', 'exact' and 'type_' must be one of {NER_EVAL_MEASURES}")

    schema["metrics"]["strict"][strict] += 1
    schema["metrics"]["partial"][partial] += 1
    schema["metrics"]["exact"][exact] += 1
    schema["metrics"]["type"][type_] += 1

    return schema


def has_keyphrases(kw_list):
    """
    Checks if a list of strings contains any phrase with more than one word.
    :param kw_list: A List of string keywords or keyphrases
    :return: Boolean that flags whether or not the list has phrases (with more than one word)
    """
    kw_split = [val.split(" ") for val in kw_list]
    kw_len = [len(val) for val in kw_split]

    has_keyphrase = any([val > 1 for val in kw_len])
    return has_keyphrase


def keyphrase_eval_metrics(ref_kw, hyp_kw, data_row=None, fill_results_dict=False, base_name=None, decimal_figures=3,
                           rouge_metrics=None):
    """
    Evaluate two sets of keyphrases, a reference (ref_kw) against an hypothesis (hyp_kw).
    Computes similarity metrics (ROUGE and R-precision), between each keyword in the sets.
    When both sets do not have a single word in common (exactly the same), the resultfiles are 0.
    R-Precision based on this article - https://www.comp.nus.edu.sg/~kanmy/papers/coling10c.pdf

    :param ref_kw: Reference keywords (usually from the news article)
    :param hyp_kw: Hypothesis keywords (usually from the tweets)
    :param data_row: Dictionary with the structure for a row of the datafiles to be exported
    :param fill_results_dict: Whether or not to fill the data_row
    :param base_name: The name of baseline being evaluated (must consist with the one in the data_row)
    :param decimal_figures: Number of decimal figures to round to
    :param rouge_metrics: List of rouge metrics for the RougeScorer class (more info in the rouge_score pkg)
    :return: Either the data_row if we are exporting or the metric means otherwise

    Note: Keyphrases are keywords with more than one word.
    """
    if (not fill_results_dict) & (data_row is not None):
        # Could also issue warning and assign data_row=None
        raise ValueError(f"If fill_results_dict param is {fill_results_dict}, then data_row param must be None.")
    if rouge_metrics is None:
        rouge_metrics = ["rouge1"]
    rouge_s = rouge_scorer.RougeScorer(rouge_metrics)

    sum_rouge1p, sum_rouge1f, sum_R_precision, total = 0, 0, 0, len(ref_kw) * len(hyp_kw)
    for t_key in ref_kw:
        for tw_key in hyp_kw:
            key_ref = t_key.split(" ")
            key_hyp = tw_key.split(" ")
            r_precision = sum(t_key in key_ref for t_key in key_hyp) / max(len(key_hyp), len(key_ref))

            scores = rouge_s.score(t_key, tw_key)
            sum_rouge1p += scores["rouge1"].precision
            sum_rouge1f += scores["rouge1"].fmeasure
            sum_R_precision += r_precision

    mean_rouge1p = round((sum_rouge1p / total) * 100, decimal_figures)
    mean_rouge1f = round((sum_rouge1f / total) * 100, decimal_figures)
    mean_R_precision = round((sum_R_precision / total) * 100, decimal_figures)

    if fill_results_dict & (data_row is not None):
        data_row["baselines"][base_name]["metrics"]["rouge1_precision"] = mean_rouge1p
        data_row["baselines"][base_name]["metrics"]["rouge1_fscore"] = mean_rouge1f
        data_row["baselines"][base_name]["metrics"]["R_precision"] = mean_R_precision
    else:
        return mean_rouge1p, mean_rouge1f, mean_R_precision

    return data_row


def semeval_metrics_computation(eval_schema, decimal_figures=3):
    COR = sum(eval_schema["metrics"][measure]["COR"] for measure in ["strict", "exact", "partial", "type"])
    INC = sum(eval_schema["metrics"][measure]["INC"] for measure in ["strict", "exact", "partial", "type"])
    PAR = sum(eval_schema["metrics"][measure]["PAR"] for measure in ["strict", "exact", "partial", "type"])
    MIS = sum(eval_schema["metrics"][measure]["MIS"] for measure in ["strict", "exact", "partial", "type"])
    SPU = sum(eval_schema["metrics"][measure]["SPU"] for measure in ["strict", "exact", "partial", "type"])

    POS = COR + INC + PAR + MIS  # TP + FN
    ACT = COR + INC + PAR + SPU  # TP + FP

    # Exact match eval
    exact_precision = COR / ACT
    exact_recall = COR / POS
    try:
        exact_F1 = (2 * exact_precision * exact_recall) / (exact_precision + exact_recall)
    except ZeroDivisionError:
        exact_F1 = 0

    # Partial match eval
    partial_precision = (COR + 0.5 * PAR) / ACT  # TP / (TP + FP)
    partial_recall = (COR + 0.5 * PAR) / POS  # TP / (TP + FP)
    try:
        partial_F1 = (2 * partial_precision * partial_recall) / (partial_precision + partial_recall)
    except ZeroDivisionError:
        partial_F1 = 0

    metrics_schema = {
        "exact_precision": round(exact_precision * 100, decimal_figures),
        "exact_recall": round(exact_recall * 100, decimal_figures),
        "exact_F1": round(exact_F1 * 100, decimal_figures),
        "partial_precision": round(partial_precision * 100, decimal_figures),
        "partial_recall": round(partial_recall * 100, decimal_figures),
        "partial_F1": round(partial_F1 * 100, decimal_figures)
    }

    return metrics_schema
