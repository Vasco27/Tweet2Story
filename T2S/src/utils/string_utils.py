import re
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))


def flatten_list(list_to_flatten):
    """
    Simply flattens a list of nested lists.

    :param list_to_flatten: list of nested lists to be flattened
    :return: the flattened list
    """
    if not isinstance(list_to_flatten, list):
        raise ValueError(f"Parameter list_to_flatten must be a list. Instead it was {type(list_to_flatten)}.")

    return [item for sublist in list_to_flatten for item in sublist]


def value_indexes_in_list(ref_list, value, negative_condition=False):
    """
    Finds the indexes where a value or values are contained in a list.

    :param ref_list: the list from where to search the value
    :param value: the value or values to search for within the list (must be a list)
    :param negative_condition: Whether to query the value through the logical negative
    :return: a list of indexes corresponding to the queried value
    """
    if not isinstance(value, list):
        raise ValueError(f"Parameter value must be of type list. Instead it was of type {type(value)}")

    if negative_condition:
        return [idx for idx, val in enumerate(ref_list) if val not in value]
    else:
        return [idx for idx, val in enumerate(ref_list) if val in value]


def multiple_index_list(ref_list, indexes):
    """
    Index a list with multiple item indexes.
    Example: having indexes - [1, 2, 3] we want to access ref_list[1, 2, 3].

    :param ref_list: list to index
    :param indexes: indexes of values within the range of the list
    :return: the values in the indexes within the list
    """
    if (not isinstance(ref_list, list)) | (not isinstance(indexes, list)):
        raise ValueError(f"Parameters must both be lists. "
                         f"Instead ref_list was {ref_list} and indexes was {type(indexes)}.")

    return [ref_list[i] for i in indexes]


def trim_whitespaces(text):
    """
    Trim whitespaces from an entire text or a list of words. Including whitespaces preceding punctuation.

    :param text: str or list of str to trim
    :return: str fully trimmed of whitespaces

    Reference:
    https://stackoverflow.com/questions/18878936/how-to-strip-whitespace-from-before-but-not-after-punctuation-in-python
    """
    # todo: Make util class to check for errors (function check_if_string_or_string_list, e.g.)
    if not isinstance(text, (str, list)):
        raise ValueError(f"Parameter text must be of type str or list. Instead it was {type(text)}")
    if isinstance(text, list):
        if len(text) <= 0:
            raise ValueError(f"List must contain at least one value.")
        if any(not isinstance(w, str) for w in text):
            raise ValueError(f"List must contain only str values.")

    if isinstance(text, list):
        text = ' '.join(text)

    temp_str = ' '.join(text.split())  # Trims whitespaces
    normalized_text = re.sub(r'\s([?.!",](?:\s|$))', r'\1', temp_str)

    return normalized_text


def normalize_entities(entities_list, nested_list=False):
    """
    Normalizes a list of entities, splits them by words, lowers them and removes stopwords.
    Its primary use is to remove the coref/tt entities from the results of the spacy NER.

    :param entities_list: list of entities in the text
    :param nested_list: indicates if the entities_list is a list of lists
    :return: a list (flattened) of the most relevant parts of the entities

    Note: The nested list option exists to deal with clusters from the co-reference resolution.
    """
    if nested_list:
        entities_list = flatten_list(entities_list)

    normalized_ents = [ent.lower() for ent in entities_list]  # Lower case
    normalized_ents = [re.sub(r"[^\w\s]", "", ent) for ent in normalized_ents]  # Remove punctuation
    normalized_ents = flatten_list([ent.split() for ent in normalized_ents])  # Split words and flatten
    normalized_ents = [ent for ent in normalized_ents if ent not in stop_words]  # Remove stopwords

    return normalized_ents
