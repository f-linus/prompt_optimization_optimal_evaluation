from scipy import stats

from prompt_optimization.types.reference_evaluation import (
    ReferenceEvaluation,
)


def spearman_rank_correlation(
    evaluation_set: list[ReferenceEvaluation],
):
    """
    Calculate Spearman's rank correlation coefficient for a set of evaluations.

    This function computes the Spearman's rank correlation between the extracted reference
    and actual values from a list of Evaluation objects. It handles both single float values
    and dictionaries of values.

    Parameters:
    -----------
    evaluation_set : list[Evaluation]
        A list of Evaluation objects containing extracted reference and actual values.

    Returns:
    --------
    float | dict
        If the extracted values are floats, returns a single Spearman's rank correlation coefficient.
        If the extracted values are dictionaries, returns a dictionary of Spearman's rank correlation
        coefficients for each key in the dictionaries.

    Notes:
    ------
    - Uses scipy.stats.spearmanr for calculation.
    - Assumes all dictionary values (if applicable) have the same keys.
    """

    reference_values = [e.extracted_reference for e in evaluation_set]
    actual_values = [e.extracted_actual for e in evaluation_set]

    if isinstance(reference_values[0], float):
        return stats.spearmanr(reference_values, actual_values)

    # if it is a dict return a dict of spearman rank correlations
    result = {}
    for key in reference_values[0].keys():
        result[key] = stats.spearmanr(
            [r[key] for r in reference_values], [a[key] for a in actual_values]
        )
    return result
