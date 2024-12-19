


def num_score_categories(range_start, range_end, step):
    return len(range(range_start, range_end, step)) + 1


def categorize_score(score, range_start, range_end, step):
    for idx, x in enumerate(range(range_start, range_end, step)):
        if score < x:
            return idx
    return idx + 1