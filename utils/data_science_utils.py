from utils.functional_utils import apply_to_result


@apply_to_result(sum)
def mse(predicted_y, expected_y):
    for y1, y2 in zip(predicted_y, expected_y):
        yield (y1 - y2) ** 2
