from typing import Tuple


def DT_TOKEN() -> str:
    # TODO: change this to your duckietown token
    dt_token = "dt2-5wjpkjyNDVkVSh5zZ1XnkmiRFq8CEh6GU3TDwdUbfhroquutQk5PwNWfSLfgcsqsv3e45dh8Gjtf714mz-43dzqWFnWd8KBa1yev1g3UKnzVxZkkTbfVHPdevkcHbng3wiLnXkgKrwUQFJfvSyqw"
    return dt_token


def MODEL_NAME() -> str:
    # TODO: change this to your model's name that you used to upload it on google colab.
    # if you didn't change it, it should be "yolov5n"
    return "yolov5n"


def NUMBER_FRAMES_SKIPPED() -> int:
    # TODO: change this number to drop more frames
    # (must be a positive integer)
    return 1


def filter_by_classes(pred_class: int) -> bool:
    """
    Remember the class IDs:

        | Object    | ID    |
        | ---       | ---   |
        | Duckie    | 0     |
        | Cone      | 1     |
        | Truck     | 2     |
        | Bus       | 3     |


    Args:
        pred_class: the class of a prediction
    """
    # Right now, this returns True for every object's class
    # TODO: Change this to only return True for duckies!
    # In other words, returning False means that this prediction is ignored.
    return pred_class==0


def filter_by_scores(score: float) -> bool:
    """
    Args:
        score: the confidence score of a prediction
    """
    # Right now, this returns True for every object's confidence
    # TODO: Change this to filter the scores, or not at all
    # (returning True for all of them might be the right thing to do!)
    return score>=0.5


def filter_by_bboxes(bbox: Tuple[int, int, int, int]) -> bool:
    """
    Args:
        bbox: is the bounding box of a prediction, in xyxy format
                This means the shape of bbox is (leftmost x pixel, topmost y, rightmost x, bottommost y)
    """
    # TODO: Like in the other cases, return False if the bbox should not be considered.
    left, top, right, bottom = bbox

    bbox_width = right - left
    bbox_height = bottom - top

    area = bbox_width * bbox_height

    min_area = 5000

    image_width = 640

    left_third = image_width / 3
    right_third = 2 * left_third

    if left < left_third or right > right_third:
        return False
    if area < min_area:
        return False
    
    return True