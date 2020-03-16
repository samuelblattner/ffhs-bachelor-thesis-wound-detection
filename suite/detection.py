from typing import List


class Detection:

    bbox: List[int] = []
    class_name: str = 'No Class'
    score: float = 0.0
