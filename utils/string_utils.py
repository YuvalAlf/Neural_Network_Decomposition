from typing import Iterable


def mkstring(iterable: Iterable, *, start: str = '', separator: str = '', ending: str = '') -> str:
    return start + separator.join(map(str, iterable)) + ending
