from functools import wraps
from typing import Callable, Any

from typing_extensions import T, VT


def curry(func: Callable[..., T]) -> Callable[..., Callable[..., T]]:
    @wraps(func)
    def first_args_func(*first_args: Any) -> Callable[..., T]:
        @wraps(func)
        def second_args_func(*second_args: Any) -> T:
            return func(*first_args, *second_args)
        return second_args_func
    return first_args_func


@curry
@curry
def apply_to_result(wrapped_func: Callable[[T], VT], func_to_apply_on_result: Callable[..., T], *args: Any,
                    **kwargs: Any) -> VT:
    return wrapped_func(func_to_apply_on_result(*args, **kwargs))


def run_on_true(boolean_value: bool, action: Callable[[], None]):
    if boolean_value:
        action()
