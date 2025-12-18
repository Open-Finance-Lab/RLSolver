from typing import List, Tuple
import collections.abc
import functools
from typing import Any, Callable, Dict, Hashable


class memoized:
    """Decorator. Caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned
    (not reevaluated).
    """
    def __init__(self, func: Callable[..., Any]) -> None:
        self.func = func
        self.cache: Dict[Hashable, Any] = {}

    def __call__(self, *args: Any) -> Any:
        if not isinstance(args, collections.abc.Hashable):
            # uncacheable. a list, for instance.
            # better to not cache than blow up.
            return self.func(*args)
        if args in self.cache:
            return self.cache[args]
        else:
            value = self.func(*args)
            self.cache[args] = value
            return value

    def __repr__(self) -> str:
        """Return the function's docstring."""
        return self.func.__doc__

    def __get__(self, obj: Any, objtype: Any) -> Any:
        """Support instance methods."""
        return functools.partial(self.__call__, obj)
def dynamic_programming(number: int, capacity: int, weight_cost: List[Tuple[int, int]]) -> Tuple[int, List[int]]:
    """
    Solve the knapsack problem by finding the most valuable
    subsequence of `weight_cost` subject that weighs no more than `capacity`.
    :param number: number of existing items
    :param capacity: is a non-negative integer
    :param weight_cost: is a sequence of pairs (weight, cost)

    :return: a pair whose first element is the sum of costs in the best combination,
    and whose second element is the combination.
    """

    # Return the value of the most valuable subsequence of the first i
    # elements in items whose weights sum to no more than j.
    @memoized
    def bestvalue(i, j):
        if i == 0:
            return 0
        weight, cost = weight_cost[i - 1]
        if weight > j:
            return bestvalue(i - 1, j)
        else:
            # maximizing the cost
            return max(bestvalue(i - 1, j), bestvalue(i - 1, j - weight) + cost)

    j = capacity
    result = [0] * number
    for i in range(len(weight_cost), 0, -1):
        if bestvalue(i, j) != bestvalue(i - 1, j):
            result[i - 1] = 1
            j -= weight_cost[i - 1][0]
    return bestvalue(len(weight_cost), capacity), result
