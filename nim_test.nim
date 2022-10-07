import nimpy

proc fib(n: int): int {.exportpy.} =
    return if n < 3: 1 else: fib(n - 1) + fib(n - 2)

