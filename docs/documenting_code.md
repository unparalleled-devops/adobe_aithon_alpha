# Documenting code

## Usage of comments

Comments should be used in places where it might be necessary to offer an explanation to what the code does for people reading it, or to offer a justification for a particular code choice.

## Usage of docstrings

Code will be documented using docstrings following the given format

```py
"""<one-line description>

<longer description if required>

Parameters:
parameter_name: type -> description
...

Returns:
<description of returned value>

Raises: <only necessary if code throws an exception>
exception_type: reason for throwing
"""
```

A realistic example could be

```py
def add(x: int, y: int) -> int:
  """Adds two integers together.

  Adds two integers together and returns their sum.
  Does not accept negative values.

  Parameters:
  x: int -> the first integer
  y: int -> the second integer

  Returns:
  An integer that is the sum of the parameters x and y.

  Raises:
  ValueError: when given a negative integer
  """

  # arbitrary constraint
  if x < 0 or y < 0:
    raise ValueException('Cannot add negative integers')

  return x + y
```
