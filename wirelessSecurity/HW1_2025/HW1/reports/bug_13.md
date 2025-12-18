# BUG-13
## Category
Local persisting pointers

## Description

The palette is dynamically allocated but not freed before the program exits.

## Affected Lines in the original program
In `solid.c:109` and `solid.c:144`

## Expected vs Observed
It is expected that the palette, after being dynamically allocated,
would be freed before termination. However, the palette is not freed
outside of the error handling path.

## Steps to Reproduce

### Command

```
valgrind --leak-check=full ./solid test.png 100 100 ffffff
```

## Suggested Fix Description
Add free(palette) before the program terminates (in the successful exit path)
and within the error_mem cleanup label.