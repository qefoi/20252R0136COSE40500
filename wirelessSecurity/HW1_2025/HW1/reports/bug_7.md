# BUG-7
## Category
Type error

## Description

To use the pointer correctly, the end_ptr variable should be used with
the address-of operator (&). Since the & was omitted, the program attempts
to write a new pointer value to the memory address currently held by
the uninitialized end_ptr (which contains an arbitrary value).

## Affected Lines in the original program
In `rect.c:34`

## Expected vs Observed
It is expected that the string will be correctly converted to a long integer via strtol.
However, due to the incorrect pointer usage, a segmentation fault occurs.

## Steps to Reproduce

### Command

```
./rect ./test_imgs/test.png output.png 0 0 10 10 ffffff
```

## Suggested Fix Description
Prepend the end_ptr variable with the address-of operator (&).