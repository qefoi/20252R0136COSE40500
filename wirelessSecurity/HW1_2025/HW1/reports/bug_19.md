# BUG-19
## Category
Local persisting pointers

## Description

The function get_pixel returns the address of a local variable px,
which is declared inside the function.

## Affected Lines in the original program
In `filter.c:107 ~ 110`

## Expected vs Observed
It is expected that the address of px would be successfully returned.
However, since px is a local variable, it becomes invalid
after the function returns, causing a segmentation fault.
## Steps to Reproduce

### Command

```
./filter test.png output.png negative
```

## Suggested Fix Description
Do not use the get_pixel function. Instead,
declare the pixel variable directly inside
the filter_negative function.