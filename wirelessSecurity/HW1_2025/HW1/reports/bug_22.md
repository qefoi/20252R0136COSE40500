# BUG-22
## Category
Unchecked system call returning code

## Description

The function store_png, which is responsible for saving the file,
does not check whether it succeeds or fails.

## Affected Lines in the original program
In `checkerboard.c:128`, `circle.c:93`, `filter.c:287`, `rect.c:85`,`resize.c:74`

## Expected vs Observed
It is expected that an error would occur if store_png fails.
However, since there is no code that checks the return value,
the program appears to succeed even when the operation fails.

## Steps to Reproduce

### Command

```
./checkerboard %s.png output.png negative
```

## Suggested Fix Description
Add code to check the return value of store_png to verify whether it succeeded.