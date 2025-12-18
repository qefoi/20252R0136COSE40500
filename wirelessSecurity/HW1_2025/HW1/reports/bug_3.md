# BUG-3
## Category
Temporal safety violation

## Description

A double free occurs when malloc for img->px fails (returning NULL),
and the program subsequently executes the error handling logic.

## Affected Lines in the original program
In `checkerboard.c:91`

## Expected vs Observed
It is expected that the program would gracefully free all dynamically
allocated memory and then exit with a return code of 1 after processing
the error handling logic.

However, when img->px is NULL (due to malloc failure), the code proceeds
to free img once, and then the error handling routine is entered, where
img is freed a second time. This results in a double free error.

## Steps to Reproduce

### Command

```
./checkerboard output.png 65534 65534 100 ffffff 000000
```

## Suggested Fix Description
The issue is resolved by removing the redundant free(img) call
that occurs before entering the common error handling code.
The memory cleanup should be centralized within the error handling section,
ensuring that free(img) is called only once before the program terminates.
This prevents the double free.