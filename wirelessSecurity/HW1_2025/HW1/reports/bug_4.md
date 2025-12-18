# BUG-4
## Category
Heap overflow/underflow

## Description

The program expects 7 arguments, which means the valid index
for the argv array goes up to argv[6]. However, the code
incorrectly accesses argv[7], leading to overflow.

## Affected Lines in the original program
In `circle.c:29` and `circle.c:30`

## Expected vs Observed
The program expected the 7th command-line argument to be assigned
to the hex_color variable, but by using argv[7], it attempted
to access the non-existent 8th value in the argument list,
resulting in a segmentation fault.

## Steps to Reproduce

### Command

```
./circle ./test_imgs/test.png output.png 0 0 10 ffffff
```

## Suggested Fix Description
To correctly use the 7th argument, modify the index from argv[7] to argv[6].