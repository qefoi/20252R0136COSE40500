# BUG-8
## Category
Iteration errors

## Description

In the nested while loop structure, the increment operation for
the outer loop variable (i++) is incorrectly placed inside the inner while loop,
instead of being outside the inner loop. Furthermore,
the inner loop variable (j) is not reset (re-initialized) to zero
before the inner while loop starts for a new iteration of the outer loop.

## Affected Lines in the original program
In `rect.c:62` and `rect.c:80`

## Expected vs Observed
A rectangle is expected to be drawn, but instead, only a diagonal line is drawn.

## Steps to Reproduce

### Command

```
./rect ./test_imgs/test.png output.png 0 0 10 10 ffffff
```

## Suggested Fix Description
The position of i++ should be moved outside the inner while loop,
and the initialization of j to 0 should be moved inside the outer while loop.