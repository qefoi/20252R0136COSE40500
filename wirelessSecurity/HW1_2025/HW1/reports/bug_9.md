# BUG-9
## Category
Arithmetic overflow/underflow

## Description

When a negative number is passed as an argument,
an underflow (wrap-around) occurs due to the variable being declared as unsigned.

## Affected Lines in the original program
In `rect.c:23`, `rect.c:24`, `rect.c:26` and `rect.c:27`

## Expected vs Observed
It is expected that if a negative number is passed as an argument,
the coordinate will be set to 0.
However, when a negative value is input, an underflow occurs.

## Steps to Reproduce

### Command

```
./rect ./test_imgs/test.png output.png 0 0 -1 -1 ffffff
```

## Suggested Fix Description
Change the type of the coordinate variables from unsigned to int,
and then set the coordinate value to 0 if it is negative.