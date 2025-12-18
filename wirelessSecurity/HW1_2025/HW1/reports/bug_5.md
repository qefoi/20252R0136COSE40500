# BUG-5
## Category
Wrong operators

## Description

The code incorrectly uses the comparison operator (==) instead of
the assignment operator (=) to calculate and assign new coordinate
values to the variables x and y. Consequently, the desired coordinate
values are not stored in the variables.

## Affected Lines in the original program
In `circle.c:61` and `circle.c:84`

## Expected vs Observed
The program expected a complete circular border should be drawn on the PNG file.
However, because the incorrect values are used for drawing the second half of the circle
(the previously calculated value is reused instead of the new one),
only the top and right sections of the circle are correctly drawn,
resulting in a distorted or incomplete boundary.

## Steps to Reproduce

### Command

```
./circle ./test_imgs/test.png output.png 100 100 100 ffffff 
```

## Suggested Fix Description
Replace the incorrect comparison operator (==) with the assignment operator (=)
in the affected lines to correctly update the coordinate variables.