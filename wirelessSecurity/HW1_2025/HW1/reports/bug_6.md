# BUG-6
## Category
Heap overflow/underflow

## Description

The x and y coordinates of image_data are valid from 0 up to width and 0 up to height, respectively.
However, the x and y coordinates in image_data exceed these values, causing overflow and underflow.

## Affected Lines in the original program
In `circle.c:52 ~ circle.c:91`

## Expected vs Observed
It is expected that the circle will not be drawn (i.e., it will be clipped)
for coordinates less than 0 or greater than or equal to width or height.
However, the circle is observed to be drawn at the opposite coordinates.

## Steps to Reproduce

### Command

```
./circle ./test_imgs/test.png output.png 0 0 100 ffffff 
```

## Suggested Fix Description
Add a conditional statement to check for coordinates less than 0 or greater than or equal to
width or height, so that the circle drawing operation is not performed for these out-of-bounds coordinates.