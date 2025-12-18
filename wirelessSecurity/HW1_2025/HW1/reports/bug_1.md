# BUG-1
## Category
Heap overflow/underflow

## Description

The Heap Overflow occurs because the inner loops responsible for filling pixels do not
check if the calculated pixel index is within the actual allocated image boundaries.

## Affected Lines in the original program
In `checkerboard.c:115`, `checkerboard.c:117`, `checkerboard.c:119`, `checkerboard.c:121`

## Expected vs Observed
The operation is expected to fill the image to the pixel coordinates without memory corruption.
The internal pixel drawing loop iterates only based on the user-provided square_width and
does not check the image boundaries. This causes the code to attempt to write
past the boundaries of the allocated pixel buffer (img->px), resulting in a heap overflow.

## Steps to Reproduce

### Command

```
./checkerboard output.png 100 100 10000 ffffff 000000
```

## Suggested Fix Description
The fix involves introducing an explicit boundary check if within the inner loops.
By using the continue statement when an out-of-bounds index is detected,
the routine skips the illegal pixel assignment and proceeds to the next iteration.
This prevents the drawing function from accessing memory outside the allocated image data,
ensuring the operation completes safely without causing a Heap Overflow.