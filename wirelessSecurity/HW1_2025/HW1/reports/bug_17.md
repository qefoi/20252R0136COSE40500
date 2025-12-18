# BUG-17
## Category
Unchecked system call returning code and Arithmetic overflow/underflow

## Description

The memory allocation fails due to the unrestricted size of n_pixels,
leading to an attempt to request more memory than the system can provide.

## Affected Lines in the original program
In `resize.c:52` 

## Expected vs Observed
It is expected that img->px is successfully allocated with the size
determined by n_pixels. However, when n_pixels exceeds the momory size
allowed by the system, malloc fails, resulting in a segmentation fault
instead of a graceful exit or a valid memory block.

## Steps to Reproduce

### Command

```
./resize ./test_imgs/test2.png output.png 600
```
### Proof-of-Concept Input (if needed)
(attached: test2.png)

## Suggested Fix Description
The size of n_pixels must be restricted to a value that does not exceed
the allowed system memory. Asuuming a safe memory limit of 4GB,
malloc should only be executed if n_pixels corresponds to a memory size
that does not surpass this limit.