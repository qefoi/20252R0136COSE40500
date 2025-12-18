# BUG-15
## Category
Wrong operators

## Description

The addition operator (+) was used instead of the multiplication operator (*).

## Affected Lines in the original program
In `resize.c:48`

## Expected vs Observed
It was expected that malloc would allocate memory for n_pixels multiplied by sizeof(struct pixel).
However, memory is allocated for n_pixels added to sizeof(struct pixel).

## Steps to Reproduce

### Command

```
./resize ./test_imgs/test.png out.png 100
```
### Proof-of-Concept Input (if needed)
(attached: test.png)

## Suggested Fix Description
Replace the addition operator (+) with the multiplication operator (*).