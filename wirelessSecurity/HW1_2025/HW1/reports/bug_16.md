# BUG-16
## Category
Arithmetic overflow/underflow

## Description

new_height and new_width can overflow when multiplied by the factor,
exceeding the range of the unsigned short type.

## Affected Lines in the original program
In `resize.c:33` and `resize.c:34`

## Expected vs Observed
It was expected that the values of height and width multiplied by factor
would be stored in new_height and new_width. However,
when the multiplication result exceeds the range of unsigned short,
an overflow occurs.

## Steps to Reproduce

### Command

```
./resize ./test_imgs/test2.png output.png 600
```
### Proof-of-Concept Input (if needed)
(attached: test2.png)

## Suggested Fix Description
Use unsigned int instead of unsigned short and add a limit check for the maximum value.