# BUG-21
## Category
Heap overflow/underflow

## Description

An overflow occurs because the loop accesses image_data
beyond its allocated size by including img->size_x and
img->size_y in its iteration range

## Affected Lines in the original program
In `filter.c64`, `filter.c:65`, `filter.c:112` and `filter.c:113`

## Expected vs Observed
It is expected that the filter iterates through
all pixels using the loop. However, a segmentation
fault is observed due to accessing memory outside
the bounds of image_data.

## Steps to Reproduce

### Command

```
./filter %s.png output.png negative
```

## Suggested Fix Description
Do not include img->size_x and img->size_y
in the loop's iteration range.