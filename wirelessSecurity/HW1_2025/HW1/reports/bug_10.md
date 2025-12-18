# BUG-10
## Category
String Vulnerability and Stack buffer Overflow

## Description

The argument for the output filename should be correctly allocated
to the output_name variable, which has a size of 500. However,
the use of the strcpy function allows input exceeding 500 bytes
to be allocated, causing an overflow.

## Affected Lines in the original program
In `solid.c:33`

## Expected vs Observed
It is expected that the argument would be correctly allocated to output_name
and that allocation would be limited if the input exceeds 500 bytes. However,
when an input exceeding 500 bytes is provided, a buffer overflow occurs.

## Steps to Reproduce

### Command

```
./solid aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa 10 10 ffffff
```

## Suggested Fix Description
Remove strcpy and use the snprintf function to ensure safe string copying
by limiting the output to the size of the buffer (OUTPUT_NAME_SIZE).