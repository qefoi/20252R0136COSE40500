# BUG-18
## Category
String Vulnerability and Stack buffer Overflow

## Description

An overflow occurs when strcpy copies input that exceeds the size of arg.

## Affected Lines in the original program
In `filter.c:225` 

## Expected vs Observed
It was expected that when the input exceeds the size of arg,
only up to the size of arg would be accepted. However,
since strcpy is used, input larger than the size of arg
is copied into arg, causing an overflow.

## Steps to Reproduce

### Command

```
./filter test.png output.png grayscale aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
```

## Suggested Fix Description
Use strncpy instead of strcpy.