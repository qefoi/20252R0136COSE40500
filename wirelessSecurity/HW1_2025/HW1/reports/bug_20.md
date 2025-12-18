# BUG-20
## Category
String Vulnerability

## Description

A format string vulnerability occurs because printf
is used without a format specifier.

## Affected Lines in the original program
In `filter.c:230`

## Expected vs Observed
The input value is expected to be printed. However,
if the input contains format specifiers like %s,
a segmentation fault occurs.

## Steps to Reproduce

### Command

```
./filter %s.png output.png negative
```

## Suggested Fix Description
Replace printf(input) with printf("%s", input).