# BUG-14
## Category
Unchecked system call returning code

## Description

The program fails to check if dynamic memory allocation
succeeded after the palette is dynamically allocated.

## Affected Lines in the original program
In `solid.c:16`

## Expected vs Observed
It is expected that if the dynamic memory allocation for the palette fails,
an error should be handled, resulting in graceful program termination.
However, No error handling is performed upon dynamic allocation failure.

## Steps to Reproduce

### Command

## Suggested Fix Description
Add code to check if malloc was successful. This typically involves checking
if the returned pointer is NULL and, if so, returning an error code or exiting the program.