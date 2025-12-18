# BUG-11
## Category
Command injection

## Description

The system function is used with the command variable as an argument,
and since the contents of command are used directly, shell commands can be executed.


## Affected Lines in the original program
In `solid.c:125`

## Expected vs Observed
It is expected that the file size will be output normally, and an error will occur
for an invalid file name. However, the shell command is executed directly.

## Steps to Reproduce

### Command

```
./solid ";id" 10 10 ffffff
```

## Suggested Fix Description
Use the stat function instead of the system function.