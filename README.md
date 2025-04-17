# Example Plugin

This plugin demonstrates three different types of nodes in the workflow system.

## Node Types

### 1. ExampleNode
A basic node that processes string input:
- Input: Takes a string parameter
- Output: Returns the input string with success status
- Use Case: Demonstrates basic node structure and error handling

### 2. ExampleGeneratorNode
A generator node that produces a sequence of numbers:
- Inputs:
  - Start Number (default: 0)
  - End Number (default: 10)
  - Step Size (default: 1)
- Output: Generates numbers in sequence
- Use Case: Useful for batch processing or creating number sequences

### 3. ExampleConditionNode
A conditional node that checks if a number is even:
- Input: Takes an integer number
- Outputs: Has two branches
  - Even Branch (true_branch): Activated when number is even
  - Odd Branch (false_branch): Activated when number is odd
- Use Case: Demonstrates workflow branching based on conditions

## Running Tests

To run the test cases:

```bash
python example.py
```

This will execute the test cases and print the results to the console.