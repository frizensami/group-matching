# Genetic Algorithm for Group Formation

## Requirements
- `Python 3.x`
- `matplotlib`

## Usage
```
./genetic.py --help
```
This will display all the relevant arguments for this Python script

To run with the exact arguments from the associated paper (Population Size = 10000, Number of Generation = 5000, Elitism Factor = 3, Number of non-elite individuals to select from parent population = 3, Mutation Chance = 0.5, Number of swaps per mutation = 1, Number of participants = 27, Group size = 3), run

```
./genetic.py -p 10000 -g 5000 -el 3 -rest 3 -mchance 0.5 -mswaps 1 -n 27 -s 3
```
You will need a file called `rankings.csv` with a `n x n` matrix of preference values starting at the first row and first column. The value in row `i` and column `j` will refer to the rating given by the `i`th participant to the `j`th participant.

