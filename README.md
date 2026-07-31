# ft_linear_regression

Small linear-regression project written in Zig 0.16. It estimates a car price
from its mileage with the formula:

```text
estimated_price = theta0 + theta1 * mileage
```

## Quick correction

You need Zig `0.16.0`. From the root of the repository, build everything once:

```sh
zig build
```

This creates `train`, `predict`, `precision`, and `plot` in `zig-out/bin`.

### 1. Train the model

```sh
zig build train -- --data data/data.csv --weights weight.pt
```

The program starts from zero, trains with gradient descent, prints the loss
while it progresses, then saves `theta0` and `theta1` in `weight.pt`.

To try different training settings:

```sh
zig build train -- \
  --data data/data.csv \
  --weights weight.pt \
  --epochs 10000 \
  --learning-rate 0.1 \
  --epsilon 1e-10
```

### 2. Estimate a price

```sh
zig build predict -- --weights weight.pt
```

Enter a mileage when prompted:

```text
Enter mileage (km): 100000
Estimated price: 6354.70
```

### 3. Check the precision bonus

```sh
zig build precision -- --data data/data.csv --weights weight.pt
```

It prints:

- `R²`: proportion of price variation explained by mileage;
- `MAE`: average absolute prediction error;
- `RMSE`: error measure that penalizes large mistakes more strongly.

### 4. Open the plot bonus

Run this inside a real terminal:

```sh
zig build plot -- --data data/data.csv --weights weight.pt
```

The plot contains:

- yellow data points;
- the trained model in green;
- the mathematical ordinary-least-squares reference in blue;
- price on the horizontal axis and mileage on the vertical axis.

Press `q`, Escape, or Ctrl-C to close it.

## Useful correction commands

```sh
# Show every supported option
zig build train -- --help
zig build predict -- --help
zig build precision -- --help
zig build plot -- --help

# Compile every executable without running it
zig build check

# Run the unit tests
zig build test
```

After `zig build`, the installed programs can also be called directly:

```sh
./zig-out/bin/train --data data/data.csv
./zig-out/bin/predict
./zig-out/bin/precision
./zig-out/bin/plot
```

The default files are `data/data.csv` and `weight.pt`, so all flags are optional
for the supplied dataset.
