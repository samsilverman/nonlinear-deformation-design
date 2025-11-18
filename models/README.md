# Models

This folder contains all trained networks from our experiments.
We repeated training ten times on different random splits and report test metrics with 95% confidence intervals.

The following random seeds were used for training:

1. 100
2. 7773
3. 5924
4. 7789
5. 6587
6. 4910
7. 9974
8. 3577
9. 6485
10. 6224

## Files

| File | Description |
| - | - |
| `forward/[seed].pt` | The forward design network trained with random seed `[seed]`. |
| `tandem/[seed]-[⍺].pt` | The tandem neural network trained with random seed `[seed]` and ⍺ value `[⍺]`. Refer to the paper for information on ⍺.  |
