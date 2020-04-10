## Features

There are 3 different level of functions here, categorized by the level of INPUT data.

- Lv0: inputs are tick data, then returns a summary.
- Lv1: inputs are outputs of lv.0, which are certain things(example: end price).
- Lv2: inputs can be lv0, lv.1 or above. Examples:
    - prices
    - STDs (within each line, which is counted by smaller units)
    - volumes
    - orders count(1, plus/minus, 5, all)

