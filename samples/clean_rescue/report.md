# Mission report

- **Outcome:** rescued after 61 turns
- **Battery left:** 60.0%  (recharges: 0)
- **Inferred location:** agent frame [79, 74] (map offset not fully resolved) - belief mass within 2 cells 0.997
- **Status:** alive (rescued)
- **Estimated remaining life:** mean 5.2 turns (10-90%: 0.0-18.4)
- **Confidence:** 1.0  (CV confirmed: True)
- **Recommended next step:** Escort the target to the hearth at [99, 99] (44 moves), keep them warm, and call for medical support.

## Sources tracked

| track | position | hits | CV labels | interpretation | speed |
|---|---|---|---|---|---|
| 0 | [98.25, 98.25] | {'box': 2, 'lidar': 2, 'thermal': 2} | - | hearth? | 0.0 |
| 1 | [80.0, 74.0] | {'thermal': 2, 'radar': 3, 'humanoid': 2, 'lidar': 2, 'cv': 1} | {'target': 1} | target | 0.041 |
| 2 | [107.31, 83.24] | {'thermal': 4} | - | warm source? | 0.0 |
| 3 | [59.0, 86.0] | {'thermal': 1} | - | warm source? | 0.0 |

## Evidence log

| turn | sensor | position | track | track's interpretation (at the end) |
|---|---|---|---|---|
| 1 | lidar | [99.0, 99.0] | 0 | hearth? |
| 3 | thermal | [99.0, 99.0] | 0 | hearth? |
| 3 | thermal | [88.0, 71.0] | 1 | target |
| 3 | thermal | [102.0, 84.0] | 2 | warm source? |
| 3 | thermal | [109.0, 86.0] | 2 | warm source? |
| 3 | thermal | [59.0, 86.0] | 3 | warm source? |
| 10 | lidar | [99.0, 99.0] | 0 | hearth? |
| 15 | thermal | [98.0, 98.0] | 0 | hearth? |
| 15 | thermal | [83.0, 80.0] | 1 | target |
| 15 | thermal | [108.0, 78.0] | 2 | warm source? |
| 15 | thermal | [107.0, 86.0] | 2 | warm source? |
| 31 | radar | [83.0, 73.0] | 1 | target |
| 39 | radar | [80.0, 73.0] | 1 | target |
| 48 | lidar | [80.0, 74.0] | 1 | target |
| 49 | radar | [79.0, 75.0] | 1 | target |
| 53 | lidar | [80.0, 74.0] | 1 | target |
| 59 | cv | [80.0, 74.0] | 1 | target |

## Battery spend

- lidar: 1.8%
- radar: 10.0%
- thermal: 10.0%
- move: 8.8%
- turn: 1.4%
- cv: 8.0%

## Endgame overrides

None fired.

## Ground truth comparison

_Appended by the evaluator; the agent never saw any of this._

- True target position (final): agent frame [80, 74], map [25, 43]
- Agent's inferred position: [79, 74] -> error 1 cells (Manhattan)
- True energy at the end: 0.460; true remaining life 53.95 turns vs estimated 5.2
- Mission life budget: 116 turns (walking distance 44); used 61 turns
- Belief mass within 2 cells of the truth: final 0.997, best 0.999
- Failure cause: n/a (rescued)

| sensor | target | dog | torch | hearth | armour | none |
|---|---|---|---|---|---|---|
| thermal | 1 | 0 | 4 | 2 | 0 | 2 |
| radar | 3 | 0 | 0 | 0 | 0 | 0 |
| lidar_humanoid | 2 | 0 | 0 | 0 | 0 | 0 |
| cv_target | 1 | 0 | 0 | 0 | 0 | 0 |

CV labels by true entity: {'target': {'target': 1}, 'dog': {}, 'torch': {}, 'hearth': {}, 'armour': {}}
