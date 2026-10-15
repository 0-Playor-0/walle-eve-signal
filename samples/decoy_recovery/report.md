# Mission report

- **Outcome:** rescued after 126 turns
- **Battery left:** 22.4%  (recharges: 0)
- **Inferred location:** agent frame [59, 113] (map offset not fully resolved) - belief mass within 2 cells 0.9957
- **Status:** alive (rescued)
- **Estimated remaining life:** mean 76.2 turns (10-90%: 20.7-143.7)
- **Confidence:** 1.0  (CV confirmed: True)
- **Recommended next step:** Escort the target to the hearth at [99, 99] (71 moves), keep them warm, and call for medical support.

## Sources tracked

| track | position | hits | CV labels | interpretation | speed |
|---|---|---|---|---|---|
| 0 | [92.93, 100.54] | {'box': 5, 'lidar': 5, 'thermal': 5} | - | hearth? | 0.0 |
| 1 | [71.03, 105.9] | {'humanoid': 1, 'lidar': 4, 'thermal': 4, 'small': 3, 'cv': 1} | {'fire': 1} | torch | 0.072 |
| 2 | [59.0, 113.0] | {'thermal': 4, 'radar': 1, 'humanoid': 2, 'lidar': 2, 'cv': 1} | {'target': 1} | target | 0.113 |
| 3 | [61.86, 97.2] | {'thermal': 4} | - | warm source? | 0.0 |
| 4 | [71.0, 114.0] | {'humanoid': 2, 'lidar': 2} | - | humanoid (armour or target)? | 0.0 |
| 5 | [33.47, 93.81] | {'thermal': 2} | - | warm source? | 0.0 |
| 6 | [108.0, 110.0] | {'thermal': 1} | - | warm source? | 0.0 |

## Evidence log

| turn | sensor | position | track | track's interpretation (at the end) |
|---|---|---|---|---|
| 6 | lidar | [99.0, 99.0] | 0 | hearth? |
| 9 | lidar | [99.0, 99.0] | 0 | hearth? |
| 14 | lidar | [99.0, 99.0] | 0 | hearth? |
| 17 | lidar | [99.0, 99.0] | 0 | hearth? |
| 20 | lidar | [99.0, 99.0] | 0 | hearth? |
| 67 | lidar | [71.0, 106.0] | 1 | torch |
| 68 | thermal | [103.0, 97.0] | 0 | hearth? |
| 68 | thermal | [99.0, 106.0] | 0 | hearth? |
| 68 | thermal | [71.0, 105.0] | 1 | torch |
| 68 | thermal | [59.0, 105.0] | 2 | target |
| 68 | thermal | [65.0, 89.0] | 3 | warm source? |
| 72 | lidar | [71.0, 106.0] | 1 | torch |
| 80 | lidar | [71.0, 106.0] | 1 | torch |
| 80 | lidar | [71.0, 114.0] | 4 | humanoid (armour or target)? |
| 83 | thermal | [94.0, 102.0] | 0 | hearth? |
| 83 | thermal | [69.0, 107.0] | 1 | torch |
| 83 | thermal | [60.0, 112.0] | 2 | target |
| 83 | thermal | [62.0, 93.0] | 3 | warm source? |
| 83 | thermal | [23.0, 90.0] | 5 | warm source? |
| 83 | thermal | [108.0, 110.0] | 6 | warm source? |
| 87 | cv | [71.0, 106.0] | 1 | torch |
| 90 | lidar | [71.0, 106.0] | 1 | torch |
| 90 | lidar | [71.0, 114.0] | 4 | humanoid (armour or target)? |
| 97 | thermal | [99.0, 104.0] | 0 | hearth? |
| 97 | thermal | [70.0, 105.0] | 1 | torch |
| 97 | thermal | [56.0, 117.0] | 2 | target |
| 97 | thermal | [60.0, 100.0] | 3 | warm source? |
| 109 | thermal | [92.0, 100.0] | 0 | hearth? |
| 109 | thermal | [73.0, 104.0] | 1 | torch |
| 109 | thermal | [58.0, 112.0] | 2 | target |
| 109 | thermal | [62.0, 97.0] | 3 | warm source? |
| 109 | thermal | [34.0, 94.0] | 5 | warm source? |
| 111 | radar | [61.0, 116.0] | 2 | target |
| 117 | lidar | [60.0, 113.0] | 2 | target |
| 121 | lidar | [59.0, 113.0] | 2 | target |
| 124 | cv | [59.0, 113.0] | 2 | target |

## Battery spend

- lidar: 4.2%
- move: 19.0%
- turn: 4.4%
- thermal: 20.0%
- radar: 6.0%
- cv: 24.0%

## Endgame overrides

- t=123: scores {'safe': 0.0295, 'dash': 0.2118, 'lean': 0.2214} -> **lean**

## Ground truth comparison

_Appended by the evaluator; the agent never saw any of this._

- True target position (final): agent frame [59, 112], map [37, 89]
- Agent's inferred position: [59, 113] -> error 1 cells (Manhattan)
- True energy at the end: 0.220; true remaining life 41.37 turns vs estimated 76.2
- Mission life budget: 174 turns (walking distance 69); used 126 turns
- Belief mass within 2 cells of the truth: final 0.995, best 0.998
- Failure cause: n/a (rescued)

| sensor | target | dog | torch | hearth | armour | none |
|---|---|---|---|---|---|---|
| thermal | 3 | 0 | 9 | 4 | 1 | 3 |
| radar | 1 | 0 | 0 | 0 | 0 | 0 |
| lidar_humanoid | 2 | 0 | 1 | 0 | 2 | 0 |
| cv_target | 1 | 0 | 0 | 0 | 0 | 0 |

CV labels by true entity: {'target': {'target': 1}, 'dog': {}, 'torch': {'fire': 1}, 'hearth': {}, 'armour': {}}
