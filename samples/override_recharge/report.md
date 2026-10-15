# Mission report

- **Outcome:** rescued after 139 turns
- **Battery left:** 89.8%  (recharges: 1)
- **Inferred location:** agent frame [115, 140] (map offset not fully resolved) - belief mass within 2 cells 0.9648
- **Status:** alive (rescued)
- **Estimated remaining life:** mean 92.0 turns (10-90%: 31.0-165.1)
- **Confidence:** 1.0  (CV confirmed: True)
- **Recommended next step:** Escort the target to the hearth at [118, 135] (7 moves), keep them warm, and call for medical support.

## Sources tracked

| track | position | hits | CV labels | interpretation | speed |
|---|---|---|---|---|---|
| 0 | [112.42, 81.62] | {'thermal': 5, 'box': 3, 'lidar': 3} | - | hearth? | 0.148 |
| 1 | [47.0, 103.0] | {'thermal': 1} | - | warm source? | 0.0 |
| 2 | [81.76, 91.07] | {'thermal': 4} | - | warm source? | 0.0 |
| 3 | [83.0, 87.0] | {'thermal': 1} | - | warm source? | 0.0 |
| 4 | [115.0, 139.0] | {'thermal': 12, 'radar': 2, 'humanoid': 2, 'lidar': 2, 'cv': 1} | {'target': 1} | target | 0.072 |
| 5 | [128.0, 131.0] | {'thermal': 6, 'small': 4, 'lidar': 4} | - | fire? | 0.077 |
| 6 | [67.0, 124.0] | {'thermal': 1} | - | warm source? | 0.0 |
| 7 | [118.0, 135.0] | {'box': 3, 'lidar': 3} | - | hearth? | 0.0 |

## Evidence log

| turn | sensor | position | track | track's interpretation (at the end) |
|---|---|---|---|---|
| 1 | thermal | [98.0, 99.0] | 0 | hearth? |
| 1 | thermal | [47.0, 103.0] | 1 | warm source? |
| 1 | thermal | [106.0, 84.0] | 2 | warm source? |
| 1 | thermal | [83.0, 87.0] | 3 | warm source? |
| 2 | lidar | [99.0, 99.0] | 0 | hearth? |
| 10 | lidar | [99.0, 99.0] | 0 | hearth? |
| 14 | lidar | [99.0, 99.0] | 0 | hearth? |
| 30 | thermal | [101.0, 99.0] | 0 | hearth? |
| 30 | thermal | [103.0, 71.0] | 2 | warm source? |
| 30 | thermal | [121.0, 136.0] | 4 | target |
| 30 | thermal | [116.0, 132.0] | 4 | target |
| 30 | thermal | [132.0, 132.0] | 5 | fire? |
| 30 | thermal | [67.0, 124.0] | 6 | warm source? |
| 43 | thermal | [100.0, 96.0] | 0 | hearth? |
| 43 | thermal | [93.0, 76.0] | 2 | warm source? |
| 43 | thermal | [114.0, 136.0] | 4 | target |
| 43 | thermal | [117.0, 132.0] | 4 | target |
| 43 | thermal | [126.0, 129.0] | 5 | fire? |
| 60 | thermal | [105.0, 89.0] | 0 | hearth? |
| 60 | thermal | [120.0, 135.0] | 4 | target |
| 60 | thermal | [117.0, 135.0] | 4 | target |
| 60 | thermal | [132.0, 126.0] | 5 | fire? |
| 74 | thermal | [114.0, 80.0] | 0 | hearth? |
| 74 | thermal | [81.0, 92.0] | 2 | warm source? |
| 74 | thermal | [115.0, 137.0] | 4 | target |
| 74 | thermal | [116.0, 138.0] | 4 | target |
| 74 | thermal | [129.0, 133.0] | 5 | fire? |
| 87 | thermal | [110.0, 139.0] | 4 | target |
| 87 | thermal | [116.0, 138.0] | 4 | target |
| 87 | thermal | [128.0, 128.0] | 5 | fire? |
| 103 | lidar | [128.0, 131.0] | 5 | fire? |
| 111 | thermal | [114.0, 138.0] | 4 | target |
| 111 | thermal | [117.0, 132.0] | 4 | target |
| 111 | thermal | [130.0, 132.0] | 5 | fire? |
| 112 | radar | [116.0, 138.0] | 4 | target |
| 116 | lidar | [128.0, 131.0] | 5 | fire? |
| 121 | lidar | [128.0, 131.0] | 5 | fire? |
| 124 | lidar | [114.0, 140.0] | 4 | target |
| 124 | lidar | [128.0, 131.0] | 5 | fire? |
| 124 | lidar | [118.0, 135.0] | 7 | hearth? |
| 125 | radar | [114.0, 139.0] | 4 | target |
| 130 | lidar | [118.0, 135.0] | 7 | hearth? |
| 135 | lidar | [115.0, 139.0] | 4 | target |
| 135 | lidar | [118.0, 135.0] | 7 | hearth? |
| 138 | cv | [115.0, 139.0] | 4 | target |

## Battery spend

- thermal: 35.0%
- lidar: 4.0%
- move: 20.4%
- turn: 5.0%
- radar: 14.0%
- charge: 0.0%
- cv: 8.0%

## Endgame overrides

- t=119: scores {'safe': 0.0625, 'dash': 0.4459, 'lean': 0.4989} -> **lean**

## Ground truth comparison

_Appended by the evaluator; the agent never saw any of this._

- True target position (final): agent frame [115, 139], map [85, 85]
- Agent's inferred position: [115, 140] -> error 1 cells (Manhattan)
- True energy at the end: 0.359; true remaining life 87.93 turns vs estimated 92.0
- Mission life budget: 236 turns (walking distance 60); used 139 turns
- Belief mass within 2 cells of the truth: final 0.965, best 0.966
- Failure cause: n/a (rescued)

| sensor | target | dog | torch | hearth | armour | none |
|---|---|---|---|---|---|---|
| thermal | 6 | 3 | 7 | 9 | 0 | 5 |
| radar | 2 | 0 | 0 | 0 | 0 | 0 |
| lidar_humanoid | 2 | 0 | 0 | 0 | 0 | 0 |
| cv_target | 1 | 0 | 0 | 0 | 0 | 0 |

CV labels by true entity: {'target': {'target': 1}, 'dog': {}, 'torch': {}, 'hearth': {}, 'armour': {}}
