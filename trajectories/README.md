# Trajectories

This folder contains tools and an API for dealing with raw or processed (exclusive) trajectories.

## Trajectory Formats

### Raw

This format contains raw observations for a list of time stamp saved in a json file.

Format:

```json
[
    [
        <timestamp>,
        [
            [
                [
                    <x>,
                    <y>
                ],
                <detection_confidence>,
                <identifiy>,
                <identification_confidence>
            ],
            ...
        ]
    ],
    ...
]
```

Example:

```json
[
    [
        1587968731.0,
        [
            [
                [
                    1450.3112386469952,
                    1467.457150376693
                ],
                0.9378536939620972,
                "Nanuq",
                1.0
            ],
            [
                [
                    1817.726278815526,
                    1753.3636793289475
                ],
                0.9405180215835571,
                "Vera",
                1.0
            ]
        ]
    ],
    [
        ...
    ],
    ...
]
```

### Exclusive

Format for processed trajectories with only one position for every identity saved in a csv file.
For further details have a closer look at the main [README](../README.md) of this repo.

Format:

```csv
time,x1,y1,x2,y2
<timestamp>,<x1>,<y1>,<x2>,<y2>
...
```

Example:

```csv
time,x1,y1,x2,y2
1629800864.60,0.3,0.4,0.1,0.1
1629800865.71,0.3,0.4,0.1,0.1
1629800865.71,0.3,0.4,-1,-1
...
```

## Tools

- [plot_trajectory.py](plot_trajectory.py): Plot processed trajectories or apply a default processing to a raw trajectory an plot it afterwards
- [process_trajectory.py](process_trajectory.py): Process raw trajectories into the exclusive format
