# Camera Vectors Used in Experiments

Source: embodied/envs/dmc.py (custom walker camera injection).

## Side Camera (dmc_custom_walker_side)
- pos: `[0.0000, -3.0000, 1.0000]`
- x axis (xyaxes[0:3]): `[1.0000, 0.0000, 0.0000]`
- y axis (xyaxes[3:6]): `[0.0000, 0.0000, 1.0000]`
- z axis (x cross y): `[0.0000, -1.0000, 0.0000]`
- forward view direction (-z): `[-0.0000, 1.0000, -0.0000]`
- unit vector camera->torso: `[0.0000, 0.9487, -0.3162]`
- forward dot camera->torso: `0.9487`

## Top Camera (dmc_custom_walker_top)
- pos: `[0.0000, 0.0000, 4.0000]`
- x axis (xyaxes[0:3]): `[0.0000, -1.0000, 0.0000]`
- y axis (xyaxes[3:6]): `[1.0000, 0.0000, 0.0000]`
- z axis (x cross y): `[-0.0000, 0.0000, 1.0000]`
- forward view direction (-z): `[0.0000, -0.0000, -1.0000]`
- unit vector camera->torso: `[0.0000, 0.0000, -1.0000]`
- forward dot camera->torso: `1.0000`
