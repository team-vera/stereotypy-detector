
POLAR_NAMES = [
    "Nanuq",
    "Vera"
]

POLAR_IDX_MAPPING = {
    "Nanuq": 0,
    "Vera": 1
}

POLAR_RGB_COLOR_MAPPING = {
    "Nanuq": [0, 0, 255],
    "Vera": [255, 0, 0],
    "Unknown": [255, 125, 0],
}

POLAR_BGR_COLOR_MAPPING = {
    "Nanuq": [255, 0, 0],
    "Vera": [0, 0, 255],
    "Unknown": [0, 125, 255],
}

POLAR_COLOR_MAPPING = {
    "RGB": POLAR_RGB_COLOR_MAPPING,
    "BGR": POLAR_BGR_COLOR_MAPPING
}

# meters per pixel
MPP = 0.041
# max speed of polar bear in meter per second (~40km/h)
MAX_SPEED_PB = 11.1
# max speed of polar bear in pixels per second (~40km/h)
MAX_SPEED_PB_PIXELS = MAX_SPEED_PB / MPP
