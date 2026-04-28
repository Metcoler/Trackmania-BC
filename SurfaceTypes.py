from __future__ import annotations


EPLUG_SURFACE_MATERIAL_NAMES = {
    0: "Concrete",
    1: "Pavement",
    2: "Grass",
    3: "Ice",
    4: "Metal",
    5: "Sand",
    6: "Dirt",
    7: "Turbo_Deprecated",
    8: "DirtRoad",
    9: "Rubber",
    10: "SlidingRubber",
    11: "Test",
    12: "Rock",
    13: "Water",
    14: "Wood",
    15: "Danger",
    16: "Asphalt",
    17: "WetDirtRoad",
    18: "WetAsphalt",
    19: "WetPavement",
    20: "WetGrass",
    21: "Snow",
    22: "ResonantMetal",
    23: "GolfBall",
    24: "GolfWall",
    25: "GolfGround",
    26: "Turbo2_Deprecated",
    27: "Bumper_Deprecated",
    28: "NotCollidable",
    29: "FreeWheeling_Deprecated",
    30: "TurboRoulette_Deprecated",
    31: "WallJump",
    32: "MetalTrans",
    33: "Stone",
    34: "Player",
    35: "Trunk",
    36: "TechLaser",
    37: "SlidingWood",
    38: "PlayerOnly",
    39: "Tech",
    40: "TechArmor",
    41: "TechSafe",
    42: "OffZone",
    43: "Bullet",
    44: "TechHook",
    45: "TechGround",
    46: "TechWall",
    47: "TechArrow",
    48: "TechHook2",
    49: "Forest",
    50: "Wheat",
    51: "TechTarget",
    52: "PavementStair",
    53: "TechTeleport",
    54: "Energy",
    55: "TechMagnetic",
    56: "TurboTechMagnetic_Deprecated",
    57: "Turbo2TechMagnetic_Deprecated",
    58: "TurboWood_Deprecated",
    59: "Turbo2Wood_Deprecated",
    60: "FreeWheelingTechMagnetic_Deprecated",
    61: "FreeWheelingWood_Deprecated",
    62: "TechSuperMagnetic",
    63: "TechNucleus",
    64: "TechMagneticAccel",
    65: "MetalFence",
    66: "TechGravityChange",
    67: "TechGravityReset",
    68: "RubberBand",
    69: "Gravel",
    70: "Hack_NoGrip_Deprecated",
    71: "Bumper2_Deprecated",
    72: "NoSteering_Deprecated",
    73: "NoBrakes_Deprecated",
    74: "RoadIce",
    75: "RoadSynthetic",
    76: "Green",
    77: "Plastic",
    78: "DevDebug",
    79: "Free3",
    80: "XXX_Null",
}

SURFACE_TRACTION_BY_PREFIX = {
    "RoadTech": 1.00,
    "RoadDirt": 0.50,
    "PlatformDirt": 0.50,
    "PlatformGrass": 0.70,
    "PlatformPlastic": 0.75,
    "PlatformIce": 0.05,
    "PlatformSnow": 0.15,
}


def surface_material_name(material_id: float | int | None) -> str:
    if material_id is None:
        return "Unknown"
    try:
        numeric_id = int(round(float(material_id)))
    except (TypeError, ValueError, OverflowError):
        return "Unknown"
    return EPLUG_SURFACE_MATERIAL_NAMES.get(numeric_id, f"Unknown({numeric_id})")


def traction_for_surface_prefix(surface_prefix: str | None) -> float:
    if surface_prefix is None:
        return 1.0
    return float(SURFACE_TRACTION_BY_PREFIX.get(str(surface_prefix), 1.0))
