import os


DATASET_ALIASES = {
    "LEVIR": "./levir_cd_256",
    "WHU": "./whu_cd_256",
    "CLCD": "./clcd_256",
    "OSCD": "./oscd_256",
}


def resolve_dataset_root(file_root):
    """Resolve a legacy dataset alias or return a user-provided path."""
    key = file_root.upper()
    return DATASET_ALIASES.get(key, os.path.expanduser(file_root))


def dataset_tag(file_root):
    """Return a filesystem-safe tag for result directories."""
    key = file_root.upper()
    if key in DATASET_ALIASES:
        return key
    normalized = os.path.normpath(os.path.expanduser(file_root))
    tag = os.path.basename(normalized) or "custom"
    for char in (":", "\\", "/", " "):
        tag = tag.replace(char, "_")
    return tag
