import os

if os.getenv("COLAB_RELEASE_TAG"):
    import google.colab as cl


def get_content_path(base_path: str) -> str:
    """
    Returns the full path for content depending on platform.
    :param base_path: The base path of content
    :return: The full path for looking up content
    """
    return os.path.join("/content/drive/MyDrive" if is_on_drive() else "../", base_path)


def is_on_drive() -> bool:
    """
    Checks whether the platform is Google Colaboratary by checking for release tag
    :return: A boolean stating whether the platform is colab or not
    """

    return os.getenv("COLAB_RELEASE_TAG") is not None


def mount(path: str) -> None:
    """
    Mounts the drive folder. Only inteded to be used on colab platform.
    :param path: The path to mount
    """

    cl.drive.mount(path, force_remount=False)
