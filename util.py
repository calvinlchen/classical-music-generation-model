import os
import torch


def mkdir(dir: str):
    """
    Create directory at dir, if one does not exist.

    :param dir: directory to make
    :type dir: str
    """
    os.makedirs(dir, exist_ok=True)


def path_join(dir: str, filename: str):
    """
    Join filename to a directory path

    :param dir: directory for file
    :type dir: str
    :param filename: filename with extension
    :type filename: str
    :return: full filepath, including directory and filename
    """
    return os.path.join(dir, filename)


def write_txt_file(
        dir: str,
        filename: str,
        text: str,
        mkdir: bool = False
):
    """
    Write a .txt file with given text, under the given
    filename and directory.

    :param dir: directory
    :type dir: str
    :param filename: filename (with or without .txt)
    :type filename: str
    :param text: text to write to file
    :type text: str
    :param mkdir: define whether to make the directory if it doesn't exist
    :type mkdir: bool
    """
    if filename[-4:] != ".txt":
        filename = f"{filename}.txt"

    if mkdir:
        mkdir(dir)

    with open(path_join(dir, filename), "w", encoding="utf-8") as f:
        f.write(text)
    return


def get_best_device():
    return "cuda" if torch.cuda.is_available() else "cpu"
