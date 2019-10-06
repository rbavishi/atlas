import os
import pickle
import tempfile
import shutil
import urllib.request

import cloudpickle
from atlas.models import AtlasModel
from typing import Type


def save_model(model: AtlasModel, path: str, no_zip=False):
    """
    Save model as a ZIP or a directory at path. Path may or may not contain the .zip extension
    Args:
        model (AtlasModel): Model to save
        path (str): Path to save the model at. May or may not have the .zip extension
        no_zip (bool): Do not create a zip
    """

    work_dir = None
    try:
        #  First create a temporary directory to put all contents in
        work_dir = tempfile.mkdtemp()

        with open(f"{work_dir}/serialized.pkl", "wb") as f:
            cloudpickle.dump(model, f)

        #  Now let the model save whatever it wants
        model.serialize(work_dir)

        if no_zip:
            shutil.copytree(work_dir, path)
            return

        #  Package it up into a zip and clean up the directory
        if path.endswith(".zip"):
            path = path[:-len(".zip")]

        shutil.make_archive(path, "zip", work_dir)

    finally:
        if work_dir is not None:
            shutil.rmtree(work_dir)


def restore_model(path: str, from_url=False) -> AtlasModel:
    """
    Deserialize the data in the ZIP/directory at path and load it back into the base model
    Args:
        path (str): Path to ZIP/directory containing serialized model data
        from_url (bool): Whether the path is hyperlink

    Returns:
        The loaded model
    """

    if from_url:
        tmp_zip = None
        try:
            _, tmp_zip = tempfile.mkstemp(suffix=".zip")

            with urllib.request.urlopen(path) as resp, open(tmp_zip, "wb") as f:
                shutil.copyfileobj(resp, f)

            return restore_model_from_zip(tmp_zip)

        finally:
            if tmp_zip is not None:
                os.unlink(tmp_zip)

    elif path.endswith(".zip"):
        return restore_model_from_zip(path)

    else:
        return restore_model_from_directory(path)


def restore_model_from_zip(path: str) -> AtlasModel:
    work_dir = None
    try:
        #  First create a temporary directory to put all contents in
        work_dir = tempfile.mkdtemp()
        shutil.unpack_archive(path, work_dir, "zip")

        return restore_model_from_directory(work_dir)

    finally:
        if work_dir is not None:
            shutil.rmtree(work_dir)


def restore_model_from_directory(path: str) -> AtlasModel:
    with open(f"{path}/serialized.pkl", "rb") as f:
        model = cloudpickle.load(f)

    #  Now let the model save whatever it wants
    model.deserialize(path)

    return model
