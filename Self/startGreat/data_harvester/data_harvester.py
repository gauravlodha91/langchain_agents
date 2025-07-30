from logger import logger
from fastapi import APIRouter, UploadFile, File
import pandas as pd
from typing import List
import importlib
router = APIRouter()

@router.get("/api/data-harvester-health", summary="Greet the world")
async def read_hello():
    """
    Returns a friendly greeting.
    """
    logger.info("Inside read hello")
    return {"message": "Hello, World from a separate file!"}


@router.post("/data-harvester", summary="Harvest file data")
async def data_harvester(files: List[UploadFile] = File(...)):
    response = {}

    for file in files:
        filename = file.filename
        ext = filename.split(".")[-1]
        logger.info(f"Processing {filename} with extension .{ext}")

        try:
            contents = await file.read()

            # Dynamically import chunking function
            chunker_module = importlib.import_module(f"chunkers.chunk_{ext}")
            chunk_func = getattr(chunker_module, f"chunk_{ext}")

            df: pd.DataFrame = chunk_func(contents)
            response[filename] = df.to_dict(orient="records")

        except ModuleNotFoundError:
            logger.error(f"No chunking logic for '.{ext}' files")
            response[filename] = {"error": f"No chunker for '.{ext}'"}
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            response[filename] = {"error": str(e)}

    return response
