from typing import Dict, Optional, Tuple, Type
from pathlib import Path
import uuid
import tempfile
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field
from langchain_core.callbacks import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain_core.tools import BaseTool

