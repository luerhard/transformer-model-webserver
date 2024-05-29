import gc
from configparser import ConfigParser
from pathlib import Path
import os

PATH = Path(__file__).parents[1]

config = ConfigParser()
config.read(PATH / "config.ini")

os.environ["TOKENIZERS_PARALLELISM"] = "false"


allocs, gen1, gen2 = gc.get_threshold()
allocs = 25_000
gc.set_threshold(allocs, gen1, gen2)
