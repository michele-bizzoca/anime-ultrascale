########################################################################################
# Imports
########################################################################################

# Standard Qualified
import sys
import math
import json
import subprocess
import atexit
import re

# Standard Unqualified
from os import getpid
from pathlib import Path
from enum import IntEnum
from typing import NoReturn
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import cast, Callable, TypeVar
from types import SimpleNamespace

# Custom Qualified
from PIL import Image
from psutil import Process
from dacite import from_dict, Config

########################################################################################
# Constants
########################################################################################

SOFTWARE_VERSION = "1.0"
OUTPUT_FORMAT    = "PNG"
OUTPUT_MODE      = "RGBA"
TEMP_FOLDER      = "temp"
MODEL_FOLDER     = "models"
SESSION_FOLDER   = "sessions"
RENV_FOLDER      = "renv"
SESSION_FILE     = "session.json"
SETTINGS_FILE    = "settings.cfg"
LOG_FILE         = "log.txt"
TEMP_INPUT_FILE  = "in.png"
TEMP_OUTPUT_FILE = "output.png"
RENV_FILE        = "realesrgan-ncnn-vulkan"
INVOCATION_FILE  = "call.txt"
MAX_MPX          = 150

########################################################################################
# Phases
########################################################################################

phases = [

    ( "input"     , ["import"    , "downscale"]) ,
    ( "main"      , ["upscale"   , "downscale"]) ,
    ( "soft"      , ["downscale" , "upscale"  ]) ,
    ( "hard"      , ["downscale" , "upscale"  ]) ,
    ( "output"    , ["resize"                 ])
]

########################################################################################
# Type Variables
########################################################################################

T = TypeVar("T")

########################################################################################
# Arguments
########################################################################################

class Argument(IntEnum):

    script_filepath   = 0
    input_filepath    = 1
    output_filepath   = 2

class PathOption(IntEnum):
    settings_filepath = 0

class PositionalOption(IntEnum):
    main_divisor      = 0
    main_multiplier   = 1
    soft_model        = 2
    soft_divisor      = 3
    soft_multiplier   = 4
    soft_iterations   = 5
    hard_model        = 6
    hard_divisor      = 7
    hard_multiplier   = 8
    hard_iterations   = 9

########################################################################################
# Session Data
########################################################################################

@dataclass
class Invocation:

    time    : str
    version : str

@dataclass
class ImageInfo:

    format : str
    mode   : str
    width  : int
    height : int

@dataclass
class MainSettings:

    divisor    : float
    multiplier : float

@dataclass
class ModelSettings:

    model      : str
    divisor    : float
    multiplier : int
    iterations : int

@dataclass
class Settings:

    main : MainSettings
    soft : ModelSettings
    hard : ModelSettings

@dataclass
class Session:

    invocation : Invocation
    input      : ImageInfo
    output     : ImageInfo
    settings   : Settings

########################################################################################
# Invocation Date and Time
########################################################################################

invocation_instant        = datetime.fromtimestamp(Process().create_time())
invocation_date           = invocation_instant.strftime('%Y-%m-%d')
invocation_time           = invocation_instant.strftime('%H-%M-%S-%f')
invocation_pid            = getpid()
invocation_time_pid       = f"{invocation_time}-{invocation_pid}"
invocation_date_time_pid  = f"{invocation_date}-{invocation_time_pid}"
invocation_date_time      = f"{invocation_date}-{invocation_time}"

########################################################################################
# Session Folder
########################################################################################

session_folder  = Path(SESSION_FOLDER) / invocation_date / invocation_time_pid
session_folder.mkdir(parents = True, exist_ok = True)

########################################################################################
# Invocation File Creation
########################################################################################

invocation_file = session_folder / INVOCATION_FILE

with open(invocation_file, "w") as invocation_handle:
    invocation_handle.write( f"PID: {invocation_pid}\n"             +
                             f"Timestamp: {invocation_date_time}\n" +
                             f"PWD: {Path.cwd}\n"                   +
                             f"Command: {' '.join(sys.argv)}\n"     )

########################################################################################
# Logging
########################################################################################

log_file = session_folder / LOG_FILE
log_handle = open(log_file, "w")

def log(message: str, level: str = "NORMAL"):
    log_handle.write( datetime.now().strftime('on %Y/%m/%d at %H:%M:%S') +
                      f", level {level}: {message}\n"                    )
    log_handle.flush()

def close_log_file():
    log_handle.close()

atexit.register(close_log_file)

log("log system is operative")

########################################################################################
# Temporary Files Management
########################################################################################

temp_folder = Path(TEMP_FOLDER) / invocation_date_time_pid
temp_folder.mkdir(parents = True, exist_ok=True)

def clean_temporaries() -> None:
    for file in temp_folder.iterdir():
        if file.is_file():
            file.unlink()

def clean_temp_folder() -> None:
    temp_folder.rmdir()

atexit.register(clean_temp_folder)
atexit.register(clean_temporaries)

log("the temporary file system is operative")

########################################################################################
# Error Reporting
########################################################################################

def fail(message: str) -> NoReturn:
    log(message, "ERROR")
    raise SystemExit( message[:1].upper()                        +
                      message[1:]                                +
                      ". Run with --help for usage information." )

def assume(condition : bool, message : str) -> None:
    if not condition:
        fail(message)

log("the fail system is operative")

########################################################################################
# Reserved Paths
########################################################################################

renv_folder      = Path(RENV_FOLDER)
model_folder     = Path(MODEL_FOLDER)
session_file     = session_folder / SESSION_FILE
settings_file    = session_folder / SETTINGS_FILE
temp_input_file  = temp_folder    / TEMP_INPUT_FILE
temp_output_file = temp_folder    / TEMP_OUTPUT_FILE
renv_file        = renv_folder    / RENV_FILE

########################################################################################
# Preliminary I/O
########################################################################################

assume(len(sys.argv) >= len(Argument), "incomplete I/O specification")

input_file  = Path(sys.argv[Argument.input_filepath])
output_file = Path(sys.argv[Argument.output_filepath])

log("I/O paths have been loaded")

with Image.open(input_file) as input_image:
    assume(input_image.format is not None, "unrecognized image format")
    input_format              = cast(str, input_image.format)
    input_width, input_height = input_image.size
    input_mode                = input_image.mode
    input_image               = input_image.copy()

log("the input image has been loaded")

########################################################################################
# State
########################################################################################

state = SimpleNamespace(image = input_image)

########################################################################################
# Nested Dictionary Underscore-Based Flattening
########################################################################################

def flatten(data: dict[str, object]) -> dict[str, object]:

    def flatten_(data_: dict[str, object], prefix: str) -> dict[str, object]:

        result = {}

        for key, value in data_.items():
            joined_key = f"{prefix}_{key}" if prefix else key

            if isinstance(value, dict):
                result.update(flatten_(value, joined_key))
            else:
                result[joined_key] = value

        return result

    return flatten_(data, "")

########################################################################################
# Nested Dictionary Underscore-Based Unflattening
########################################################################################

def unflatten(data: dict[str, object]) -> dict[str, object]:

    result = {}

    for key, value in data.items():

        current = result
        parts   = key.split("_")

        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        current[parts[-1]] = value

    return result

########################################################################################
# Session Export
########################################################################################

def export_session(s: Session) -> str:

    return json.dumps(asdict(s), indent = 4, sort_keys = False)

########################################################################################
# Session Import
########################################################################################

def import_session(s : str) -> Session:

    return from_dict( data_class = Session,
                      data       = json.loads(s),
                      config     = Config(check_types=True) )

########################################################################################
# Settings Export
########################################################################################

def export_settings(s: Settings) -> str:

    result: str = ""

    for key, value in flatten(asdict(s)).items():
        result += f"{key} = {json.dumps(value)}\n"

    return result

########################################################################################
# Settings Import
########################################################################################

def import_settings(s : str) -> Settings:

    s = re.sub(r'^\s*(#.*)?$\n?', '', s, flags = re.MULTILINE)
    s = re.sub(r'^\s*(\w+)\s*=([^#]*)(#.*)?$', r'"\1": \2,', s, flags=re.MULTILINE)
    s = "{" + s[:-1] + "}"

    return from_dict( data_class = Settings,
                      data = unflatten(json.loads(s)),
                      config = Config(check_types=True) )

########################################################################################
# Path Options -> Settings
########################################################################################

def settings_from_path_options(args: list[str]) -> Settings:

    path      = Path(args[PathOption.settings_filepath])
    extension = path.suffix[1:]

    with open(path) as handle:
        text = handle.read()

    if extension == "json":
        return import_session(text).settings
    elif extension == "cfg":
        return import_settings(text)
    else:
        fail("settings file type is unknown")

########################################################################################
# Positional Options -> Settings
########################################################################################

def str_from_posi_options(opts: list[str], index: PositionalOption) -> str:

    name = index.name.replace("_", " ")
    assume(index < len(opts), f'the argument {name} is missing')
    return opts[index]

def number_from_posi_options (

    opts      : list[str],
    index     : PositionalOption,
    read      : Callable[[str], T],
    modifiers : str

) -> tuple[T, str | None]:

    text = str_from_posi_options(opts, index)
    name = index.name.replace("_", " ")

    assume(text != "", f'the argument {name} is empty')
    assume(len(text) > 1 or text not in modifiers, f'the argument {name} has no value')

    prefix = text[0]  if text[0]  in modifiers else None
    suffix = text[-1] if text[-1] in modifiers else None

    assume( prefix is None or suffix is None,
            f'the argument {name} has multiple modifiers' )

    if prefix is not None: return read(text[1:]), prefix
    if suffix is not None: return read(text[:-1]), suffix
    return read(text), None

def int_from_posi_options(opts: list[str], index: PositionalOption) -> int:

    value, _ = number_from_posi_options(opts, index, int, "")
    return value

def float_from_posi_options (
        
    opts: list[str],
    index: PositionalOption

) -> float:

    value, modifier = number_from_posi_options(opts, index, float, "wh%")

    divisors = [ PositionalOption.main_divisor ,
                 PositionalOption.soft_divisor ,
                 PositionalOption.hard_divisor ]

    multipliers = [ PositionalOption.main_multiplier ]

    if modifier is not None:

        if modifier == 'w' and index in divisors:
            value = input_width / value

        elif modifier == "w" and index in multipliers:
            value = value / input_width

        elif modifier == 'h' and index in divisors:
            value = input_height / value

        elif modifier == "h" and index in multipliers:
            value = value / input_height

        elif modifier == "%":
            value = value / 100

        else:
            fail("unexpected real number modifier")

    return value

def settings_from_posi_options(opts: list[str]) -> Settings:

    return Settings (

        MainSettings(
            float_from_posi_options(opts, PositionalOption.main_divisor),
            float_from_posi_options(opts, PositionalOption.main_multiplier)
        ),

        ModelSettings(
            str_from_posi_options(opts, PositionalOption.soft_model),
            float_from_posi_options(opts, PositionalOption.soft_divisor),
            int_from_posi_options(opts, PositionalOption.soft_multiplier),
            int_from_posi_options(opts, PositionalOption.soft_iterations)
        ),

        ModelSettings(
            str_from_posi_options(opts, PositionalOption.hard_model),
            float_from_posi_options(opts, PositionalOption.hard_divisor),
            int_from_posi_options(opts, PositionalOption.hard_multiplier),
            int_from_posi_options(opts, PositionalOption.hard_iterations)
        )
    )

########################################################################################
# Options -> Settings
########################################################################################

if len(sys.argv) == len(Argument) + len(PathOption):
    settings = settings_from_path_options(sys.argv[len(Argument):])
elif len(sys.argv) == len(Argument) + len(PositionalOption):
    settings = settings_from_posi_options(sys.argv[len(Argument):])
else:
    fail( "incorrect parameter count, "
          f"{len(Argument) + len(PathOption)} or "
          f"{len(Argument) + len(PositionalOption)} expected" )

log("options have been loaded")

########################################################################################
# Disjoint Settings Validation
########################################################################################

assume( settings.main.multiplier >= 1.0 , "main multiplier < 1"       )
assume( settings.main.divisor    >= 1.0 , "main divisor < 1"          )
assume( settings.soft.multiplier >= 1   , "soft-phase multiplier < 1" )
assume( settings.soft.divisor    >= 1.0 , "soft-phase divisor < 1"    )
assume( settings.soft.iterations >= 0   , "soft-phase iterations < 0" )
assume( settings.hard.multiplier >= 1   , "hard-phase multiplier < 1" )
assume( settings.hard.divisor    >= 1.0 , "hard-phase divisor < 1"    )
assume( settings.hard.iterations >= 0   , "hard-phase iterations < 0" )

assume( (model_folder/(settings.soft.model + ".bin")).is_file()     ,
        "missing soft-phase model weights (.bin)"                           )
assume( (model_folder / (settings.soft.model + ".param")).is_file() ,
        "missing soft-phase model parameters (.param)"                      )

assume( (model_folder / (settings.hard.model + ".bin")).is_file()   ,
        "missing hard-phase model weights (.bin)"                           )
assume( (model_folder / (settings.hard.model + ".param")).is_file() ,
        "missing hard-phase model parameters (.param)"                      )

log("settings have passed disjoint validation")

########################################################################################
# Shorthands
########################################################################################

input_min_length = min(input_width, input_height)
input_max_length = max(input_width, input_height)
input_mpx        = input_width * input_height / float(1000000)

main_factor  = settings.main.multiplier / settings.main.divisor
soft_factor  = settings.soft.multiplier / settings.soft.divisor
hard_factor  = settings.hard.multiplier / settings.hard.divisor

total_main_multiplier = settings.main.multiplier * settings.main.divisor
total_factor = max( settings.main.multiplier * max(soft_factor, hard_factor) ,
                    1 if total_main_multiplier >= settings.soft.multiplier
                      else settings.soft.multiplier / settings.main.divisor  )

final_width      = int(input_width  * settings.main.multiplier)
final_height     = int(input_height * settings.main.multiplier)
final_min_length = min(final_width, final_height)
final_max_length = max(final_width, final_height)

base_main_width  = int(input_width  / settings.main.divisor)
base_main_height = int(input_height / settings.main.divisor)
base_soft_width  = int(final_width  / settings.soft.divisor)
base_soft_height = int(final_height / settings.soft.divisor)
base_hard_width  = int(final_width  / settings.hard.divisor)
base_hard_height = int(final_height / settings.hard.divisor)

########################################################################################
# Combined Settings Validation
########################################################################################

assume ( settings.main.multiplier >= settings.main.divisor ,
         "main-phase divisor exceeds multiplier"           )

assume ( settings.soft.multiplier >= settings.soft.divisor ,
         "soft-phase divisor exceeds multiplier"           )

assume ( settings.hard.multiplier >= settings.hard.divisor ,
         "hard-phase divisor exceeds multiplier"            )

assume (input_min_length >= settings.main.divisor and
        final_min_length >= settings.soft.divisor and
        final_min_length >= settings.hard.divisor,
         "attempt to generate an empty intermediate image")

assume( input_mpx * total_factor ** 2 < MAX_MPX                                ,
        f"attempt to generate an intermediate image larger than {MAX_MPX} Mpx" )

log("settings have passed combined validation")

########################################################################################
# Session Construction
########################################################################################

session = Session (

    Invocation(invocation_date + "_" + invocation_time, SOFTWARE_VERSION) ,
    ImageInfo(input_format, input_mode, input_width, input_height)        ,
    ImageInfo(OUTPUT_FORMAT, OUTPUT_MODE, final_width, final_height)      ,
    settings
)

########################################################################################
# Internal Output Naming System
########################################################################################

naming_state = SimpleNamespace(i = 0, j = 0, k = 0)

def step_forward() -> Path:

    phase, steps = phases[naming_state.i]
    step         = steps[naming_state.j]

    file_name = "_".join(( f"{(naming_state.k + 1):02}"        ,
                           f"{phase}-phase"                    ,
                           f"{step}-step"                      ,
                           f"{state.image.width}x{state.image.height}.png" ))

    file = session_folder / file_name

    state.image.save(file)

    naming_state.k += 1
    naming_state.j = (naming_state.j + 1) % len(steps)

    return file

def phase_forward() -> None:

    naming_state.i += 1
    naming_state.j = 0

########################################################################################
# RealESRGAN Runner
########################################################################################

def run_realesrgan (

    model_name   : str         ,
    tiling_level : int = 2     , # 128px
    gpu          : bool = True ,

):

    state.image.save(temp_input_file)

    subprocess.run(

        [ str(renv_file)                                                        ,
          "-i", str(temp_input_file)                                            ,
          "-o", str(temp_output_file)                                           ,
          "-m", str(model_folder)                                               ,
          "-n", model_name                                                      ,
          "-t", "0" if tiling_level == 0 else str(64 * 2 ** (tiling_level - 1)) ,
          "-g", "0" if gpu else "-1"                                            ,
          "-j", "1:1:1"                                                         ],

        check = True
    )

    with Image.open(temp_output_file) as result:
        state.image = result.copy()

########################################################################################
# Image Processing
########################################################################################

def scale(factor: float):

    size = int(state.image.width * factor), int(state.image.height * factor)

    state.image = state.image.resize(size, Image.Resampling.BICUBIC)

def resize(width: int, height: int):

    state.image = state.image.resize((width, height), Image.Resampling.BICUBIC)

########################################################################################
# Session Recording
########################################################################################

with open(session_file, "w") as session_handle:
    session_handle.write(export_session(session))

log("session file has been written")

########################################################################################
# Configuration Recording
########################################################################################

with open(settings_file, "w") as settings_handle:
    settings_handle.write(export_settings(settings))

log("settings file has been written")

########################################################################################
# Input Phase
########################################################################################

step_forward()

resize(base_main_width, base_main_height)
step_forward()

phase_forward()

log("the input phase has been completed")

########################################################################################
# Main Phase
########################################################################################

main_iterations = math.ceil(math.log(total_main_multiplier, settings.soft.multiplier))

for _ in range(main_iterations - 1):

    run_realesrgan(settings.soft.model)
    step_forward()

    scale( (total_main_multiplier / settings.soft.multiplier ** main_iterations) **
           (1 / (main_iterations - 1))                                            )
    step_forward()

run_realesrgan(settings.soft.model)
step_forward()

phase_forward()

log("the main phase has been completed")

########################################################################################
# Soft Phase
########################################################################################

for _ in range(settings.soft.iterations):

    resize(base_soft_width, base_soft_height)
    step_forward()

    run_realesrgan(settings.soft.model)
    step_forward()

phase_forward()

log("the soft phase has been completed")

########################################################################################
# Hard Phase
########################################################################################

for _ in range(settings.hard.iterations):

    resize(base_hard_width, base_hard_height)
    step_forward()

    run_realesrgan(settings.hard.model)
    step_forward()

phase_forward()

log("the hard phase has been completed")

########################################################################################
# Output Phase
########################################################################################

resize(final_width, final_height)
step_forward()

phase_forward()

log("the output phase has been completed")

########################################################################################
# Output Saving
########################################################################################

state.image.save(output_file)

log("the output image has been saved")

########################################################################################
# End
########################################################################################