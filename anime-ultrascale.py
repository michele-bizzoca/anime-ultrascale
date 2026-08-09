########################################################################################
# Imports
########################################################################################

# Qualified
import types
import sys
import os
import time
import atexit
import re
import math
import json
import dacite
import pyvips
import textwrap
import psutil
import dataclasses
import subprocess

# Unqualified
from pathlib import Path
from enum import IntEnum
from datetime import datetime
from dataclasses import dataclass
from threading import Event, Thread, Lock
from typing import (cast, Callable, TypeVar, NoReturn,
                    TypeAlias, Final, TextIO, BinaryIO, Any)

########################################################################################
# Type Variables
########################################################################################

T = TypeVar("T")

########################################################################################
# Constants
########################################################################################

SOFTWARE_VERSION : Final = "1.0"
OUTPUT_FORMAT    : Final = "PNG"
OUTPUT_MODE      : Final = "RGBA"
TEMP_FOLDER      : Final = "temp"
MODEL_FOLDER     : Final = "models"
SESSION_FOLDER   : Final = "sessions"
RENV_FOLDER      : Final = "renv"
SESSION_FILE     : Final = "session.json"
SETTINGS_FILE    : Final = "settings.cfg"
LOG_FILE         : Final = "log.txt"
TEMP_INPUT_FILE  : Final = "in.png"
TEMP_OUTPUT_FILE : Final = "output.png"
RENV_FILE        : Final = "realesrgan-ncnn-vulkan"
INVOCATION_FILE  : Final = "invocation.txt"
NCNN_FILE        : Final = "ncnn.txt"
PROGRESS_FILE    : Final = "progress.txt"
MAX_MPX          : Final = 200
MAX_TILING       : Final = 16
MAX_ITERATIONS   : Final = 16

########################################################################################
# Phases
########################################################################################

PHASES : Final = [

    ( "input"     , ["export"      , "downscaling" ]) ,
    ( "main"      , ["upscaling"   , "downscaling" ]) ,
    ( "soft"      , ["downscaling" , "upscaling"   ]) ,
    ( "hard"      , ["downscaling" , "upscaling"   ]) ,
    ( "output"    , ["downscaling" , "export"      ])
]

########################################################################################
# Scalers
########################################################################################

SCALERS : Final = ['bicubic', 'lanczos']

########################################################################################
# Invocation Data
########################################################################################

INVOCATION_INSTANT : Final = datetime.fromtimestamp(psutil.Process().create_time())
INVOCATION_STAMP   : Final = INVOCATION_INSTANT.strftime('%Y/m/%d--%H-%M-%S--%f')
INVOCATION_DATE    : Final = INVOCATION_INSTANT.strftime('%Y-%m-%d')
INVOCATION_TIME    : Final = INVOCATION_INSTANT.strftime('%H-%M-%S')
INVOCATION_USEC    : Final = INVOCATION_INSTANT.strftime('%f')
INVOCATION_PID     : Final = os.getpid()

########################################################################################
# Paths
########################################################################################

PARENT_PATH       : Final = Path(__file__).resolve().parent
MODEL_FOLDER_PATH : Final = PARENT_PATH / MODEL_FOLDER
RENV_FOLDER_PATH  : Final = PARENT_PATH / RENV_FOLDER


SESSION_FOLDER_PATH : Final = ( PARENT_PATH           /
                                SESSION_FOLDER        /
                                INVOCATION_DATE       /
                                f"{INVOCATION_TIME}-"  
                                f"{INVOCATION_PID}"   )

TEMP_FOLDER_PATH : Final = ( PARENT_PATH           /
                             TEMP_FOLDER           /
                             f"{INVOCATION_DATE}-"      
                             f"{INVOCATION_TIME}-"      
                             f"{INVOCATION_PID}"   )

INVOCATION_FILE_PATH  : Final = SESSION_FOLDER_PATH / INVOCATION_FILE
LOG_FILE_PATH         : Final = SESSION_FOLDER_PATH / LOG_FILE
SESSION_FILE_PATH     : Final = SESSION_FOLDER_PATH / SESSION_FILE
SETTINGS_FILE_PATH    : Final = SESSION_FOLDER_PATH / SETTINGS_FILE
NCNN_FILE_PATH        : Final = SESSION_FOLDER_PATH / NCNN_FILE
PROGRESS_FILE_PATH    : Final = SESSION_FOLDER_PATH / PROGRESS_FILE
TEMP_INPUT_FILE_PATH  : Final = TEMP_FOLDER_PATH    / TEMP_INPUT_FILE
TEMP_OUTPUT_FILE_PATH : Final = TEMP_FOLDER_PATH    / TEMP_OUTPUT_FILE
RENV_FILE_PATH        : Final = RENV_FOLDER_PATH    / RENV_FILE

########################################################################################
# Save Levels
########################################################################################

class SaveLevel(IntEnum):

    nothing   = 0
    text      = 1
    error     = 2
    endpoints = 3
    research  = 4
    debug     = 5

def descriptor(level: SaveLevel):
    if level == SaveLevel.error:
        return "ERROR"
    if level == SaveLevel.debug:
        return "DEBUG"
    else:
        return "NORMAL"


########################################################################################
# Arguments
########################################################################################

class Argument(IntEnum):

    script_filepath   = 0
    input_filepath    = 1
    output_filepath   = 2

class PathOption(IntEnum):
    settings_filepath = 0

class CoreOption(IntEnum):
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
    main_scaler       = 10
    final_scaler      = 11

@dataclass
class FlexOption:
    name      : str
    letter    : str
    values    : list[str]
    default   : str

FLEX_OPTIONS : Final = [

    FlexOption ("save", "s", [level.name for level in SaveLevel], "endpoints"),
    FlexOption ("tile", "t", [str(i) for i in range(MAX_TILING)], "0")
]

########################################################################################
# Session
########################################################################################

@dataclass
class Invocation:

    time    : str
    version : str
    save    : str

@dataclass
class ImageInfo:

    format : str
    mode   : str
    width  : int
    height : int

@dataclass
class MainSettings:

    divisor      : float
    multiplier   : float

@dataclass
class ModelSettings:

    model      : str
    divisor    : float
    multiplier : int
    iterations : int

@dataclass
class ScalerSettings:

    main  : str
    final : str

@dataclass
class Settings:

    main    : MainSettings
    soft    : ModelSettings
    hard    : ModelSettings
    scaling : ScalerSettings

@dataclass
class Session:

    invocation : Invocation
    input      : ImageInfo
    output     : ImageInfo
    settings   : Settings

########################################################################################
# Work Units
########################################################################################

@dataclass
class Scaling:
    algorithm  : str
    in_width   : int
    in_height  : int
    out_width  : int
    out_height : int

@dataclass
class ScalingAI:
    model      : str
    multiplier : int
    in_width   : int
    in_height  : int
    out_width  : int
    out_height : int

@dataclass
class PureSAI:
    model      : str
    multiplier : int
    in_width   : int
    in_height  : int
    out_width  : int
    out_height : int

@dataclass
class StepForward:
    save   : bool
    width  : int
    height : int

@dataclass
class PhaseForward:
    pass

Unit: TypeAlias = Scaling | ScalingAI | PureSAI | StepForward | PhaseForward

UNIT_CLASSES : Final = [Scaling, ScalingAI, PureSAI, StepForward, PhaseForward]

def unit_cost(unit: Unit) -> float:

    if isinstance(unit, ScalingAI):
        return unit.in_width * unit.in_height * 1.33 / 1000000
    elif isinstance(unit, Scaling) and unit.in_width != unit.out_width:
        return unit.out_width * unit.out_height * 0.1 / 1000000
    elif isinstance(unit, StepForward):
        return unit.save * unit.width * unit.height * 0.33 / 1000000
    elif isinstance(unit, PureSAI):
        return unit.in_width * unit.in_height * 1.00 / 1000000
    return 0

########################################################################################
# Current Time
########################################################################################

def now() -> str: return datetime.now().strftime('on %Y/%m/%d at %H:%M:%S and %f')

########################################################################################
# Early Error Reporting
########################################################################################

def early_fail( message   : str                              ,
                suggest   : bool = True                      ,
                exception : type[BaseException] = SystemExit ) -> NoReturn:

    tail = " Run with --help for usage information." if suggest else ""
    raise exception(message[:1].upper() + message[1:] + "." + tail)

def early_assume( condition : bool                             ,
                  message   : str                              ,
                  suggest   : bool = True                      ,
                  exception : type[BaseException] = SystemExit ) -> None:

    if not condition:
        early_fail(message, suggest, exception)

########################################################################################
# Help and Version
########################################################################################

def help_and_version() -> None:

    if len(sys.argv) == 2:
        if sys.argv[1] in ["-h", "--help"]:
            print(HELP, end = "")
            exit()
        elif sys.argv[1] in ["-v", "--version"]:
            print(SOFTWARE_VERSION)
            exit()

########################################################################################
# Options Sorting
########################################################################################

fixed_arguments    : list[str]      = []
positional_options : list[str]      = []
flex_options       : dict[str, str] = {}

def sort_options() -> None:

    global fixed_arguments
    global positional_options
    global flex_options

    fixed_arguments    = sys.argv[:len(Argument)]
    positional_options = []
    flex_options       = {}

    flex_option_by_name   = {}
    flex_option_by_letter = {}

    for option in FLEX_OPTIONS:
        flex_option_by_name[option.name]     = option
        flex_option_by_letter[option.letter] = option

    early_assume(len(sys.argv) >= len(Argument), "incomplete I/O specification")

    i = len(Argument)
    while i < len(sys.argv):

        arg = sys.argv[i]

        if arg.startswith("--"):
            early_assume( arg[2:] in flex_option_by_name   ,
                          f"unknown floating option {arg}" )
            option = flex_option_by_name[arg[2:]]

        elif arg.startswith("-"):
            early_assume( arg[1:] in flex_option_by_letter ,
                          f"unknown floating option {arg}" )
            option = flex_option_by_letter[arg[1:]]

        else:
            early_assume( not flex_options                             ,
                          "positional option following a floating one" )
            positional_options.append(arg)
            i += 1; continue

        early_assume( i + 1 < len(sys.argv)                      ,
                      f"missing value for floating option {arg}" )

        early_assume( not sys.argv[i + 1].startswith("-")        ,
                      f"missing value for floating option {arg}" )

        early_assume( option.name not in flex_options              ,
                      f"multiple values for floating option {arg}" )

        early_assume( sys.argv[i + 1] in option.values           ,
                      f"unknown value for floating option {arg}" )

        flex_options[option.name] = sys.argv[i + 1]
        i += 2; continue

    for option in FLEX_OPTIONS:
        flex_options.setdefault(option.name, option.default)

########################################################################################
# Save Level
########################################################################################

def save_level() -> SaveLevel:

    return SaveLevel[flex_options["save"]]

########################################################################################
# Session Folder Creation
########################################################################################

def create_session_folder() -> None:

    if save_level() >= SaveLevel.text:
        SESSION_FOLDER_PATH.mkdir(parents = True, exist_ok = True)

########################################################################################
# Invocation File Creation
########################################################################################

def create_invocation_file() -> None:

    if save_level() >= SaveLevel.text:
        with open(INVOCATION_FILE_PATH, "w") as INVOCATION_FILE_HANDLE:
            timestamp = f"{INVOCATION_DATE}-{INVOCATION_TIME}-{INVOCATION_USEC}"
            INVOCATION_FILE_HANDLE.write( f"PID: {INVOCATION_PID}\n"         +
                                          f"Timestamp: {timestamp}\n"        +
                                          f"PWD: {Path.cwd()}\n"             +
                                          f"Command: {' '.join(sys.argv)}\n" )

########################################################################################
# Logging
########################################################################################

log_file_handle: TextIO

def create_log_file() -> None:

    global log_file_handle

    if save_level() >= SaveLevel.text:
        log_file_handle = open(LOG_FILE_PATH, "w")
        def close_log_file(): log_file_handle.close()
        atexit.register(close_log_file)
        log("the main log system is operative")

def log(message: str, level: SaveLevel = SaveLevel.text):

    if save_level() >= SaveLevel.text and save_level() >= level:
        message = f"{now()}, level {descriptor(level)}: {message}"
        log_file_handle.write(message + "\n")
        log_file_handle.flush()

########################################################################################
# Error Reporting
########################################################################################

def fail( message   : str                              ,
          suggest   : bool = True                      ,
          exception : type[BaseException] = SystemExit ) -> NoReturn:

    log(message, SaveLevel.error)
    early_fail(message, suggest, exception)

def assume( condition : bool                             ,
            message   : str                              ,
            suggest   : bool = True                      ,
            exception : type[BaseException] = SystemExit ) -> None:

    if not condition:
        fail(message, suggest, exception)

########################################################################################
# Temporary Files Support
########################################################################################

def clean_temp_folder() -> None:

    for file in TEMP_FOLDER_PATH.iterdir():
        if file.is_file():
            file.unlink()

def remove_temp_folder() -> None:

    TEMP_FOLDER_PATH.rmdir()

def create_temp_folder() -> None:

    TEMP_FOLDER_PATH.mkdir(parents = True, exist_ok = True)

    atexit.register(remove_temp_folder)
    atexit.register(clean_temp_folder)

    log("the temporary file system is operative")

########################################################################################
# User I/O Files
########################################################################################

def input_file_path()  -> Path : return Path(fixed_arguments[Argument.input_filepath])
def output_file_path() -> Path : return Path(fixed_arguments[Argument.output_filepath])

########################################################################################
# I/O Files Existence Checks
########################################################################################

def existence_checks() -> None:

    assume(RENV_FILE_PATH.is_file()                   , "missing Real ESRGAN runner"  )
    assume(input_file_path().is_file()                , "input file does not exist"   )
    assume(output_file_path().parent.is_dir()         , "output folder does not exist")
    assume(output_file_path().suffix.lower() == ".png", "output extension is not png" )

########################################################################################
# Image Loading
########################################################################################

def load(path: Path) -> pyvips.Image:

    source_image = pyvips.Image.new_from_file(str(path), access = "sequential")
    source_image = source_image.colourspace("srgb")

    if source_image.bands == 3:
        source_image = source_image.addalpha()
    elif source_image.bands > 4:
        source_image = source_image[:4]

    source_image = source_image.cast("uchar")

    return source_image.copy_memory()

########################################################################################
# Input Image Loading
########################################################################################

input_image   : pyvips.Image
current_image : pyvips.Image

def current_width()  -> int: return current_image.width
def current_height() -> int: return current_image.height
def input_width()    -> int: return input_image.height
def input_height()   -> int: return input_image.height
def input_format()   -> str: return input_image.get("vips-loader").removesuffix("load")
def input_mode()     -> str: return ( f"{input_image.bands}bands--"                +
                                    f"{input_image.interpretation}"              +
                                    ("+alpha" if input_image.hasalpha() else "") )

def load_input_image() -> None:

    global input_image
    global current_image

    input_image   = load(input_file_path())
    current_image = input_image.copy()

    log( f"the input image has been loaded: {input_image.width}"
         f"x{input_image.height}px {OUTPUT_MODE}" )

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

    return json.dumps(dataclasses.asdict(s), indent = 4, sort_keys = False)

########################################################################################
# Session Import
########################################################################################

def import_session(s : str) -> Session:

    return dacite.from_dict( data_class = Session,
                             data       = json.loads(s),
                             config     = dacite.Config(check_types=True) )

########################################################################################
# Settings Export
########################################################################################

def export_settings(s: Settings) -> str:

    result: str = ""

    for key, value in flatten(dataclasses.asdict(s)).items():
        result += f"{key} = {json.dumps(value)}\n"

    return result

########################################################################################
# Settings Import
########################################################################################

def import_settings(s : str) -> Settings:

    s = re.sub(r'^\s*(#.*)?$\n?', '', s, flags = re.MULTILINE)
    s = re.sub(r'^\s*(\w+)\s*=([^#]*)(#.*)?$', r'"\1": \2,', s, flags=re.MULTILINE)
    s = "{" + s[:-1] + "}"

    return dacite.from_dict( data_class = Settings,
                             data = unflatten(json.loads(s)),
                             config = dacite.Config(check_types=True) )

########################################################################################
# Path Options -> Settings
########################################################################################

def settings_from_path_options() -> Settings:

    path      = Path(positional_options[PathOption.settings_filepath])
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
# Core Options -> Settings
########################################################################################

def str_from_core_options(index: CoreOption) -> str:

    name = index.name.replace("_", " ")
    assume(index < len(positional_options), f'the argument {name} is missing')
    return positional_options[index]

def number_from_core_options (

    index     : CoreOption,
    read      : Callable[[str], T],
    modifiers : str

) -> tuple[T, str | None]:

    text = str_from_core_options(index)
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

def int_from_core_options(index: CoreOption) -> int:

    value, _ = number_from_core_options(index, int, "")
    return value

def float_from_core_options (

    index: CoreOption

) -> float:

    value, modifier = number_from_core_options(index, float, "wh%")

    divisors = [ CoreOption.main_divisor ,
                 CoreOption.soft_divisor ,
                 CoreOption.hard_divisor ]

    multipliers = [ CoreOption.main_multiplier ]

    if modifier is not None:

        if modifier == 'w' and index in divisors:
            value = input_width() / value

        elif modifier == "w" and index in multipliers:
            value = value / input_width()

        elif modifier == 'h' and index in divisors:
            value = input_height() / value

        elif modifier == "h" and index in multipliers:
            value = value / input_height()

        elif modifier == "%":
            value = value / 100

        else:
            fail("unexpected real number modifier")

    return value

def settings_from_core_options() -> Settings:

    return Settings (

        MainSettings(
            float_from_core_options(CoreOption.main_divisor),
            float_from_core_options(CoreOption.main_multiplier)
        ),

        ModelSettings(
            str_from_core_options(CoreOption.soft_model),
            float_from_core_options(CoreOption.soft_divisor),
            int_from_core_options(CoreOption.soft_multiplier),
            int_from_core_options(CoreOption.soft_iterations)
        ),

        ModelSettings(
            str_from_core_options(CoreOption.hard_model),
            float_from_core_options(CoreOption.hard_divisor),
            int_from_core_options(CoreOption.hard_multiplier),
            int_from_core_options(CoreOption.hard_iterations)
        ),

        ScalerSettings(
            str_from_core_options(CoreOption.main_scaler),
            str_from_core_options(CoreOption.final_scaler)
        )
    )

########################################################################################
# Options -> Settings
########################################################################################

settings: Settings

def load_settings() -> None:

    if len(positional_options) == len(PathOption):
        settings = settings_from_path_options()
    elif len(positional_options) == len(CoreOption):
        settings = settings_from_core_options()
    else:
        fail( "incorrect parameter count, "
              f"{len(Argument) - 1 + len(PathOption)} or "
              f"{len(Argument) - 1 + len(CoreOption)} expected" )

    log("the settings have been loaded")

########################################################################################
# Disjoint Settings Validation
########################################################################################

def disjoint_settings_validation() -> None:

    assume( settings.main.multiplier >= 1.0 , "main multiplier < 1"       )
    assume( settings.main.divisor    >= 1.0 , "main divisor < 1"          )
    assume( settings.soft.multiplier >= 2   , "soft-phase multiplier < 2" )
    assume( settings.soft.divisor    >= 1.0 , "soft-phase divisor < 1"    )
    assume( settings.soft.iterations >= 0   , "soft-phase iterations < 0" )
    assume( settings.hard.multiplier >= 2   , "hard-phase multiplier < 2" )
    assume( settings.hard.divisor    >= 1.0 , "hard-phase divisor < 1"    )
    assume( settings.hard.iterations >= 0   , "hard-phase iterations < 0" )


    assume( settings.soft.iterations <= MAX_ITERATIONS ,
            f"soft-phase iterations > {MAX_ITERATIONS}" )

    assume( settings.hard.iterations <= MAX_ITERATIONS ,
            f"hard-phase iterations > {MAX_ITERATIONS}" )

    assume( (MODEL_FOLDER_PATH/(settings.soft.model + ".bin")).is_file()     ,
            "missing soft-phase model weights (.bin)"                        )
    assume( (MODEL_FOLDER_PATH / (settings.soft.model + ".param")).is_file() ,
            "missing soft-phase model parameters (.param)"                   )

    assume( (MODEL_FOLDER_PATH / (settings.hard.model + ".bin")).is_file()   ,
            "missing hard-phase model weights (.bin)"                        )
    assume( (MODEL_FOLDER_PATH / (settings.hard.model + ".param")).is_file() ,
            "missing hard-phase model parameters (.param)"                   )

    assume ( settings.scaling.main in SCALERS ,
            "unknown scaling algorithm"       )

    assume ( settings.scaling.final in SCALERS ,
            "unknown scaling algorithm"        )

    log("the settings have passed disjoint validation")

########################################################################################
# Shorthands
########################################################################################

def input_min_length() : return int(min(input_width(), input_height()))
def input_max_length() : return int(max(input_width(), input_height()))
def input_mpx()        : return input_width() * input_height() / float(1000000)

def main_factor() : return settings.main.multiplier / settings.main.divisor
def soft_factor() : return settings.soft.multiplier / settings.soft.divisor
def hard_factor() : return settings.hard.multiplier / settings.hard.divisor

def max_factor()   : return max(soft_factor(), hard_factor())
def main_scaling() : return settings.main.multiplier * settings.main.divisor
def limit_factor() : return ( 1 if main_scaling() >= settings.soft.multiplier
                                else settings.soft.multiplier / settings.main.divisor )
def total_factor() : return max(settings.main.multiplier * max_factor(), limit_factor())

def output_width()      : return int(input_width() * settings.main.multiplier)
def output_height()     : return int(input_height() * settings.main.multiplier)
def output_min_length() : return min(output_width(), output_height())
def output_max_length() : return max(output_width(), output_height())

def base_main_width()  : return int(input_width()   / settings.main.divisor)
def base_main_height() : return int(input_height()  / settings.main.divisor)
def base_soft_width()  : return int(output_width()  / settings.soft.divisor)
def base_soft_height() : return int(output_height() / settings.soft.divisor)
def base_hard_width()  : return int(output_width()  / settings.hard.divisor)
def base_hard_height() : return int(output_height() / settings.hard.divisor)

########################################################################################
# Combined Settings Validation
########################################################################################

def combined_settings_validation() -> None:

    assume ( settings.soft.multiplier >= settings.soft.divisor ,
             "soft-phase divisor exceeds multiplier"           )

    assume ( settings.hard.multiplier >= settings.hard.divisor ,
             "hard-phase divisor exceeds multiplier"            )

    assume (input_min_length() >= settings.main.divisor and
            output_min_length() >= settings.soft.divisor and
            output_min_length() >= settings.hard.divisor,
             "attempt to generate an empty intermediate image")

    assume( input_mpx() * total_factor() ** 2 < MAX_MPX                            ,
            f"attempt to generate an intermediate image larger than {MAX_MPX} Mpx" )

    log("the settings have passed combined validation")

########################################################################################
# Session Construction
########################################################################################

session: Session

def create_session() -> None:

    global session

    session = Session (

        Invocation(INVOCATION_STAMP, SOFTWARE_VERSION, flex_options["save"])   ,
        ImageInfo(input_format(), input_mode(), input_width(), input_height()) ,
        ImageInfo(OUTPUT_FORMAT, OUTPUT_MODE, output_width(), output_height()) ,
        settings
    )

########################################################################################
# Session Recording
########################################################################################

def record_session() -> None:

    if save_level() >= SaveLevel.text:

        with open(SESSION_FILE_PATH, "w") as session_handle:
            session_handle.write(export_session(session))

        log("the session file has been written")

########################################################################################
# Settings Recording
########################################################################################

def record_settings() -> None:

    if save_level() >= SaveLevel.text:

        with open(SETTINGS_FILE_PATH, "w") as settings_handle:
            settings_handle.write(export_settings(settings))

        log("the settings file has been written")

# ########################################################################################
# # Progress Bar
# ########################################################################################
#
# def clamp(l: float | None, x: float, r:float | None) -> float:
#     if l is not None:
#         x = max(l, x)
#     if r is not None:
#         x = min(r, x)
#     return x
#
#
# def make_bar(percentage: float, width: int = 40) -> str:
#     blocks = " ▏▎▍▌▋▊▉█"
#
#     units = percentage / 100.0 * width
#     full = int(units)
#     fraction = int((units - full) * 8)
#
#     return ( "█" * full                                       +
#              (blocks[fraction] if percentage < 100.0 else "") +
#              " " * max(0, width - full - 1)                   )
#
# class ProgressBar:
#
#     def __init__(self, total_cost: float) -> None:
#         self.total_cost = total_cost
#         self.total_cost_done = 0.0
#         self.unit_class_name = PhaseForward.__name__
#         self.unit_cost = 0.0
#         self.high_speed = False
#         self.unit_cost_done = 0.0
#         self.progress_average_speed = 0.0
#         self.refresh_average_speed = 0.0
#         self.bonus_cost_done = 0.0
#         self.last_progress_instant = perf_counter()
#         self.last_render_instant = self.last_progress_instant
#         self.last_refresh_instant = self.last_progress_instant
#         self.last_progress_percentage = 0.0
#         self.last_render_percentage = 0.0
#         self.lock = Lock()
#         self.stop_event = Event()
#         self.refresh_thread = Thread( target = self._refresh_call ,
#                                       daemon = True               )
#         self.refresh_thread.start()
#         self.timespans = {k.__name__: 0 for k in unit_classes}
#         self.costs = {k.__name__: 0 for k in unit_classes}
#         self.started = {k.__name__: False for k in unit_classes}
#         self._render()
#
#     def new_unit(self, unit: Unit) -> None:
#         with (self.lock):
#             if self.unit_cost > 0.0:
#                 self.progress(100.0)
#             self.progress_average_speed = (
#                 self.costs[type(unit).__name__] / self.timespans[type(unit).__name__]
#                     if self.timespans[type(unit).__name__] != 0
#                     else 0 )
#             self.unit_cost = cost(unit)
#             self.unit_class_name = type(unit).__name__
#             self.unit_cost_done = 0.0
#             self.bonus_cost_done = 0.0
#             self.last_progress_instant = perf_counter()
#             self.last_refresh_instant = perf_counter()
#             self.last_progress_percentage = 0.0
#
#     def progress(self, percentage: float) -> None:
#         delta = percentage - self.last_progress_percentage
#         self.last_progress_percentage = percentage
#         chunk_cost = self.unit_cost * delta / 100.0
#         old_part = clamp(None, self.bonus_cost_done, chunk_cost)
#         new_part = chunk_cost - old_part
#         now = perf_counter()
#         delta = now - self.last_progress_instant
#         self.last_progress_instant = now
#         if self.started[self.unit_class_name] and delta > 0.0:
#             self.timespans[self.unit_class_name] += delta
#             self.costs[self.unit_class_name] += chunk_cost
#             current_speed = chunk_cost / delta
#             self.progress_average_speed = (
#                 current_speed if self.progress_average_speed == 0.0
#                               else ( 0.8 * self.progress_average_speed +
#                                      0.2 * current_speed               ) )
#         self.started[self.unit_class_name] = True
#         self._old_progress(old_part)
#         self._new_progress(new_part)
#         self._render()
#
#     def _old_progress(self, chunk_cost: float) -> None:
#         self.bonus_cost_done -= clamp(0, chunk_cost, None)
#
#     def _new_progress(self, chunk_cost: float) -> None:
#         self.total_cost_done = clamp( None                              ,
#                                       self.total_cost_done + chunk_cost ,
#                                       self.total_cost                   )
#         self.unit_cost_done = clamp( None                              ,
#                                      self.unit_cost_done +  chunk_cost ,
#                                      self.unit_cost                    )
#
#     def refresh(self) -> None:
#         with self.lock:
#             now = perf_counter()
#             delta = now - self.last_refresh_instant
#             self.last_refresh_instant = now
#             bonus_chunk_cost = clamp( 0.0,
#                                       delta * self.progress_average_speed ,
#                                       self.unit_cost - self.unit_cost_done )
#             self.bonus_cost_done += bonus_chunk_cost
#             self.total_cost_done = clamp( None,
#                                           self.total_cost_done + bonus_chunk_cost ,
#                                           self.total_cost                         )
#             self.unit_cost_done = clamp( None                                   ,
#                                          self.unit_cost_done + bonus_chunk_cost ,
#                                          self.unit_cost                         )
#             self._render()
#
#     def _render(self) -> None:
#         now = perf_counter()
#         delta = now - self.last_render_instant
#         self.last_render_instant = now
#         percentage = 100.0 * ( 1.0 if self.total_cost == 0.0
#                                    else self.total_cost_done / self.total_cost )
#         if delta > 0.0:
#             speed = ( (percentage - self.last_render_percentage) *
#                       self.total_cost                            /
#                       (100 * delta)                              )
#             decay = math.exp(-delta / 0.6)
#             self.refresh_average_speed = (
#                 speed if self.refresh_average_speed == 0.0
#                       else ( decay * self.refresh_average_speed +
#                              (1 - decay)* speed                 ) )
#
#         self.last_render_percentage = percentage
#
#         bar = make_bar(percentage)
#
#         line = (
#             f" [{bar}]"
#             f" {percentage:6.2f}% "
#             f"({self.total_cost_done:6.2f}/{self.total_cost:6.2f} Mpx), "
#             f"{self.refresh_average_speed:5.2f} Mpx/s"
#         )
#
#         print( f"\r\033[K" + line ,
#                end=""             ,
#                flush=True         )
#
#         if save_level >= SaveLevel.debug:
#             bar_handle.write(sys.modules[__name__].now() + ": " + line + "\n")
#             bar_handle.flush()
#
#     def close(self) -> None:
#         self.stop_event.set()
#         self.refresh_thread.join()
#         with self.lock:
#             self.total_cost_done = self.total_cost
#             self.last_render_percentage = 100.0
#             self.refresh_average_speed = 0.0
#             self._render()
#
#     def _refresh_call(self) -> None:
#         while not self.stop_event.wait(0.1):
#             self.refresh()
#
#     def __enter__(self) -> "ProgressBar":
#         return self
#
#     def __exit__(self, exc_type, exc_value, traceback) -> None:
#         self.stop_event.set()
#         self.refresh_thread.join()

class ProgressBar:

    def __init__(self, total_cost: float) -> None:
        pass

    def new_unit(self, unit: Unit) -> None:
        pass

    def progress(self, percentage: float) -> None:
        pass

    def refresh(self) -> None:
        pass

    def close(self) -> None:
        pass

    def __enter__(self) -> "ProgressBar":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        pass

########################################################################################
# Image Transformations
########################################################################################

def save(unit: StepForward, path: Path, bar: ProgressBar) -> None:

    if unit.save:

        bar.new_unit(unit)

        image           = current_image.copy()
        last_percentage = 0.0
        started         = False

        def start_bar(image: pyvips.Image, progress: Any) -> None:

            nonlocal started

            if not started:
                started = True
                bar.progress(0.0)

        def update_bar(image: pyvips.Image, progress: Any) -> None:

            nonlocal last_percentage

            percentage = float(progress.percent)
            if percentage > last_percentage:
                last_percentage = percentage
                bar.progress(percentage)

        image.set_progress(True)
        image.signal_connect("preeval", start_bar)
        image.signal_connect("eval", update_bar)

        image.write_to_file(str(path))

        if last_percentage < 100.0:
            bar.progress(100.0)

def scale(unit: Scaling, bar: ProgressBar) -> None:

    global current_image

    kernels         = {"lanczos": "lanczos3", "bicubic": "cubic"}
    kernel          = kernels.get(unit.algorithm)
    factor          = unit.out_width / unit.in_width
    image           = current_image.resize(factor, kernel = kernel)
    last_percentage = 0.0
    started         = False

    def start_bar(image: pyvips.Image, progress: Any) -> None:

        nonlocal started

        if not started:
            started = True
            bar.progress(0.0)

    def update_bar(image: pyvips.Image, progress: Any) -> None:

        nonlocal last_percentage

        percentage = float(progress.percent)
        if percentage > last_percentage:
            last_percentage = percentage
            bar.progress(percentage)

    image.set_progress(True)
    image.signal_connect("preeval", start_bar)
    image.signal_connect("eval", update_bar)

    current_image = image.copy_memory()

    if last_percentage < 100.0:
        bar.progress(100.0)

def scale_ai(unit: ScalingAI, bar: ProgressBar) -> None:

    step_forward_unit = StepForward(True, unit.in_width, unit.in_height)
    pure_sai_unit     = PureSAI(** vars(unit))

    save(step_forward_unit, TEMP_INPUT_FILE_PATH, bar)

    bar.new_unit(pure_sai_unit)

    process = subprocess.Popen(

        [ str(RENV_FILE_PATH)                                ,
          "-i", str(TEMP_INPUT_FILE_PATH)                    ,
          "-o", str(TEMP_OUTPUT_FILE_PATH)                   ,
          "-m", str(MODEL_FOLDER_PATH)                       ,
          "-n", unit.model                                   ,
          "-t", "0" if flex_options["tile"] == "0"
                    else str(64 * int(flex_options["tile"])) ,
          "-g", "0"                                          ,
          "-j", "1:1:1"                                      ,
          "-s", str(unit.multiplier)                         ],

        stdout  = subprocess.PIPE   ,
        stderr  = subprocess.STDOUT ,
        text    = True              ,
        bufsize = 1
    )

    if process.stdout is None:
        fail("failed to capture Real ESRGAN's output")

    for line in process.stdout:
        if save_level() >= SaveLevel.debug:
            ncnn_file_handle.write(now() + ": " + line)
            ncnn_file_handle.flush()
        line = "".join(line.split())
        if re.search(r"^[0-9]+(\.[0-9]+)?%$", line):
            bar.progress(float(line[:-1]))

    return_code = process.wait()
    assume( return_code == 0                              ,
            f"Real ESRGAN failed with code {return_code}" )

    current_image = load(TEMP_OUTPUT_FILE_PATH)

########################################################################################
# Export System
########################################################################################

forward_i: int
forward_j: int
forward_k: int

def init_export_system() -> None:

    global forward_i
    global forward_j
    global forward_k

    forward_i = 0
    forward_j = 0
    forward_k = 0

def first_step() -> bool:
    return forward_i == 0 and forward_j == 0

def run_step_forward(unit: StepForward, bar: ProgressBar):

    global forward_j
    global forward_k

    phase, steps = PHASES[forward_i]
    step         = steps[forward_j]

    if unit.save:

        file_name = "_".join((f"{(forward_k + 1):02}",
                              f"{phase}-phase",
                              f"{step}-step",
                              f"{unit.width}x{unit.height}.png"))
        file_path = SESSION_FOLDER_PATH / file_name
        export(unit, file_path, bar)

        forward_k += 1

    log( f"a{'n' if step[0] in 'aeiou' else ''} {step} "
         f"step in the {phase} phase has been completed "
         f"with output size {unit.width}x{unit.height}" )

    forward_j = (forward_j + 1) % len(steps)

def run_phase_forward(unit : PhaseForward, bar: ProgressBar) -> None:

    global forward_i
    global forward_j

    log(f"the {PHASES[forward_i][0]} phase has been completed")

    forward_i += 1
    forward_j = 0

def run_input_export(unit: InputExport, bar: ProgressBar) -> None:

    global forward_k

    if unit.save:

        file_name = "_".join((f"{(forward_k + 1):02}",
                              f"{PHASES[forward_i][0]}-phase",
                              f"export-step",
                              f"{unit.width}x{unit.height}.png"))
        file_path = SESSION_FOLDER_PATH / file_name
        export(unit, file_path, bar)

        forward_k += 1

def run_output_export(unit: OutputExport, bar: ProgressBar) -> None:

    global forward_k

    if unit.save:

        file_name = "_".join((f"{(forward_k + 1):02}",
                              f"{PHASES[forward_i][0]}-phase",
                              f"export-step",
                              f"{unit.width}x{unit.height}.png"))
        file_path = SESSION_FOLDER_PATH / file_name
        export(unit, file_path, bar)

        forward_k += 1

# def run_step_forward(bar: ProgressBar):
#
#     global forward_j
#     global forward_k
#
#     phase, steps = PHASES[forward_i]
#     step         = steps[forward_j]
#
#     if ( save_level() >= SaveLevel.endpoints and (first_step() or last_step()) or
#          save_level() >= SaveLevel.research                                     ):
#
#         file_name = "_".join((f"{(forward_k + 1):02}",
#                               f"{phase}-phase",
#                               f"{step}-step",
#                               f"{current_width()}x{current_height()}.png"))
#         file_path = SESSION_FOLDER_PATH / file_name
#         current_image.write_to_file(str(file_path))
#
#         forward_k += 1
#
#     log( f"a{'n' if step[0] in 'aeiou' else ''} {step} step in the {phase}"
#          f" phase has been completed with output size {current_width()}x"
#          f"{current_height()}" )
#
#     if last_step():
#         current_image.write_to_file(str(output_file_path()))
#         log(f"the output image has been saved, {current_width()}x"
#             f"{current_height()}px {OUTPUT_FORMAT} {OUTPUT_MODE}")
#
#     forward_j = (forward_j + 1) % len(steps)

########################################################################################
# NCNN Logging
########################################################################################

ncnn_file_handle: TextIO

def create_ncnn_file() -> None:

    global ncnn_file_handle

    if save_level() >= SaveLevel.debug:
        ncnn_file_handle = open(NCNN_FILE_PATH, "w")
        def close_ncnn_file(): ncnn_file_handle.close()
        atexit.register(close_ncnn_file)
        log("the ncnn logging system is operative")

########################################################################################
# Runners
########################################################################################


def run_algorithm(
    algorithm: str,
    size: tuple[int, int],
    bar: ProgressBar,
) -> None:

    if state.image.size == size:
        bar.progress(100.0)
        return

    input_data = state.image.tobytes()

    source = pyvips.Image.new_from_memory(
        input_data,
        state.image.width,
        state.image.height,
        len(OUTPUT_MODE),
        "uchar",
    )

    result = source.resize(
        size[0] / source.width,
        vscale=size[1] / source.height,
        kernel=algorithm,
    )

    result.set_progress(True)

    started = False
    last_percentage = 0

    def on_start(
            _image: pyvips.Image,
            _progress: pyvips.Progress,
    ) -> None:
        nonlocal started

        if not started:
            started = True
            bar.progress(0.0)

    def on_progress(
            _image: pyvips.Image,
            progress: pyvips.Progress,
    ) -> None:
        nonlocal last_percentage

        percentage = int(progress.percent)

        if not started or percentage <= last_percentage:
            return

        last_percentage = percentage
        bar.progress(float(percentage))

    result.set_progress(True)
    result.signal_connect("preeval", on_start)
    result.signal_connect("eval", on_progress)

    output_data = result.write_to_memory()

    new_image = Image.frombytes(
        OUTPUT_MODE,
        size,
        output_data,
        "raw",
        OUTPUT_MODE,
        0,
        1,
    )

    state.image.close()
    state.image = new_image

    if last_percentage < 100.0:
        bar.progress(100.0)

# ########################################################################################
# # Scaling Algorithms
# ########################################################################################
#
# scalers = {"bicubic": "cubic", "lanczos": "lanczos3"}
#
# main_scaler  = scalers[settings.scaling.main]
# final_scaler = scalers[settings.scaling.final]
#
# ########################################################################################
# # Progress Bar Logging
# ########################################################################################
#
# if save_level >= SaveLevel.debug:
#     bar_handle = open(bar_file, "w")
#
# if save_level >= SaveLevel.debug:
#     def close_bar_file(): bar_handle.close()
#     atexit.register(close_bar_file)
#
# log("the progress bar logging system is operative")
#
# ########################################################################################
# # Planners
# ########################################################################################
#
# def current_size() -> tuple[int, int]:
#
#     for i in range(len(state.units) - 1, -1, -1):
#
#         unit = state.units[i]
#
#         if isinstance(unit, (Scaling, ScalingAI)):
#             return unit.out_width, unit.out_height
#         else:
#             continue
#
#     return input_width, input_height
#
# def plan_algorithm( arg: float |tuple[int, int]               ,
#                     algorithm: str = main_scaler ) -> None:
#
#     in_width  , in_height  = current_size()
#     out_width , out_height = ( arg if isinstance(arg, tuple)
#                                    else [int(in_width  * arg),
#                                          int(in_height * arg)] )
#     state.units.append(Scaling( algorithm  ,
#                                 in_width   ,
#                                 in_height  ,
#                                 out_width  ,
#                                 out_height ))
#
# def plan_realesrgan(settings: ModelSettings = settings.soft) -> None:
#
#     in_width  , in_height  = current_size()
#     out_width , out_height = ( int(in_width  * settings.multiplier) ,
#                                int(in_height * settings.multiplier) )
#     state.units.append(ScalingAI( settings.model ,
#                                   in_width       ,
#                                   in_height      ,
#                                   out_width      ,
#                                   out_height     ))
#
# def plan_phase_forward() -> None:
#     state.units.append(PhaseForward())
#
# def plan_step_forward(first: bool = False, last:bool = False) -> None:
#     save = last + ( (first or last) and save_level >= SaveLevel.endpoints or
#                     save_level >= SaveLevel.research                       )
#     width, height  = current_size()
#     state.units.append(StepForward(save, width, height))
#
# ########################################################################################
# # Planning - Input Phase
# ########################################################################################
#
# plan_step_forward(True, False)
#
# plan_algorithm((base_main_width, base_main_height))
# plan_step_forward()
#
# plan_phase_forward()
#
# ########################################################################################
# # Planning - Main Phase
# ########################################################################################
#
# main_iterations = math.ceil(math.log(total_main_multiplier, settings.soft.multiplier))
# factor = ( (total_main_multiplier / settings.soft.multiplier ** main_iterations) **
#            (1 / (main_iterations - 1) if main_iterations != 1 else 0)             )
#
# for _ in range(main_iterations - 1):
#     plan_realesrgan()
#     plan_step_forward()
#
#     plan_algorithm(factor)
#     plan_step_forward()
#
# if main_iterations != 0:
#     plan_realesrgan()
#     plan_step_forward()
#
# plan_phase_forward()
#
# ########################################################################################
# # Planning - Soft Phase
# ########################################################################################
#
# for _ in range(settings.soft.iterations):
#
#     plan_algorithm((base_soft_width, base_soft_height))
#     plan_step_forward()
#
#     plan_realesrgan()
#     plan_step_forward()
#
# plan_phase_forward()
#
# ########################################################################################
# # Planning - Hard Phase
# ########################################################################################
#
# for _ in range(settings.hard.iterations):
#
#     plan_algorithm((base_hard_width, base_hard_height))
#     plan_step_forward()
#
#     plan_realesrgan(settings.hard)
#     plan_step_forward()
#
# plan_phase_forward()
#
# ########################################################################################
# # Planning - Output Phase
# ########################################################################################
#
# plan_algorithm((output_width, output_height), final_scaler)
# plan_step_forward()
#
# plan_step_forward(False, True)
#
# plan_phase_forward()
#
# ########################################################################################
# # Planning Processing
# ########################################################################################
#
# total_cost = sum([cost(unit) for unit in state.units])
#
# log("the execution plan has been created")
#
# ########################################################################################
# # Newline at Exit
# ########################################################################################
#
# def newline() -> None: print()
# atexit.register(newline)
#
# ########################################################################################
# # Plan Execution
# ########################################################################################
#
# try:
#     with ProgressBar(total_cost) as bar:
#         for unit in state.units:
#             bar.new_unit(unit)
#             if isinstance(unit, Scaling):
#                 run_algorithm( unit.algorithm                         ,
#                                (unit.out_width, unit.out_height), bar )
#             elif isinstance(unit, ScalingAI):
#                 run_realesrgan( unit.model                           ,
#                                 unit.out_width // unit.in_width, bar )
#             elif isinstance(unit, StepForward):
#                 run_step_forward(bar)
#             elif isinstance(unit, PhaseForward):
#                 run_phase_forward(bar)
#         bar.close()
# except KeyboardInterrupt:
#     print()
#     fail("interrupted by user", False)
#
# ########################################################################################
# # Main
# ########################################################################################
#
# def main():
#     pass
#
# ########################################################################################
# # Help
# ########################################################################################
#
# HELP = textwrap.dedent("""\
#     Anime-Ultrascale
#     A Tool for Extreme Anime Upscaling.
#
#     USAGE
#
#     anime-ultrascale INPUT OUTPUT
#         MAIN_DIVISOR MAIN_MULTIPLIER
#         SOFT_MODEL SOFT_DIVISOR SOFT_MULTIPLIER SOFT_ITERATIONS
#         HARD_MODEL HARD_DIVISOR HARD_MULTIPLIER HARD_ITERATIONS
#         MAIN_SCALER FINAL_SCALER
#         [OPTIONS]
#
#     anime-ultrascale INPUT OUTPUT.png
#                      {SESSION.json│SETTINGS.cfg}
#                      [OPTIONS]
#
#     anime-ultrascale {-h│--help│-v│--version}
#
#     POSITIONAL ARGUMENTS
#
#     INPUT (str)
#         Input image in any of the following formats: PNG, JPG/JPEG,
#         BMP, TIF/TIFF.
#
#     OUTPUT.png (str)
#         Output image in any of the following formats: PNG (RGBA),
#         JPG/JPEG (RGB), BMP (RGB), TIF/TIFF (RGBA).
#
#     MAIN_DIVISOR (float)
#         Main-phase downscaling divisor. Use it to revert an upscaling
#         already present in the input image.
#
#     MAIN_MULTIPLIER (float)
#         Main-phase upscaling multiplier. It determines the output
#         width and height, namely output = (input / MAIN_DIVISOR) *
#         MAIN_MULTIPLIER.
#
#     SOFT_MODEL (str)
#         Real ESRGAN model specialized in preserving detail (basename
#         only).
#
#     SOFT_DIVISOR (float)
#         Soft-phase downscaling divisor. Use it to lose detail.
#
#     SOFT_MULTIPLIER (int)
#         Soft-phase upscaling multiplier. Use it to restore detail. It
#         has to be supported by the model SOFT_MODEL.
#
#     SOFT_ITERATIONS (int)
#         Soft-phase iterations.
#
#     HARD_MODEL (str)
#         Real ESRGAN model specialized in adding detail (basename only).
#
#     HARD_DIVISOR (float)
#         Hard-phase downscaling divisor. Use it to lose detail.
#
#     HARD_MULTIPLIER (int)
#         Hard-phase upscaling multiplier. Use it to enhance detail. It
#         has to be supported by the model SOFT_MODEL.
#
#     HARD_ITERATIONS (int)
#         Hard-phase iterations.
#
#     MAIN_SCALER (str)
#         The downscaling algorithm used in intermediate steps.
#
#     FINAL_SCALER (str)
#         The downscaling algorithm used in the final step.
#
#     {SESSION.json│SETTINGS.cfg} (str)
#         A session or settings file to import settings from.
#
#     {-h│--help}
#         Shows this help message.
#
#     {-v│--version}
#         Shows this program's version.
#
#     CONSTRAINTS
#
#         MAIN_DIVISOR    >= 1
#         MAIN_MULTIPLIER >= 1
#
#         SOFT_MULTIPLIER >= 2
#         SOFT_DIVISOR    >= 1
#         SOFT_DIVISOR    <= SOFT_MULTIPLIER
#         SOFT_ITERATIONS >= 0
#
#         HARD_MULTIPLIER >= 2
#         HARD_DIVISOR    >= 1
#         HARD_DIVISOR    <= HARD_MULTIPLIER
#         HARD_ITERATIONS >= 0
#
#         MAIN_SCALER     in ['bicubic', 'lanczos']
#         FINAL_SCALER    in ['bicubic', 'lanczos']
#
#     OPTIONS
#
#     {-t│--tile} (int)
#         The width/height of the square used to tile the total area.
#             0            -> automatic selection
#             1 <= n <= 16 -> n * 64px
#
#     {-s│--save} (str)
#         Determines the session data that is saved.
#             nothing   -> nothing
#             text      -> basic textual data
#             endpoints -> as 'text' + input/output images
#             research  -> as 'endpoints' + intermediate images
#             debug     -> as 'research' + debug textual data
#
#     DESCRIPTION
#
#     Anime-Ultrascale performs extreme image enlargement by controlled
#     alternation of downscaling and AI upscaling, where downscaling is
#     performed by the bicubic and lanczos algorithms, and AI upscaling
#     is performed using Real ESRGAN models.
#
#     Knowledge of the three phases (main, soft, hard) is required for
#     correct usage. These are described in a dedicated article, see
#     the repositories section.
#
#     DIRECTORY TREE
#
#     Real ESRGAN models, following the usual .bin/.param convention,
#     have to be stored in the 'models' folder.
#
#     The official Real ESRGAN executable, 'realesrgan-ncnn-vulkan',
#     has to be stored in the 'renv' folder.
#
#     Useful additional data, determined by the 'save' option, will
#     be stored in the 'sessions' folder.
#
#     Temporary data will be stored in the 'temp' folder.
#
#     If this program has been installed from the official repository
#     using 'install.sh', the repository's folder will contain:
#
#     ┌── LICENSE
#     ├── README
#     ├── anime-ultrascale.py
#     ├── anime-ultrascale
#     ├── pyproject.toml
#     ├── install
#     ├── .bin
#     │     └── anime-ultrascale
#     ├── .venv
#     │     └── ·······
#     ├── renv
#     │     └── realesrgan-ncnn-vulkan
#     ├── models
#     │     ├── <model>.bin
#     │     ├── <model>.param
#     │     └── ·······
#     ├── sessions
#     │     ├── <date>
#     │     │      ├── <time+pid>
#     │     │      │        └── ·······
#     │     │      └── ·······
#     │     └── ·······
#     └── temp
#           ├── <date+time+pid>
#           │      └── ·······
#           └── ·······
#
#     REPOSITORIES
#
#     Concept -> https://github.com/michele-bizzoca/anime-upscaling
#     Program -> https://github.com/michele-bizzoca/anime-ultrascale
#
#     LICENSE
#
#     Copyright (c) 2026 Michele Bizzoca
#     Licensed under the MIT License.
# """)
#
# ########################################################################################
# # Main Call
# ########################################################################################
#
# if __name__ == "__main__":
#     main()
#
# ########################################################################################
# # End
# ########################################################################################
