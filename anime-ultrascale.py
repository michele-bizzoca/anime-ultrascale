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
import pyvips

# Standard Unqualified
from os import getpid
from pathlib import Path
from enum import IntEnum
from datetime import datetime
from dataclasses import dataclass, asdict
from termios import TIOCPKT_DOSTOP
from typing import cast, Callable, TypeVar, NoReturn, TypeAlias
from types import SimpleNamespace
from time import perf_counter
from threading import Event, Thread, Lock
from subprocess import Popen, PIPE, STDOUT
from textwrap import dedent

# Custom Qualified
from PIL import Image
from psutil import Process
from dacite import from_dict, Config
from io import BytesIO

########################################################################################
# Libraries Configuration
########################################################################################

Image.MAX_IMAGE_PIXELS = None

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
NCNN_FILE        = "ncnn.txt"
BAR_FILE         = "bar.txt"
MAX_MPX          = 150000

########################################################################################
# Phases
########################################################################################

phases = [

    ( "input"     , ["import"      , "downscaling"]) ,
    ( "main"      , ["upscaling"   , "downscaling"]) ,
    ( "soft"      , ["downscaling" , "upscaling"  ]) ,
    ( "hard"      , ["downscaling" , "upscaling"  ]) ,
    ( "output"    , ["resizing"    , "export"     ])
]

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

flex_register = [

    FlexOption (

        "save"                              ,
        "s"                                 ,
        [level.name for level in SaveLevel] ,
        "endpoints"
    ),

    FlexOption (

        "tile"                      ,
        "t"                         ,
        [str(i) for i in range(16)] ,
        "0"
    )
]

########################################################################################
# Session Data
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
# Transformations Description
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
    in_width   : int
    in_height  : int
    out_width  : int
    out_height : int

@dataclass
class StepForward:
    saves    : int
    width    : int
    height   : int

@dataclass
class PhaseForward:
    pass

Unit: TypeAlias = Scaling | ScalingAI | PhaseForward | StepForward

unit_classes = [Scaling, ScalingAI, PhaseForward, StepForward]

def cost(unit: Unit) -> float:

    if isinstance(unit, ScalingAI):
        return unit.in_width * unit.in_height * 1.0 / 1000000
    elif isinstance(unit, Scaling) and unit.in_width != unit.out_width:
        return unit.out_width * unit.out_height * 0.1 / 1000000
    elif isinstance(unit, StepForward):
        return (unit.width * unit.height * 0.33 / 1000000) * unit.saves
    else:
        return 0

########################################################################################
# Error Reporting
########################################################################################

def early_fail( message   : str                              ,
                suggest   : bool = True                      ,
                exception : type[BaseException] = SystemExit ) -> NoReturn:
    raise exception( message[:1].upper() + message[1:] + "."                        +
                     (" Run with --help for usage information." if suggest else "") )

def early_assume( condition : bool                            ,
                  message   : str                             ,
                  suggest   : bool = True                     ,
                  exception: type[BaseException] = SystemExit ) -> None:
    if not condition:
        early_fail(message, suggest, exception)

########################################################################################
# Options Sorting
########################################################################################

flex_option_by_name   = {}
flex_option_by_letter = {}

for option in flex_register:
    flex_option_by_name[option.name]     = option
    flex_option_by_letter[option.letter] = option

early_assume( len(sys.argv) >= len(Argument) ,
              "incomplete I/O specification" )

fixed_arguments    = sys.argv[:len(Argument)]
flex_options       = {}
positional_options = []

i = len(Argument)
while i < len(sys.argv):

    arg = sys.argv[i]

    if arg.startswith("--"):
        key = arg[2:]
        early_assume( key in flex_option_by_name       ,
                      f"unknown floating option {arg}" )
        option = flex_option_by_name[key]

    elif arg.startswith("-"):
        key = arg[1:]
        early_assume( key in flex_option_by_letter     ,
                      f"unknown floating option {arg}" )
        option = flex_option_by_letter[key]

    else:
        early_assume( not flex_options                             ,
                      "positional option following a floating one" )
        positional_options.append(arg)
        i += 1
        continue

    early_assume( i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("-") ,
                  f"missing value for floating option {arg}"                    )

    early_assume( option.name not in flex_options              ,
                  f"multiple values for floating option {arg}" )

    early_assume( sys.argv[i + 1] in option.values           ,
                  f"unknown value for floating option {arg}" )

    flex_options[option.name] = sys.argv[i + 1]
    i += 2

for option in flex_register:
    flex_options.setdefault(option.name, option.default)

########################################################################################
# Save Level
########################################################################################

save_level = SaveLevel[flex_options["save"]]

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
# Parent Path
########################################################################################

parent_path = Path(__file__).resolve().parent

########################################################################################
# Session Folder
########################################################################################

session_folder  = parent_path / SESSION_FOLDER / invocation_date / invocation_time_pid

if save_level >= SaveLevel.text:
    session_folder.mkdir(parents = True, exist_ok = True)

########################################################################################
# Invocation File Creation
########################################################################################

invocation_file = session_folder / INVOCATION_FILE

if save_level >= SaveLevel.text:
    with open(invocation_file, "w") as invocation_handle:
        invocation_handle.write( f"PID: {invocation_pid}\n"             +
                                 f"Timestamp: {invocation_date_time}\n" +
                                 f"PWD: {Path.cwd()}\n"                 +
                                 f"Command: {' '.join(sys.argv)}\n"     )

########################################################################################
# Time String
########################################################################################

def now() -> str: return datetime.now().strftime('on %Y/%m/%d at %H:%M:%S and %f')

########################################################################################
# Logging
########################################################################################

log_file = session_folder / LOG_FILE

if save_level >= SaveLevel.text:
    log_handle = open(log_file, "w")

def log(message: str, level: SaveLevel = SaveLevel.text):
    if save_level >= SaveLevel.text and save_level >= level:
        log_handle.write( now() + f", level {descriptor(level)}: {message}\n"               )
    log_handle.flush()

if save_level >= SaveLevel.text:

    def close_log_file(): log_handle.close()
    atexit.register(close_log_file)

log("the main log system is operative")

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

temp_folder = parent_path / TEMP_FOLDER / invocation_date_time_pid
temp_folder.mkdir(parents = True, exist_ok=True)

def clean_temporaries() -> None:
    for file in temp_folder.iterdir():
        if file.is_file():
            file.unlink()

def clean_temp_folder() -> None:
    temp_folder.rmdir()

atexit.register(clean_temp_folder)
atexit.register(clean_temporaries)

log("ready to manage temporary files")

########################################################################################
# Paths
########################################################################################

renv_folder      = parent_path    / RENV_FOLDER
model_folder     = parent_path    / MODEL_FOLDER
session_file     = session_folder / SESSION_FILE
settings_file    = session_folder / SETTINGS_FILE
ncnn_file        = session_folder / NCNN_FILE
bar_file         = session_folder / BAR_FILE
temp_input_file  = temp_folder    / TEMP_INPUT_FILE
temp_output_file = temp_folder    / TEMP_OUTPUT_FILE
renv_file        = renv_folder    / RENV_FILE
input_file       = Path(fixed_arguments[Argument.input_filepath])
output_file      = Path(fixed_arguments[Argument.output_filepath])

########################################################################################
# Path Related Checks
########################################################################################

# TODO

assume(input_file.is_file(), "input file does not exist")
assume(output_file.parent.is_dir(), "output folder is not a directory")
assume(output_file.parent.exists(), "output folder does not exist")
assume(renv_file.is_file(), "missing realesrgan executable")
assume(output_file.suffix.lower() == ".png", "the output extension is not png")

########################################################################################
# Input Image Loading
########################################################################################

with Image.open(input_file) as input_image:
    assume(input_image.format is not None, "unrecognized image format")
    input_format              = cast(str, input_image.format)
    input_width, input_height = input_image.size
    input_mode                = input_image.mode
    input_image               = input_image.convert(OUTPUT_MODE)


log(f"the input image has been loaded: {input_width}x{input_height}px {OUTPUT_MODE}")

########################################################################################
# State
########################################################################################

state = SimpleNamespace(image = input_image, units = [])

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

if len(positional_options) == len(PathOption):
    settings = settings_from_path_options()
elif len(positional_options) == len(CoreOption):
    settings = settings_from_core_options()
else:
    fail( "incorrect parameter count, "
          f"{len(Argument) - 1 + len(PathOption)} or "
          f"{len(Argument) - 1 + len(CoreOption)} expected" )

log("the options have been loaded")

########################################################################################
# Disjoint Settings Validation
########################################################################################

assume( settings.main.multiplier >= 1.0 , "main multiplier < 1"       )
assume( settings.main.divisor    >= 1.0 , "main divisor < 1"          )
assume( settings.soft.multiplier >= 2   , "soft-phase multiplier < 2" )
assume( settings.soft.divisor    >= 1.0 , "soft-phase divisor < 1"    )
assume( settings.soft.iterations >= 0   , "soft-phase iterations < 0" )
assume( settings.hard.multiplier >= 2   , "hard-phase multiplier < 2" )
assume( settings.hard.divisor    >= 1.0 , "hard-phase divisor < 1"    )
assume( settings.hard.iterations >= 0   , "hard-phase iterations < 0" )

assume( (model_folder/(settings.soft.model + ".bin")).is_file()     ,
        "missing soft-phase model weights (.bin)"                   )
assume( (model_folder / (settings.soft.model + ".param")).is_file() ,
        "missing soft-phase model parameters (.param)"              )

assume( (model_folder / (settings.hard.model + ".bin")).is_file()   ,
        "missing hard-phase model weights (.bin)"                   )
assume( (model_folder / (settings.hard.model + ".param")).is_file() ,
        "missing hard-phase model parameters (.param)"              )

assume ( settings.scaling.main in ["bicubic", "lanczos"] ,
        "unknown scaling algorithm"                     )

assume ( settings.scaling.final in ["bicubic", "lanczos"] ,
        "unknown scaling algorithm"                      )

log("the settings have passed disjoint validation")

########################################################################################
# Shorthands
########################################################################################

input_min_length = int(min(input_width, input_height))
input_max_length = int(max(input_width, input_height))
input_mpx        = input_width * input_height / float(1000000)

main_factor  = settings.main.multiplier / settings.main.divisor
soft_factor  = settings.soft.multiplier / settings.soft.divisor
hard_factor  = settings.hard.multiplier / settings.hard.divisor

total_main_multiplier = settings.main.multiplier * settings.main.divisor
total_factor = max( settings.main.multiplier * max(soft_factor, hard_factor) ,
                    1 if total_main_multiplier >= settings.soft.multiplier
                      else settings.soft.multiplier / settings.main.divisor  )

output_width      = int(input_width * settings.main.multiplier)
output_height     = int(input_height * settings.main.multiplier)
output_min_length = min(output_width, output_height)
output_max_length = max(output_width, output_height)

base_main_width  = int(input_width  / settings.main.divisor)
base_main_height = int(input_height / settings.main.divisor)
base_soft_width  = int(output_width / settings.soft.divisor)
base_soft_height = int(output_height / settings.soft.divisor)
base_hard_width  = int(output_width / settings.hard.divisor)
base_hard_height = int(output_height / settings.hard.divisor)

########################################################################################
# Combined Settings Validation
########################################################################################

assume ( settings.soft.multiplier >= settings.soft.divisor ,
         "soft-phase divisor exceeds multiplier"           )

assume ( settings.hard.multiplier >= settings.hard.divisor ,
         "hard-phase divisor exceeds multiplier"            )

assume (input_min_length >= settings.main.divisor and
        output_min_length >= settings.soft.divisor and
        output_min_length >= settings.hard.divisor,
         "attempt to generate an empty intermediate image")

assume( input_mpx * total_factor ** 2 < MAX_MPX                                ,
        f"attempt to generate an intermediate image larger than {MAX_MPX} Mpx" )

log("the settings have passed combined validation")

########################################################################################
# Session Construction
########################################################################################

session = Session (

    Invocation( invocation_date + "_" + invocation_time ,
                SOFTWARE_VERSION                        ,
                flex_options["save"]                    ) ,
    ImageInfo(input_format, input_mode, input_width, input_height)   ,
    ImageInfo(OUTPUT_FORMAT, OUTPUT_MODE, output_width, output_height) ,
    settings
)

########################################################################################
# Session Recording
########################################################################################

if save_level >= SaveLevel.text:

    with open(session_file, "w") as session_handle:
        session_handle.write(export_session(session))

    log("the session file has been written")

########################################################################################
# Settings Recording
########################################################################################

if save_level >= SaveLevel.text:

    with open(settings_file, "w") as settings_handle:
        settings_handle.write(export_settings(settings))

    log("the settings file has been written")

########################################################################################
# Progress Bar
########################################################################################

def clamp(l: float | None, x: float, r:float | None) -> float:
    if l is not None:
        x = max(l, x)
    if r is not None:
        x = min(r, x)
    return x


def make_bar(percentage: float, width: int = 40) -> str:
    blocks = " ▏▎▍▌▋▊▉█"

    units = percentage / 100.0 * width
    full = int(units)
    fraction = int((units - full) * 8)

    return ( "█" * full                                       +
             (blocks[fraction] if percentage < 100.0 else "") +
             " " * max(0, width - full - 1)                   )

class ProgressBar:

    def __init__(self, total_cost: float) -> None:
        self.total_cost = total_cost
        self.total_cost_done = 0.0
        self.unit_class_name = PhaseForward.__name__
        self.unit_cost = 0.0
        self.high_speed = False
        self.unit_cost_done = 0.0
        self.progress_average_speed = 0.0
        self.refresh_average_speed = 0.0
        self.bonus_cost_done = 0.0
        self.last_progress_instant = perf_counter()
        self.last_render_instant = self.last_progress_instant
        self.last_refresh_instant = self.last_progress_instant
        self.last_progress_percentage = 0.0
        self.last_render_percentage = 0.0
        self.lock = Lock()
        self.stop_event = Event()
        self.refresh_thread = Thread( target = self._refresh_call ,
                                      daemon = True               )
        self.refresh_thread.start()
        self.timespans = {k.__name__: 0 for k in unit_classes}
        self.costs = {k.__name__: 0 for k in unit_classes}
        self.started = {k.__name__: False for k in unit_classes}
        self._render()

    def new_unit(self, unit: Unit) -> None:
        with (self.lock):
            if self.unit_cost > 0.0:
                self.progress(100.0)
            self.progress_average_speed = (
                self.costs[type(unit).__name__] / self.timespans[type(unit).__name__]
                    if self.timespans[type(unit).__name__] != 0
                    else 0 )
            self.unit_cost = cost(unit)
            self.unit_class_name = type(unit).__name__
            self.unit_cost_done = 0.0
            self.bonus_cost_done = 0.0
            self.last_progress_instant = perf_counter()
            self.last_refresh_instant = perf_counter()
            self.last_progress_percentage = 0.0

    def progress(self, percentage: float) -> None:
        delta = percentage - self.last_progress_percentage
        self.last_progress_percentage = percentage
        chunk_cost = self.unit_cost * delta / 100.0
        old_part = clamp(None, self.bonus_cost_done, chunk_cost)
        new_part = chunk_cost - old_part
        now = perf_counter()
        delta = now - self.last_progress_instant
        self.last_progress_instant = now
        if self.started[self.unit_class_name] and delta > 0.0:
            self.timespans[self.unit_class_name] += delta
            self.costs[self.unit_class_name] += chunk_cost
            current_speed = chunk_cost / delta
            self.progress_average_speed = (
                current_speed if self.progress_average_speed == 0.0
                              else ( 0.8 * self.progress_average_speed +
                                     0.2 * current_speed               ) )
        self.started[self.unit_class_name] = True
        self._old_progress(old_part)
        self._new_progress(new_part)
        self._render()

    def _old_progress(self, chunk_cost: float) -> None:
        self.bonus_cost_done -= clamp(0, chunk_cost, None)

    def _new_progress(self, chunk_cost: float) -> None:
        self.total_cost_done = clamp( None                              ,
                                      self.total_cost_done + chunk_cost ,
                                      self.total_cost                   )
        self.unit_cost_done = clamp( None                              ,
                                     self.unit_cost_done +  chunk_cost ,
                                     self.unit_cost                    )

    def refresh(self) -> None:
        with self.lock:
            now = perf_counter()
            delta = now - self.last_refresh_instant
            self.last_refresh_instant = now
            bonus_chunk_cost = clamp( 0.0,
                                      delta * self.progress_average_speed ,
                                      self.unit_cost - self.unit_cost_done )
            self.bonus_cost_done += bonus_chunk_cost
            self.total_cost_done = clamp( None,
                                          self.total_cost_done + bonus_chunk_cost ,
                                          self.total_cost                         )
            self.unit_cost_done = clamp( None                                   ,
                                         self.unit_cost_done + bonus_chunk_cost ,
                                         self.unit_cost                         )
            self._render()

    def _render(self) -> None:
        now = perf_counter()
        delta = now - self.last_render_instant
        self.last_render_instant = now
        percentage = 100.0 * ( 1.0 if self.total_cost == 0.0
                                   else self.total_cost_done / self.total_cost )
        if delta > 0.0:
            speed = ( (percentage - self.last_render_percentage) *
                      self.total_cost                            /
                      (100 * delta)                              )
            decay = math.exp(-delta / 0.6)
            self.refresh_average_speed = (
                speed if self.refresh_average_speed == 0.0
                      else ( decay * self.refresh_average_speed +
                             (1 - decay)* speed                 ) )

        self.last_render_percentage = percentage

        bar = make_bar(percentage)

        line = (
            f" [{bar}]"
            f" {percentage:6.2f}% "
            f"({self.total_cost_done:6.2f}/{self.total_cost:6.2f} Mpx), "
            f"{self.refresh_average_speed:5.2f} Mpx/s"
        )

        print( f"\r\033[K" + line ,
               end=""             ,
               flush=True         )

        if save_level >= SaveLevel.debug:
            bar_handle.write(sys.modules[__name__].now() + ": " + line + "\n")
            bar_handle.flush()

    def close(self) -> None:
        self.stop_event.set()
        self.refresh_thread.join()
        with self.lock:
            self.total_cost_done = self.total_cost
            self.last_render_percentage = 100.0
            self.refresh_average_speed = 0.0
            self._render()

    def _refresh_call(self) -> None:
        while not self.stop_event.wait(0.1):
            self.refresh()

    def __enter__(self) -> "ProgressBar":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop_event.set()
        self.refresh_thread.join()

########################################################################################
# Internal Output Naming System
########################################################################################

naming_state = SimpleNamespace(i = 0, j = 0, k = 0)

def first_step() -> bool:
    return naming_state.i == 0 and naming_state.j == 0

def last_step() -> bool:
    return ( naming_state.i == len(phases) - 1                  and
             naming_state.j == len(phases[naming_state.i][1]) - 1 )

def step_forward():

    phase, steps = phases[naming_state.i]
    step         = steps[naming_state.j]

    if ( save_level >= SaveLevel.endpoints and (first_step() or last_step()) or
         save_level >= SaveLevel.research                                     ):

        file_name = "_".join((f"{(naming_state.k + 1):02}",
                              f"{phase}-phase",
                              f"{step}-step",
                              f"{state.image.width}x{state.image.height}.png"))
        file = session_folder / file_name
        state.image.save(file)

        naming_state.k += 1

    log( f"a{'n' if step[0] in 'aeiou' else ''} {step} step in the {phase}"
         f" phase has been completed with output size {state.image.width}x"
         f"{state.image.height}" )

    if last_step():
        state.image.save(output_file)
        log(f"the output image has been saved, {state.image.width}x"
            f"{state.image.height}px {OUTPUT_FORMAT} {OUTPUT_MODE}")

    naming_state.j = (naming_state.j + 1) % len(steps)

def phase_forward() -> None:

    log(f"the {phases[naming_state.i][0]} phase has been completed")

    naming_state.i += 1
    naming_state.j = 0

########################################################################################
# NCNN Logging
########################################################################################

if save_level >= SaveLevel.debug:
    ncnn_handle = open(ncnn_file, "w")

if save_level >= SaveLevel.debug:
    def close_ncnn_file(): ncnn_handle.close()
    atexit.register(close_ncnn_file)

log("the ncnn logging system is operative")

########################################################################################
# Runners
########################################################################################

def run_phase_forward(bar: ProgressBar) -> None:
    phase_forward()
    bar.progress(100)
    
def run_step_forward(bar: ProgressBar) -> None:
    step_forward()
    bar.progress(100)

# TODO

def run_realesrgan( model      : str         ,
                    multiplier : int         ,
                    bar        : ProgressBar ) -> None:

    state.image.save(temp_input_file)

    process = Popen(

        [ str(renv_file)                                     ,
          "-i", str(temp_input_file)                         ,
          "-o", str(temp_output_file)                        ,
          "-m", str(model_folder)                            ,
          "-n", model                                        ,
          "-t", "0" if flex_options["tile"] == "0"
                    else str(64 * int(flex_options["tile"])) ,
          "-g", "0"                                          ,
          "-j", "1:1:1"                                      ,
          "-s", str(multiplier)                              ],

        stdout  = PIPE   ,
        stderr  = STDOUT ,
        text    = True   ,
        bufsize = 1
    )

    if process.stdout is None:
        fail("failed to capture Real ESRGAN's output")

    for line in process.stdout:
        if save_level >= SaveLevel.debug:
            ncnn_handle.write(now() + ": " + line)
            ncnn_handle.flush()
        line = "".join(line.split())
        if re.search(r"^[0-9]+(\.[0-9]+)?%$", line):
            bar.progress(float(line[:-1]))

    return_code = process.wait()
    assume( return_code == 0                              ,
            f"Real ESRGAN failed with code {return_code}" )

    with Image.open(temp_output_file) as result:
        state.image = result.copy()

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

########################################################################################
# Scaling Algorithms
########################################################################################

scalers = {"bicubic": "cubic", "lanczos": "lanczos3"}

main_scaler  = scalers[settings.scaling.main]
final_scaler = scalers[settings.scaling.final]

########################################################################################
# Progress Bar Logging
########################################################################################

if save_level >= SaveLevel.debug:
    bar_handle = open(bar_file, "w")

if save_level >= SaveLevel.debug:
    def close_bar_file(): bar_handle.close()
    atexit.register(close_bar_file)

log("the progress bar logging system is operative")

########################################################################################
# Planners
########################################################################################

def current_size() -> tuple[int, int]:

    for i in range(len(state.units) - 1, -1, -1):

        unit = state.units[i]

        if isinstance(unit, (Scaling, ScalingAI)):
            return unit.out_width, unit.out_height
        else:
            continue

    return input_width, input_height

def plan_algorithm( arg: float |tuple[int, int]               ,
                    algorithm: str = main_scaler ) -> None:

    in_width  , in_height  = current_size()
    out_width , out_height = ( arg if isinstance(arg, tuple)
                                   else [int(in_width  * arg),
                                         int(in_height * arg)] )
    state.units.append(Scaling( algorithm  ,
                                in_width   ,
                                in_height  ,
                                out_width  ,
                                out_height ))

def plan_realesrgan(settings: ModelSettings = settings.soft) -> None:

    in_width  , in_height  = current_size()
    out_width , out_height = ( int(in_width  * settings.multiplier) ,
                               int(in_height * settings.multiplier) )
    state.units.append(ScalingAI( settings.model ,
                                  in_width       ,
                                  in_height      ,
                                  out_width      ,
                                  out_height     ))

def plan_phase_forward() -> None:
    state.units.append(PhaseForward())

def plan_step_forward(first: bool = False, last:bool = False) -> None:
    save = last + ( (first or last) and save_level >= SaveLevel.endpoints or
                    save_level >= SaveLevel.research                       )
    width, height  = current_size()
    state.units.append(StepForward(save, width, height))
    
########################################################################################
# Planning - Input Phase
########################################################################################

plan_step_forward(True, False)

plan_algorithm((base_main_width, base_main_height))
plan_step_forward()

plan_phase_forward()

########################################################################################
# Planning - Main Phase
########################################################################################

main_iterations = math.ceil(math.log(total_main_multiplier, settings.soft.multiplier))
factor = ( (total_main_multiplier / settings.soft.multiplier ** main_iterations) **
           (1 / (main_iterations - 1) if main_iterations != 1 else 0)             )

for _ in range(main_iterations - 1):
    plan_realesrgan()
    plan_step_forward()

    plan_algorithm(factor)
    plan_step_forward()

if main_iterations != 0:
    plan_realesrgan()
    plan_step_forward()

plan_phase_forward()

########################################################################################
# Planning - Soft Phase
########################################################################################

for _ in range(settings.soft.iterations):

    plan_algorithm((base_soft_width, base_soft_height))
    plan_step_forward()

    plan_realesrgan()
    plan_step_forward()

plan_phase_forward()

########################################################################################
# Planning - Hard Phase
########################################################################################

for _ in range(settings.hard.iterations):

    plan_algorithm((base_hard_width, base_hard_height))
    plan_step_forward()

    plan_realesrgan(settings.hard)
    plan_step_forward()

plan_phase_forward()

########################################################################################
# Planning - Output Phase
########################################################################################

plan_algorithm((output_width, output_height), final_scaler)
plan_step_forward()

plan_step_forward(False, True)

plan_phase_forward()

########################################################################################
# Planning Processing
########################################################################################

total_cost = sum([cost(unit) for unit in state.units])

log("the execution plan has been created")

########################################################################################
# Newline at Exit
########################################################################################

def newline() -> None: print()
atexit.register(newline)

########################################################################################
# Plan Execution
########################################################################################

try:
    with ProgressBar(total_cost) as bar:
        for unit in state.units:
            bar.new_unit(unit)
            if isinstance(unit, Scaling):
                run_algorithm( unit.algorithm                         ,
                               (unit.out_width, unit.out_height), bar )
            elif isinstance(unit, ScalingAI):
                run_realesrgan( unit.model                           ,
                                unit.out_width // unit.in_width, bar )
            elif isinstance(unit, StepForward):
                run_step_forward(bar)
            elif isinstance(unit, PhaseForward):
                run_phase_forward(bar)
        bar.close()
except KeyboardInterrupt:
    print()
    fail("interrupted by user", False)

########################################################################################
# Help
########################################################################################

HELP = dedent("""\
    Anime-Ultrascale
    A Tool for Extreme Anime Upscaling.
    
    USAGE
    
    anime-ultrascale INPUT OUTPUT.png
        MAIN_DIVISOR MAIN_MULTIPLIER
        SOFT_MODEL SOFT_DIVISOR SOFT_MULTIPLIER SOFT_ITERATIONS
        HARD_MODEL HARD_DIVISOR HARD_MULTIPLIER HARD_ITERATIONS
        MAIN_SCALER FINAL_SCALER
        [OPTIONS]
    
    anime-ultrascale INPUT OUTPUT.png
                     {SESSION.json│SETTINGS.cfg}
                     [OPTIONS]
    
    anime-ultrascale {-h│--help}
    
    anime-ultrascale {-v│--version}
    
    POSITIONAL ARGUMENTS
    
    INPUT (str)
        Input image in any format supported by Python's Pillow.
    
    OUTPUT.png (str)
        Output image in PNG RGBA format.
    
    MAIN_DIVISOR (float)
        Main-phase downscaling divisor. Use it to revert an upscaling
        already present in the input image.
    
    MAIN_MULTIPLIER (float)
        Main-phase upscaling multiplier. It determines the output
        width and height, namely output = (input / MAIN_DIVISOR) *
        MAIN_MULTIPLIER.
    
    SOFT_MODEL (str)
        Real ESRGAN model specialized in preserving detail (basename).
    
    SOFT_DIVISOR (float)
        Soft-phase downscaling divisor. Use it to cut high frequencies
        before the successive upscaling.
    
    SOFT_MULTIPLIER (int)
        Soft-phase upscaling multiplier. It has to be supported by
        SOFT_MODEL.
    
    SOFT_ITERATIONS (int)
        Soft-phase downscaling/upscaling iterations.
    
    HARD_MODEL (str)
        Real ESRGAN model specialized in enhancing detail (basename).
    
    HARD_DIVISOR (float)
        Hard-phase downscaling divisor. Use it to cut high frequencies
        before the successive upscaling.
    
    HARD_MULTIPLIER (int)
        Hard-phase upscaling multiplier. It has to be supported by
        SOFT_MODEL.
    
    HARD_ITERATIONS (int)
        Hard-phase downscaling/upscaling iterations.
    
    MAIN_SCALER (str)
        The resizing algorithm used in all intermediate steps.
    
    FINAL_SCALER (str)
        The resizing algorithm used in the final step.
    
    {SESSION.json│SETTINGS.cfg} (str)
        A session or settings file to import settings from.
    
    {-h│--help}
        Shows this help message.
    
    {-v│--version}
        Shows the program's version.
    
    CONSTRAINTS
    
        MAIN_DIVISOR    >= 1
        MAIN_MULTIPLIER >= 1
    
        SOFT_MULTIPLIER >= 2
        SOFT_DIVISOR    >= 1
        SOFT_DIVISOR    <= SOFT_MULTIPLIER
        SOFT_ITERATIONS >= 0
    
        HARD_MULTIPLIER >= 1
        HARD_DIVISOR    >= 1
        HARD_DIVISOR    <= HARD_MULTIPLIER
        HARD_ITERATIONS >= 0
    
        MAIN_SCALER     in ['bicubic', 'lanczos']
        FINAL_SCALER    in ['bicubic', 'lanczos']
    
    OPTIONS
    
    {-t│--tile} (int)
        The width/height of the square used to tile the total area.
            0            -> automatic selection
            1 <= n <= 16 -> n * 64px
    
    {-s│--save} (str)
        Determines the session data that is saved.
            nothing   -> nothing
            text      -> basic textual data
            endpoints -> as 'text' + input/output
            research  -> as 'endpoints' + intermediate images
            debug     -> as 'research'' + debug textual data
    
    DESCRIPTION
    
    Anime-Ultrascale performs extreme image enlargement by controlled 
    alternation of downscaling and AI upscaling, where downscaling is
    performed by the bicubic and lanczos algorithms, and AI upscaling
    is performed using Real ESRGAN models.
    
    Knowledge of the three phases (main, soft, hard) is required for
    correct usage. These are described in a dedicated article, see
    the repositories section.
    
    DIRECTORY TREE
    
    Real ESRGAN models, following the usual .bin/.param convention,
    have to be stored in the 'models' folder.
    
    The official Real ESRGAN executable, 'realesrgan-ncnn-vulkan',
    has to be stored in the 'renv' folder.
    
    Useful additional data, determined by the 'save' option, will
    be stored in the 'sessions' folder.
    
    Temporary data will be stored in the 'temp' folder.
    
    If this program has been installed from the official repository
    using 'install.sh', the program's folder will contain:
    
    ┌── LICENSE
    ├── README
    ├── anime-ultrascale.py
    ├── pyproject.toml
    ├── install
    ├── bin
    │     └── anime-ultrascale
    ├── venv
    │     └── ·······
    ├── renv
    │     └── realesrgan-ncnn-vulkan
    ├── models
    │     ├── <model>.bin
    │     ├── <model>.param
    |     └── ·······
    ├── sessions
    │     ├── <date>
    │     │      ├── <time+pid>
    │     │      │        └── ·······
    │     │      └── ·······
    │     └── ·······
    └── temp
          ├── <date+time+pid>
          │      └── ·······
          └── ·······
    
    REPOSITORIES
    
    Concept -> https://github.com/michele-bizzoca/anime-upscaling
    Program -> https://github.com/michele-bizzoca/anime-ultrascale

    LICENSE

    Copyright (c) 2026 Michele Bizzoca
    Licensed under the MIT License.
""")

########################################################################################
# End
########################################################################################
