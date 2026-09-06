####################################################################################################
# Imports
####################################################################################################

import sys
import os
import time
import atexit
import re
import math
import json

#---------------------------------------------------------------------------------------------------

import dacite
import pyvips
import textwrap
import psutil
import dataclasses
import subprocess
import traceback
import numpy
import signal
import skimage
import ssim
import PIL.Image

#---------------------------------------------------------------------------------------------------

from pathlib import Path
from enum import IntEnum, Enum
from datetime import datetime
from dataclasses import dataclass
from threading import Event # Thread, Lock
from typing import NoReturn, Final, TextIO, Any, cast, NamedTuple

####################################################################################################
# Constants
####################################################################################################

SOFTWARE_VERSION   : Final = "1.0"
DEVELOPMENT_MODE   : Final = False

#---------------------------------------------------------------------------------------------------

RENV_FOLDER        : Final = "renv"
MODEL_FOLDER       : Final = "models"
PRESET_FOLDER      : Final = "presets"
SESSION_FOLDER     : Final = "sessions"
TEMP_FOLDER        : Final = "temp"

#---------------------------------------------------------------------------------------------------

RENV_FILE          : Final = "realesrgan-ncnn-vulkan"
SESSION_FILE       : Final = "session.json"
LOG_FILE           : Final = "log.txt"
INVOCATION_FILE    : Final = "invocation.txt"
SCALING_FILE       : Final = "scaling.txt"
UPSCALING_FILE     : Final = "enhancement.txt"
PROGRESS_FILE      : Final = "progress.txt"
EXIT_FILE          : Final = "exit.txt"
TEMP_INPUT_FILE    : Final = "input.png"
TEMP_OUTPUT_FILE   : Final = "output.png"

#---------------------------------------------------------------------------------------------------

DEFAULT_FORMAT        : Final = "4k"
DEFAULT_SCALER        : Final = "lanczos"
DEFAULT_COMPARER      : Final = "msss"
DEFAULT_ENDING        : Final = "bicubic"
DEFAULT_PRESET        : Final = "quality"
DEFAULT_LOGLEVEL      : Final = "text"
DEFAULT_TILE_SIZE     : Final = "4"
DEFAULT_REPAIR_MODEL  : Final = "ani2x"
DEFAULT_ENHANCE_MODEL : Final = "us4x"
DEFAULT_STYLIZE_MODEL : Final = "rpa4x"

#---------------------------------------------------------------------------------------------------

MIN_TILE_SIZE      : Final = 1
MAX_TILE_SIZE      : Final = 16
MIN_MPX_LIMIT      : Final = 1
MAX_MPX_LIMIT      : Final = 1000
MIN_DROP           : Final = 1
MAX_DROP           : Final = 16
MIN_CYCLES         : Final = 0
MAX_CYCLES         : Final = 4

#---------------------------------------------------------------------------------------------------

PROMPT_WIDTH       : Final = 80
DESCALE_SIMILARITY : Final = 0.95
DESCALE_ITERATIONS : Final = 8
OPAQUE_EXTENSIONS  : Final = ["jpg", "jpeg", "bmp"]
ALPHA_EXTENSIONS   : Final = ["png", "webp", "tif", "tiff"]
PRESET_EXTENSION   : Final = "preset"
OUTPUT_PRESET      : Final = "preset"

####################################################################################################
# Phases
####################################################################################################

PHASES : Final = [
    ( "input"     , [ "import"      , "downscaling" ,          ] ) ,
    ( "main"      , [ "upscaling"   , "downscaling" ,          ] ) ,
    ( "soft"      , [ "downscaling" , "upscaling"   ,          ] ) ,
    ( "hard"      , [ "downscaling" , "upscaling"   ,          ] ) ,
    ( "output"    , [ "downscaling" , "export"      , "saving" ] ) ]

####################################################################################################
# Invocation Data
####################################################################################################

INVOCATION_INSTANT : Final = datetime.fromtimestamp(psutil.Process().create_time())
INVOCATION_PATH    : Final = Path(__file__).resolve().parent
INVOCATION_PID     : Final = os.getpid()

#---------------------------------------------------------------------------------------------------

INVOCATION_DATE  : Final = INVOCATION_INSTANT.strftime('%Y-%m-%d')
INVOCATION_TIME  : Final = INVOCATION_INSTANT.strftime('%H-%M-%S')
INVOCATION_USEC  : Final = INVOCATION_INSTANT.strftime('%f')
INVOCATION_STAMP : Final = f"{INVOCATION_DATE}--{INVOCATION_TIME}--{INVOCATION_USEC}"

####################################################################################################
# Folder Paths
####################################################################################################

RENV_FOLDER_PATH     : Final = INVOCATION_PATH / RENV_FOLDER
MODEL_FOLDER_PATH    : Final = INVOCATION_PATH / MODEL_FOLDER
PRESET_FOLDER_PATH   : Final = INVOCATION_PATH / PRESET_FOLDER
SESSION_FOLDER_PATH  : Final = ( INVOCATION_PATH / SESSION_FOLDER / INVOCATION_DATE /
                                 f"{INVOCATION_TIME}--{INVOCATION_USEC}--{INVOCATION_PID}" )
TEMP_FOLDER_PATH     : Final = ( INVOCATION_PATH / TEMP_FOLDER /
                                 f"{INVOCATION_STAMP}--{INVOCATION_PID}" )

####################################################################################################
# File Paths
####################################################################################################

INVOCATION_FILE_PATH   : Final = SESSION_FOLDER_PATH / INVOCATION_FILE
LOG_FILE_PATH          : Final = SESSION_FOLDER_PATH / LOG_FILE
SESSION_FILE_PATH      : Final = SESSION_FOLDER_PATH / SESSION_FILE
PRESET_FILE_PATH       : Final = SESSION_FOLDER_PATH / f"{OUTPUT_PRESET}.{PRESET_EXTENSION}"
SCALING_FILE_PATH      : Final = SESSION_FOLDER_PATH / SCALING_FILE
UPSCALING_FILE_PATH    : Final = SESSION_FOLDER_PATH / UPSCALING_FILE
PROGRESS_BAR_FILE_PATH : Final = SESSION_FOLDER_PATH / PROGRESS_FILE
EXIT_FILE_PATH         : Final = SESSION_FOLDER_PATH / EXIT_FILE
TEMP_INPUT_FILE_PATH   : Final = TEMP_FOLDER_PATH    / TEMP_INPUT_FILE
TEMP_OUTPUT_FILE_PATH  : Final = TEMP_FOLDER_PATH    / TEMP_OUTPUT_FILE
RENV_FILE_PATH         : Final = RENV_FOLDER_PATH    / RENV_FILE

####################################################################################################
# Preliminaries
####################################################################################################

EXTENSIONS: Final = ALPHA_EXTENSIONS + OPAQUE_EXTENSIONS

class Size(NamedTuple): width: int; height: int

type IntMap[T] = dict[int, T]

####################################################################################################
# Log Levels
####################################################################################################

class LogLevel(IntEnum):
    dry       = 0
    nothing   = 1
    error     = 2
    text      = 3
    endpoints = 4
    debug     = 5
    research  = 6

#---------------------------------------------------------------------------------------------------

def loglevel_descriptor(loglevel: LogLevel) -> str:
    if   loglevel == LogLevel.error: return "error"
    elif loglevel == LogLevel.debug: return "debug"
    else:                            return "normal"

####################################################################################################
# Enumerations
####################################################################################################

class Model(Enum):
    repair  = "repair"
    enhance = "enhance"
    stylize = "stylize"

class Scaler(Enum):
    bilinear = "bilinear"
    bicubic  = "bicubic"
    lanczos  = "lanczos"

class Comparer(Enum):
    ssim = "ssim"
    msss = "msss"
    cwss = "cwss"

class Drop(Enum):
    auto = "auto"
    full = "full"
    root = "root"
    unit = "unit"

class Cycles(Enum):
    auto = "auto"

#---------------------------------------------------------------------------------------------------

def scaler_descriptor(scaler: Scaler) -> str:
    if   scaler == Scaler.bilinear : return "linear"
    elif scaler == Scaler.bicubic  : return "cubic"
    elif scaler == Scaler.lanczos  : return "lanczos3"
    raise ValueError

####################################################################################################
# Arguments
####################################################################################################

class BasicArgument(IntEnum):
    program_path = 0
    input_path   = 1
    output_path  = 2

class ConfigArgument(IntEnum):
    main_format    = 0
    main_scaler    = 1
    main_comparer  = 2
    main_ending    = 3
    repair_drop    = 4
    repair_model   = 5
    repair_cycles  = 6
    enhance_drop   = 7
    enhance_model  = 8
    enhance_cycles = 9
    stylize_drop   = 10
    stylize_model  = 11
    stylize_cycles = 12

class QuickArgument(IntEnum):
    format_or_preset_a = 0
    format_or_preset_b = 1

class RegularOption(Enum):
    log    = 0
    tiling = 1

class Flag(Enum):
    quiet = 0

#---------------------------------------------------------------------------------------------------

def long_option(option: str, override: bool) -> str:
    if override:
        [x, y] = option.split("_")
        return "--" + (y if x == "main" else option)
    else:
        return "--" + option

def short_option(option: str, override: bool = True) -> str:
    if override:
        [x, y] = option.split("_")
        return "-" + (y[0] if x == "main" else x[0] + y[0])
    else:
        return "-" + option[0]

override_options_map = ( { short_option(x.name, True)  : x for x in list(ConfigArgument) } |
                         { long_option(x.name, True)   : x for x in list(ConfigArgument) } )

regular_options_map  = ( { short_option(x.name, False) : x for x in list(RegularOption)  } |
                         { long_option(x.name, False)  : x for x in list(RegularOption)  } )

flags_map            = ( { short_option(x.name, False) : x for x in list(Flag)           } |
                         { long_option(x.name, False)  : x for x in list(Flag)           } )

####################################################################################################
# Settings
####################################################################################################

@dataclass
class UserMainSettings:
    format   : str      | None = None
    scaler   : Scaler   | None = None
    comparer : Comparer | None = None
    ending   : Scaler   | None = None

@dataclass
class UserStageSettings:
    drop    : float | Drop   | None = None
    model   : str            | None = None
    cycles  : int   | Cycles | None = None

@dataclass
class UserSettings:
    main    : UserMainSettings  = UserMainSettings()
    repair  : UserStageSettings = UserStageSettings()
    enhance : UserStageSettings = UserStageSettings()
    stylize : UserStageSettings = UserStageSettings()

#---------------------------------------------------------------------------------------------------

@dataclass
class GroundMainSettings:
    format   : str
    scaler   : Scaler
    comparer : Comparer
    ending   : Scaler

@dataclass
class GroundStageSettings:
    drop    : float | Drop
    model   : str
    cycles  : int   | Cycles

@dataclass
class GroundSettings:
    main    : GroundMainSettings
    repair  : GroundStageSettings
    enhance : GroundStageSettings
    stylize : GroundStageSettings

# --------------------------------------------------------------------------------------------------

def enrich_settings(base: UserSettings, extra: UserSettings) -> UserSettings:
    result = UserSettings()
    for arg in ConfigArgument:
        n, m = arg.name.split('_')
        setattr( getattr(result, n), m, getattr(getattr(base , n), m) or
                                        getattr(getattr(extra, n), m)  )
    return result

def freeze_settings(user: UserSettings) -> GroundSettings:
    ground_main    = GroundMainSettings (** vars(user.main))
    ground_repair  = GroundStageSettings(** vars(user.repair))
    ground_enhance = GroundStageSettings(** vars(user.enhance))
    ground_stylize = GroundStageSettings(** vars(user.stylize))
    return GroundSettings(ground_main, ground_repair, ground_enhance, ground_stylize)

####################################################################################################
# Sessions
####################################################################################################

@dataclass
class CallInfo:
    time    : str
    version : str
    log     : str

@dataclass
class ImageInfo:
    mode    : str
    width   : int
    height  : int

@dataclass
class ExtraInfo:
    tile    : int

@dataclass
class Session:
    invocation : CallInfo
    input      : ImageInfo
    output     : ImageInfo
    settings   : GroundSettings
    extra      : ExtraInfo

####################################################################################################
# Work Units
####################################################################################################

@dataclass
class Unit:
    def __new__(cls, *args, **kwargs):
        if cls is Unit: raise TypeError
        return super().__new__(cls)

@dataclass
class Scale(Unit):
    known_size : Size
    scaler     : Scaler
    scale      : float

@dataclass
class Upscale(Unit):
    known_size : Size
    model      : Model
    scale      : float

@dataclass
class Descale(Unit):
    known_size : Size
    comparer: Comparer
    scaler: Scaler

@dataclass
class Save(Unit):
    known_size : Size

@dataclass
class Load(Unit):
    known_size : Size

@dataclass
class StepForward(Unit):
    pass

@dataclass
class PhaseForward(Unit):
    pass

#---------------------------------------------------------------------------------------------------

unit_weights: dict[type[Unit], float] = \
    { Upscale : 1.00 , Scale   : 0.10, Descale    : 0.50 ,
      Save    : 0.30 , Load    : 0.05, StepForward: 0.00 , PhaseForward: 0.00 }

unit_categories: dict[type[Unit], int] = \
    { Upscale : 0 , Scale : 1 , Descale     : 2 ,
      Save    : 3 , Load  : 4 , StepForward : 5 , PhaseForward : 6 }

unit_entry_points: dict[type[Unit], float] = \
    { Upscale : 0.1 , Scale : 0.1 , Descale    : 0.1 ,
      Save    : 0.1 , Load  : 0.1 , StepForward: 0.1 , PhaseForward: 0.1 }

unit_exit_points: dict[type[Unit], float] = \
    { Upscale : 0.9 , Scale : 0.9 , Descale    : 0.7 ,
      Save    : 0.9 , Load  : 0.9 , StepForward: 0.9 , PhaseForward: 0.9 }

def unit_cost(unit: Unit) -> float:
    result =  unit_weights[unit.__class__]
    if isinstance(unit, (Scale, Upscale, Descale, Save, Load)):
        result *= unit.known_size.width * unit.known_size.height
    if isinstance(unit, Scale):
        result *= unit.scale ** 2
    result /= 1000000
    return result

####################################################################################################
# Path Operations
####################################################################################################

def extension(path: str | Path) -> str:
    return Path(path).suffix.lower()[1:]

####################################################################################################
# Time Operations
####################################################################################################

def timestring(timestamp: datetime):
    return timestamp.strftime('on %Y/%m/%d at %H:%M:%S and %f')

def now() -> str:
    return timestring(datetime.now())

#---------------------------------------------------------------------------------------------------

call_timestamps: dict[object, datetime]

def register(function: object):
    global call_timestamps
    call_timestamps[function] = datetime.now()

def recall(function: object) -> str:
    return timestring(call_timestamps[function])

####################################################################################################
# File Operations
####################################################################################################

def safe_open(path: Path) -> TextIO:
    handle = open(path, "w+")
    def close():handle.close()
    atexit.register(close)
    return handle

def fast_print(handle:TextIO, message: str) -> None:
    handle.write(message + "\n")
    handle.flush()

####################################################################################################
# Failing Early
####################################################################################################

def early_fail( message   : str                     ,
                suggest   : bool = True             ,
                exception : Exception | None = None ) -> NoReturn:

    suggestion = " Run with --help for usage information."
    text = f"{message[:1].upper()}{message[1:]}.{suggestion if suggest else ''}"
    if exception is not None and DEVELOPMENT_MODE:
        raise RuntimeError(text) from exception
    else:
        raise SystemExit(text)

####################################################################################################
# Information Dispatch
####################################################################################################

def information_dispatch() -> None:
    if len(sys.argv) == 1 or len(sys.argv) == 2 and sys.argv[1] in ["-h", "--help"]:
        print_help()
        exit()
    if len(sys.argv) == 2 and sys.argv[1] in ["-v", "--version"]:
        print(SOFTWARE_VERSION)
        exit()
    register(information_dispatch)

####################################################################################################
# Early Checks
####################################################################################################

def early_checks() -> None:
    if len(sys.argv) <= 2: early_fail("low-argument invocation asking neither help nor version")
    if not RENV_FILE_PATH.is_file(): early_fail("missing runner")
    if not Path(sys.argv[1]).is_file(): early_fail("invalid input file path")
    if not Path(sys.argv[2]).parent.is_dir(): early_fail("invalid output file path")
    if not extension(Path(sys.argv[1])) in EXTENSIONS: early_fail("invalid input file extension")
    if not extension(Path(sys.argv[2])) in EXTENSIONS: early_fail("invalid output file extension")
    register(early_checks)

####################################################################################################
# Arguments Sorting
####################################################################################################

input_file_path      : Path
output_file_path     : Path
positional_arguments : list[str]
override_options     : dict[ConfigArgument, str]
regular_options      : dict[RegularOption, str]
flags                : set[Flag]

def sort_arguments() -> None:

    global input_file_path
    global output_file_path
    global positional_arguments
    global override_options
    global regular_options
    global flags

    input_file_path      = Path(sys.argv[1])
    output_file_path     = Path(sys.argv[2])
    positional_arguments = []
    override_options     = {}
    regular_options      = {}
    flags                = set()

    i = 3
    while i < len(sys.argv):

        arg = sys.argv[i]

        if not arg.startswith("-"):
            if regular_options or override_options or flags:
                early_fail(f"argument '{arg}' is preceded by options" )
            positional_arguments.append(arg)
            i += 1; continue

        if arg in flags_map.keys():
            if flags_map[arg] in flags:
                early_fail(f"multiple occurrences of flag '{arg}'" )
            flags.add(flags_map[arg])
            i += 1; continue

        if i + 1 >= len(sys.argv) or sys.argv[i + 1].startswith("-"):
            early_fail(f"missing value for option '{arg}'")

        if arg in regular_options_map.keys():
            resolver  = regular_options_map
            collector = regular_options
        elif arg in override_options.keys():
            resolver  = override_options_map
            collector = override_options
        else:
            early_fail(f"the option '{arg}' is invalid")

        option = resolver[arg]
        if option.name in collector: early_fail (f"multiple values for option {arg}" )
        collector[option] = sys.argv[i + 1]
        i += 2

    register(sort_arguments)

####################################################################################################
# Regular Option Processing
####################################################################################################

loglevel  : LogLevel
tile_size : int

def process_regular_options() -> None:

    global loglevel
    global tile_size

    if RegularOption.log in regular_options.keys():
        values = [level.name for level in LogLevel]
        if regular_options[RegularOption.log] not in values:
            fail("invalid log level")

    if RegularOption.tiling in regular_options.keys():
        try: n = int(regular_options[RegularOption.tiling])
        except Exception as e: fail("tile size is not an integer", True, e)
        if n < int(MIN_TILE_SIZE) or n > int(MAX_TILE_SIZE):
            fail(f"tile size is out of range [{MIN_TILE_SIZE}, {MAX_TILE_SIZE}]")

    loglevel  = LogLevel(regular_options.get(RegularOption.log, DEFAULT_LOGLEVEL))
    tile_size = int(regular_options.get(RegularOption.tiling, DEFAULT_TILE_SIZE))

    register(process_regular_options)

####################################################################################################
# Session Folder
####################################################################################################

def create_session_folder() -> None:
    if loglevel >= LogLevel.text:
        SESSION_FOLDER_PATH.mkdir(parents = True, exist_ok = True)
    register(create_session_folder)

####################################################################################################
# Exiting
####################################################################################################

exit_file_handle: TextIO

def prepare_exit_file() -> None:
    global exit_file_handle
    if loglevel >= LogLevel.debug:
        exit_file_handle = safe_open(EXIT_FILE_PATH)
    register(prepare_exit_file)

def record_exit_message(success: bool, message: str) -> None:
    if loglevel >= LogLevel.debug:
        fast_print(exit_file_handle, f"{'SUCCESS' if success else 'FAILURE'}\n\n{message}")

####################################################################################################
# Logging
####################################################################################################

log_file_handle: TextIO

def prepare_log_file() -> None:
    global log_file_handle
    if loglevel >= LogLevel.text:
        log_file_handle = safe_open(LOG_FILE_PATH)
    register(prepare_log_file)

def log(message: str, level: LogLevel = LogLevel.text, now_ : str | None = None):

    if loglevel >= LogLevel.text and loglevel >= level:
        message = f"{now_ or now()}, level {loglevel_descriptor(level).upper()}: {message}"
        fast_print(log_file_handle, message)

####################################################################################################
# Failing
####################################################################################################

def fail( message   : str                     ,
          suggest   : bool = True             ,
          exception : Exception | None = None ) -> NoReturn:

    log(message, LogLevel.error)
    record_exit_message(False, message)
    early_fail(message, suggest, exception)

####################################################################################################
# Invocation File
####################################################################################################

def create_invocation_file() -> None:
    if loglevel >= LogLevel.text:
        INVOCATION_FILE_PATH.write_text( f"PID: {INVOCATION_PID}\n"         +
                                         f"Timestamp: {INVOCATION_STAMP}\n" +
                                         f"PWD: {Path.cwd()}\n"             +
                                         f"Command: {' '.join(sys.argv)}\n" )

####################################################################################################
# Temporary Files
####################################################################################################

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

####################################################################################################
# Scaling Logging
####################################################################################################

scaling_file_handle: TextIO

def prepare_scaling_file() -> None:
    global scaling_file_handle
    if loglevel >= LogLevel.debug:
        scaling_file_handle = safe_open(SCALING_FILE_PATH)

def record_scaling_progress(job: str, event: str, progress: Any) -> None:
    if loglevel >= LogLevel.debug:
        fast_print(scaling_file_handle, f"{now()}: "                    +
                                        f"job={job}, "                  +
                                        f"event={event}, "              +
                                        f"percent={progress.percent}, " +
                                        f"run={progress.run}, "         +
                                        f"eta={progress.eta}, "         +
                                        f"npels={progress.npels}, "     +
                                        f"tpels={progress.tpels}"       )

####################################################################################################
# Upscaling Logging
####################################################################################################

upscaling_file_handle: TextIO

def create_upscaling_file() -> None:
    global upscaling_file_handle
    if loglevel >= LogLevel.debug:
        upscaling_file_handle = safe_open(UPSCALING_FILE_PATH)

####################################################################################################
# Progress Bar
####################################################################################################

class ProgressBar:

    def __init__(self, cost: float, mpxs: float, entry: IntMap, final: IntMap) -> None:
        pass

    def __enter__(self) -> "ProgressBar":
        pass

    def __exit__(self, *args) -> None:
        pass

    def __del__(self) -> None:
        pass

    def start(self, category: int, cost: float) -> None:
        pass

    def progress(self, percentage: float) -> None:
        pass

    def stop(self) -> None:
        pass

    def finish(self) -> None:
        pass

####################################################################################################
# Progress Bar Feeding
####################################################################################################

def start_unit(unit: Unit, bar: ProgressBar) -> None:
    bar.start(unit_categories[unit.__class__], unit_cost(unit))

def create_bar(cost: float) -> ProgressBar:
    data  = numpy.random.bytes(2000 * 2000 * 3)
    image = pyvips.Image.new_from_memory(data, 2000, 2000, 3, "uchar")
    start = time.perf_counter()
    image.resize(0.5, kernel = scaler_descriptor(Scaler.lanczos)).copy_memory()
    delta = time.perf_counter() - start
    cost_ = unit_cost(Scale(Size(2000, 2000), Scaler.lanczos, 0.5))
    mpxs  = cost_ / delta
    entry = { unit_categories[u] : e for u, e in unit_entry_points.items() }
    exit_ = { unit_categories[u] : e for u, e in unit_exit_points.items()  }
    return ProgressBar(cost, mpxs, entry, exit_)

####################################################################################################
# Loading
####################################################################################################

def load(unit: Load, path: Path, bar: ProgressBar | None = None) -> pyvips.Image:

    if bar is not None: start_unit(unit, bar)

    loaded = pyvips.Image.new_from_file(str(path), access = "sequential")
    loaded = loaded.colourspace("srgb")
    if loaded.bands > 4: loaded = loaded[:4]
    loaded = loaded.cast("uchar")

    interrupted = Event()
    sigint_handler = signal.getsignal(signal.SIGINT)
    percentage = -1

    def update_interrupt(image: pyvips.Image, _) -> None:
        if interrupted.is_set(): image.set_kill(True)

    def update_progress(_, progress: Any) -> None:
        nonlocal percentage
        if bar is not None and percentage != progress.percent:
            bar.progress(float(progress.percent))
            percentage = progress.percent

    loaded.set_progress(True)

    loaded.signal_connect \
        ("preeval", lambda image, progress: record_scaling_progress("load", "preeval", progress))

    loaded.signal_connect("eval", update_interrupt)

    loaded.signal_connect("eval", update_progress)

    loaded.signal_connect \
        ("eval", lambda image, progress: record_scaling_progress("load", "eval", progress))

    loaded.signal_connect \
        ("posteval", lambda image, progress: record_scaling_progress("load", "posteval", progress))

    signal.signal(signal.SIGINT, lambda signum, frame: interrupted.set())

    try:
        if bar is not None: bar.progress(0.0)
        loaded = loaded.copy_memory()
        if bar is not None: bar.progress(100.0)

        if interrupted.is_set(): raise KeyboardInterrupt

    except pyvips.Error:
        if interrupted.is_set():
            raise KeyboardInterrupt from None
        raise

    finally:
        signal.signal(signal.SIGINT, sigint_handler)

    if bar is not None: bar.stop()
    return loaded

####################################################################################################
# Saving
####################################################################################################

def save(unit: Save, image: pyvips.Image, path: Path, bar: ProgressBar | None = None) -> None:

    if bar is not None: start_unit(unit, bar)

    copied = image.copy()

    interrupted    = Event()
    sigint_handler = signal.getsignal(signal.SIGINT)
    percentage     = -1

    def update_interrupt(image: pyvips.Image, _) -> None:
        if interrupted.is_set(): image.set_kill(True)

    def update_progress(_, progress: Any) -> None:
        nonlocal percentage
        if bar is not None and percentage != progress.percent:
            bar.progress(float(progress.percent))
            percentage = progress.percent

    copied.set_progress(True)

    copied.signal_connect \
        ("preeval", lambda image, progress: record_scaling_progress("save", "preeval", progress))

    copied.signal_connect("eval", update_interrupt)

    copied.signal_connect("eval", update_progress)

    copied.signal_connect \
        ("eval", lambda image, progress: record_scaling_progress("save", "eval", progress))

    copied.signal_connect \
        ("posteval", lambda image, progress: record_scaling_progress("save", "posteval", progress))

    signal.signal(signal.SIGINT, lambda signum, frame: interrupted.set())

    try:
        kwargs = {}
        if extension(path) == "webp":
            kwargs["lossless"] = True
            kwargs["effort"] = 4

        if bar is not None: bar.progress(0.0)
        copied.write_to_file(str(path), **kwargs)
        if bar is not None: bar.progress(100.0)

        if interrupted.is_set(): raise KeyboardInterrupt

    except pyvips.Error:
        if interrupted.is_set():
            raise KeyboardInterrupt from None
        raise

    finally:
        signal.signal(signal.SIGINT, sigint_handler)

    if bar is not None: bar.stop()

####################################################################################################
# Scaling
####################################################################################################

def scale(unit: Scale, image: pyvips.Image, bar: ProgressBar | None = None) -> pyvips.Image:

    if unit.scale == 1:
        return image.copy()

    if bar is not None: start_unit(unit, bar)

    scaled = image.resize(unit.scale, kernel = scaler_descriptor(unit.scaler))

    interrupted    = Event()
    sigint_handler = signal.getsignal(signal.SIGINT)
    percentage     = -1

    def update_interrupt(image: pyvips.Image, _) -> None:
        if interrupted.is_set(): image.set_kill(True)

    def update_progress(_, progress: Any) -> None:
        nonlocal percentage
        if bar is not None and percentage != progress.percent:
            bar.progress(float(progress.percent))
            percentage = progress.percent

    scaled.set_progress(True)

    scaled.signal_connect \
        ("preeval", lambda image, progress: record_scaling_progress("scale", "preeval", progress))

    scaled.signal_connect("eval", update_interrupt)

    scaled.signal_connect("eval", update_progress)

    scaled.signal_connect \
        ("eval", lambda image, progress: record_scaling_progress("scale", "eval", progress))

    scaled.signal_connect \
        ("posteval", lambda image, progress: record_scaling_progress("scale", "posteval", progress))

    signal.signal(signal.SIGINT, lambda signum, frame: interrupted.set())

    try:
        if bar is not None: bar.progress(0.0)
        scaled = scaled.copy_memory()
        if bar is not None: bar.progress(100.0)

        if interrupted.is_set(): raise KeyboardInterrupt

    except pyvips.Error:
        if interrupted.is_set():
            raise KeyboardInterrupt from None
        raise

    finally:
        signal.signal(signal.SIGINT, sigint_handler)

    if bar is not None: bar.stop()
    return scaled

####################################################################################################
# I/O Preparation
####################################################################################################

input_image   : pyvips.Image
input_mode    : str
input_size    : Size
output_image  : pyvips.Image
output_mode   : str

def prepare_io() -> None:

    global input_image
    global input_mode
    global input_size
    global output_image
    global output_mode

    temp = pyvips.Image.new_from_file(str(input_file_path), access = "sequential")
    iformat = str(input_file_path.suffix.lower()[1:])
    imode = temp.interpretation
    ialpha = 'alpha' if temp.hasalpha() else 'opaque'
    input_mode  = f"{iformat}--{imode}--{ialpha}"
    input_size = Size(temp.width(), temp.height())
    input_image  = load(Load(input_size), input_file_path).copy_memory()

    if ialpha == 'alpha' and extension(output_file_path) in OPAQUE_EXTENSIONS:
        fail(f"the output can't carry the input's alpha channel")

    oformat = str(output_file_path.suffix.lower()[1:])
    omode   = "srgb"
    oalpha  = ialpha
    output_mode = f"{oformat}--{omode}--{oalpha}"
    output_image = input_image.copy_memory()

####################################################################################################
# Nested Dictionary Un/Flattening
####################################################################################################

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

####################################################################################################
# Settings Import/Export
####################################################################################################

def import_settings(s : str) -> UserSettings:
    s = re.sub(r'^\s*(#.*)?$\n?', '', s, flags = re.MULTILINE)
    s = re.sub(r'^\s*(\w+)\s*=([^#\n]*)(#.*)?$\n?', r'"\1": \2,', s, flags=re.MULTILINE)
    s = "{" + s[:-1] + "}"
    return dacite.from_dict( data_class = UserSettings,
                             data = unflatten(json.loads(s)),
                             config = dacite.Config(check_types=True) )

def export_settings(s: UserSettings) -> str:
    result: str = ""
    for key, value in flatten(dataclasses.asdict(s)).items():
        result += f"{key} = {json.dumps(value)}\n"
    return result

####################################################################################################
# Session Import/Export
####################################################################################################

def import_session(s : str) -> Session:
    return dacite.from_dict( data_class = Session,
                             data       = json.loads(s),
                             config     = dacite.Config(check_types=True) )


def export_session(s: Session) -> str:
    return json.dumps(dataclasses.asdict(s), indent = 4, sort_keys = False)

####################################################################################################
# Format Interpretation
####################################################################################################

def interpret_format(s: str) -> Size | None:

    def interpret_k(s: str, horizontal: bool) -> tuple[int, int]:
        mul = int(s[:-1])
        k1  = (960.0 if horizontal else 540.0) * mul / input_size.width
        k2  = (540.0 if horizontal else 960.0) * mul / input_size.height
        k   = min(k1, k2) if s[-1:] == "k" else max(k1, k2)
        w   = round(input_size.width * k)
        h   = round(input_size.height * k)
        return w, h

    if re.fullmatch("w[0-9]+", s):
        w = int(s[1:])
        h = round(float(w) * input_size.height / input_size.width)
    elif re.fullmatch("h[0-9]+", s):
        h = int(s[1:])
        w = round(float(h) * input_size.width / input_size.height)
    elif re.fullmatch("[0-9]+%", s):
        k = float(s[:-1]) / 100.0
        w = round(input_size.width  * k)
        h = round(input_size.height * k)
    elif re.fullmatch("[0-9]+(\\.[0-9]+)?", s):
        k = float(s)
        w = round(input_size.width  * k)
        h = round(input_size.height * k)
    elif re.fullmatch("[0-9]+[kK]", s):
        w, h = interpret_k(s, input_size.width > input_size.height)
    elif re.fullmatch("[0-9]+(kh|kv|KH|KV)?", s):
        w, h = interpret_k(s[:-1], s[-1:] in "hH")
    else:
        return None

    return Size(w, h)

####################################################################################################
# Quick Arguments -> Settings
####################################################################################################

def settings_from_format_and_preset(arg1: str, arg2: str) -> UserSettings:

    b1 = interpret_format(arg1) is not None
    b2 = interpret_format(arg2) is not None

    if not b1 and not b2: fail("format is missing")
    if     b1 and     b2: fail("format specified twice")

    format_ = arg1 if b1 else arg2
    preset  = arg2 if b1 else arg1

    if extension(preset) == "preset":
        preset_path = Path(preset)
        if not preset_path.is_file():
            fail("the preset file does not exists")
    else:
        preset_path = PRESET_FOLDER_PATH / f"{preset}.{PRESET_EXTENSION}"
        if not preset_path.is_file():
            fail("the specified preset is unavailable")

    settings = import_settings(preset_path.read_text())
    settings.main.format = format_

    return settings

#---------------------------------------------------------------------------------------------------

def settings_from_two_arguments(quick_args: list[str]) -> UserSettings:
    arg1 = quick_args[QuickArgument.format_or_preset_a]
    arg2 = quick_args[QuickArgument.format_or_preset_b]
    return settings_from_format_and_preset(arg1, arg2)

def settings_from_one_argument(quick_args: list[str]) -> UserSettings:
    arg1 = quick_args[QuickArgument.format_or_preset_a]
    arg2 = DEFAULT_FORMAT if interpret_format(arg1) is None else DEFAULT_PRESET
    return settings_from_format_and_preset(arg1, arg2)

def settings_from_zero_arguments() -> UserSettings:
    return settings_from_format_and_preset(DEFAULT_FORMAT, DEFAULT_PRESET)

####################################################################################################
# Config Arguments -> Settings
####################################################################################################

def parse_cycles(name: str, s: str) -> int | Cycles | None:
    if s == 'base' : return None
    if s in Cycles : return Cycles(s)
    try: return int(s)
    except Exception as e: fail(f"the argument '{name}' is invalid", True, e)

def parse_drop(name: str, s: str) -> float | Drop | None:
    if s == 'base' : return None
    if s  in Drop  : return Drop(s)
    try: return float(s)
    except Exception as e: fail(f"the argument '{name}' is invalid", True, e)

def parse_scaler(name: str, s: str) -> Scaler | None:
    if s == 'base': return None
    try: return Scaler(s)
    except Exception as e: fail(f"the argument '{name}' is invalid", True, e)

def parse_comparer(name: str, s: str) -> Comparer | None:
    if s == 'base': return None
    try: return Comparer(s)
    except Exception as e: fail(f"the argument '{name}' is invalid", True, e)

def parse_model(_: str, s: str) -> str | None:
    if s == 'base': return None
    return s

def parse_format(_: str, s: str) -> str | None:
    if s == 'base': return None
    return s

#---------------------------------------------------------------------------------------------------

def settings_from_config_arguments(config_args: list[str]) -> UserSettings:

   def feed(arg: ConfigArgument, parser):
       return parser(arg.name.replace("_", " "), config_args[arg])

   return UserSettings \
        ( UserMainSettings  ( feed(ConfigArgument.main_format    , parse_format   ) ,
                              feed(ConfigArgument.main_scaler    , parse_scaler   ) ,
                              feed(ConfigArgument.main_comparer  , parse_comparer ) ,
                              feed(ConfigArgument.main_ending    , parse_scaler   ) ) ,
          UserStageSettings ( feed(ConfigArgument.repair_drop    , parse_drop     ) ,
                              feed(ConfigArgument.repair_model   , parse_model    ) ,
                              feed(ConfigArgument.repair_cycles  , parse_cycles   ) ) ,
          UserStageSettings ( feed(ConfigArgument.enhance_drop   , parse_drop     ) ,
                              feed(ConfigArgument.enhance_model  , parse_model    ) ,
                              feed(ConfigArgument.enhance_cycles , parse_cycles   ) ) ,
          UserStageSettings ( feed(ConfigArgument.stylize_drop   , parse_drop     ) ,
                              feed(ConfigArgument.stylize_model  , parse_model    ) ,
                              feed(ConfigArgument.stylize_cycles , parse_cycles   ) ) )

####################################################################################################
# User Settings
####################################################################################################

user_settings: UserSettings

def load_user_settings() -> None:

    global user_settings

    if   len(positional_arguments) == 0:
        user_settings = settings_from_zero_arguments()
    elif len(positional_arguments) == 1:
        user_settings = settings_from_one_argument(positional_arguments)
    elif len(positional_arguments) == 2:
        user_settings = settings_from_two_arguments(positional_arguments)
    elif len(positional_arguments) == len(ConfigArgument):
        user_settings = settings_from_config_arguments(positional_arguments)
    else:
        fail( "incorrect parameter count")

####################################################################################################
# Overrides Resolution
####################################################################################################

def resolve_overrides() -> None:
    global user_settings
    override_args     = [override_options.get(arg, "auto") for arg in ConfigArgument]
    override_settings = settings_from_config_arguments(override_args)
    user_settings     = enrich_settings(override_settings, user_settings)

####################################################################################################
# Defaults Resolution
####################################################################################################

ground_settings: GroundSettings

def resolve_defaults() -> None:
    global ground_settings
    default_args = [ DEFAULT_FORMAT        ,
                     DEFAULT_SCALER        ,
                     DEFAULT_COMPARER      ,
                     DEFAULT_ENDING        ,
                     "auto"                ,
                     DEFAULT_REPAIR_MODEL  ,
                     "auto"                ,
                     "auto"                ,
                     DEFAULT_ENHANCE_MODEL ,
                     1                     ,
                     "auto"                ,
                     DEFAULT_STYLIZE_MODEL ,
                     1                     ]
    default_settings = settings_from_config_arguments(default_args)
    final_settings   = enrich_settings(user_settings, default_settings)
    ground_settings  = freeze_settings(final_settings)

####################################################################################################
# Settings Processing
####################################################################################################

output_size     : Size
overall_scaling : float
model_names     : dict[Model, str]
model_scales    : dict[Model, float]

def process_settings() -> None:
    global ground_settings
    global output_size
    global overall_scaling
    global model_names
    global model_scales

    size = interpret_format(ground_settings.main.format)
    if size is None: fail("invalid format")
    output_size = size
    overall_scaling = output_size.width / input_size.width
    model_names = { Model.repair : ground_settings.repair.model  ,
                    Model.enhance: ground_settings.enhance.model ,
                    Model.stylize: ground_settings.stylize.model }
    model_scales = {}
    for model in Model:
        match = ( re.search("([2-8])[xX]", model_names[model]) or
                  re.search("[xX]([2-8])", model_names[model])  )
        if match is None: fail(f"cannot deduce {model.name} model's scale")
        model_scales[model] = int(match.group(1))
        if not (MODEL_FOLDER_PATH / (model_names[model] + '.bin')).is_file():
            fail(f"missing {model.name} model's weights (.bin)")
        if not (MODEL_FOLDER_PATH / (model_names[model] + '.param')).is_file():
            fail(f"missing {model.name} model's parameters (.param)")
    if ground_settings.repair.drop  is 1: ground_settings.repair.drop  = Drop.unit
    if ground_settings.enhance.drop is 1: ground_settings.enhance.drop = Drop.unit
    if ground_settings.stylize.drop is 1: ground_settings.stylize.drop = Drop.unit

####################################################################################################
# Preset File
####################################################################################################

def create_preset_file() -> None:
    if loglevel >= LogLevel.text:
        PRESET_FILE_PATH.write_text(export_settings(user_settings))

####################################################################################################
# Session Folder
####################################################################################################

session: Session

def create_session() -> None:
    global session
    session = Session ( CallInfo(INVOCATION_STAMP, SOFTWARE_VERSION, loglevel.name)   ,
                        ImageInfo(input_mode, input_size.width, input_size.height)    ,
                        ImageInfo(output_mode, output_size.width, output_size.height) ,
                        ground_settings                                               ,
                        ExtraInfo(tile_size)                                          )

####################################################################################################
# Session File
####################################################################################################

def create_session_file() -> None:
    if loglevel >= LogLevel.text:
        SESSION_FILE_PATH.write_text(export_session(session))

####################################################################################################
# Final Settings Validation
####################################################################################################

def final_settings_validation() -> None:
    
    if isinstance(ground_settings.repair.drop, float):
        if ground_settings.repair.drop < MIN_DROP: fail(f"repair drop  < {MIN_DROP}")
        if ground_settings.repair.drop > MAX_DROP: fail(f"repair drop  > {MAX_DROP}")
    if isinstance(ground_settings.repair.drop, float):
        if ground_settings.repair.drop < MIN_DROP: fail(f"enhance drop < {MIN_DROP}")
        if ground_settings.repair.drop > MAX_DROP: fail(f"enhance drop > {MAX_DROP}")
    if isinstance(ground_settings.repair.drop, float):
        if ground_settings.repair.drop < MIN_DROP: fail(f"stylize drop < {MIN_DROP}")
        if ground_settings.repair.drop > MAX_DROP: fail(f"stylize drop > {MAX_DROP}")
    if isinstance(ground_settings.repair.cycles, int):
        if ground_settings.repair.cycles < MIN_CYCLES: fail(f"repair cycles  < {MIN_CYCLES}")
        if ground_settings.repair.cycles > MAX_CYCLES: fail(f"repair cycles  > {MAX_CYCLES}")
    if isinstance(ground_settings.repair.cycles, int):
        if ground_settings.repair.cycles < MIN_CYCLES: fail(f"enhance cycles < {MIN_CYCLES}")
        if ground_settings.repair.cycles > MAX_CYCLES: fail(f"enhance cycles > {MAX_CYCLES}")
    if isinstance(ground_settings.repair.cycles, int):
        if ground_settings.repair.cycles < MIN_CYCLES: fail(f"stylize cycles < {MIN_CYCLES}")
        if ground_settings.repair.cycles > MAX_CYCLES: fail(f"stylize cycles > {MAX_CYCLES}")

####################################################################################################
# Upscaling
####################################################################################################

def upscale(unit: Upscale, bar: ProgressBar | None = None) -> None:

    if bar is not None: start_unit(unit, bar)

    process = subprocess.Popen( [ str(RENV_FILE_PATH)                      ,
                                  "-i", str(TEMP_INPUT_FILE_PATH)          ,
                                  "-o", str(TEMP_OUTPUT_FILE_PATH)         ,
                                  "-m", str(MODEL_FOLDER_PATH)             ,
                                  "-n", model_names[unit.model]            ,
                                  "-t", str(64 * tile_size)                ,
                                  "-g", "0"                                ,
                                  "-j", "1:1:1"                            ,
                                  "-s", str(model_scales[unit.model])      ],
                                  stdout  = subprocess.PIPE                 ,
                                  stderr  = subprocess.STDOUT               ,
                                  text    = True                            ,
                                  bufsize = 1                               )

    if process.stdout is None:
        fail("failed to capture the runner's output")

    for line in process.stdout:
        if loglevel >= LogLevel.debug:
            fast_print(upscaling_file_handle, f"{now()}: {line}")
        if bar is not None:
            x = re.search(r"^([0-9]+(\.[0-9]+)?)%$", line)
            if x is not None: bar.progress(float(x.group(1)))

    exit_code = process.wait()
    if exit_code != 0: fail(f"the runner failed with code {exit_code}" )

    if bar is not None: bar.stop()

####################################################################################################
# Descaling
####################################################################################################

def ndarray(image: pyvips.Image) -> numpy.ndarray:
    return numpy.ndarray( buffer = image.write_to_memory()  ,
                          dtype  = numpy.uint8              ,
                          shape=(image.height, image.width) )

def roundtrip(image: pyvips.Image, div: float) -> pyvips.Image:
    width  = max(1, round(image.width  / div))
    height = max(1, round(image.height / div))
    x = image.resize( width / image.width                      ,
                      height / image.height                    ,
                      kernel=scaler_descriptor(Scaler.lanczos) )
    return x.resize (image.width / x.width                    ,
                     image.height / x.height                  ,
                     kernel=scaler_descriptor(Scaler.lanczos) )

def similarity(reference: numpy.ndarray, candidate: numpy.ndarray, comparer: Comparer) -> float:
    if   comparer == Comparer.ssim:
        return float(skimage.metrics.structural_similarity \
                         (reference, candidate, data_range = 255))
    elif comparer == Comparer.msss:
        return float(skimage.metrics.structural_similarity \
                         (reference[..., None], candidate[..., None], data_range = 255))
    elif comparer == Comparer.cwss:
        return float(ssim.SSIM(PIL.Image.fromarray(reference_)) \
                         .cw_ssim_value(PIL.Image.fromarray(candidate_)))
    else:
        raise ValueError

#---------------------------------------------------------------------------------------------------

def descale(unit: Descale, image: pyvips.Image, bar: ProgressBar | None = None) -> pyvips.Image:

    bw   = image.copy()
    bw   = bw[:3] if bw.bands > 3 else bw
    bw   = bw.colourspace("b-w").cast("uchar")

    ref  = ndarray(bw)

    hi_div = 2.0
    while ( similarity(ref, ndarray(roundtrip(bw, hi_div))) >= DESCALE_SIMILARITY and
            (round(bw.width / hi_div) >= 1 or round(bw.height / hi_div) >= 1)       ):
        hi_div *= 2.0
    lo_div = hi_div / 2.0

    div = (lo_div + hi_div) / 2.0
    for _ in range(DESCALE_ITERATIONS):
        b = similarity(ref, ndarray(roundtrip(bw, div))) >= DESCALE_SIMILARITY
        lo_div = div if     b else lo_div
        hi_div = div if not b else hi_div
        div = (lo_div + hi_div) / 2.0

    return image.resize(1 / div, scaler_descriptor(unit.scaler)).copy_memory()

####################################################################################################
# Shorthands
####################################################################################################

# def input_min_length() : return int(min(input_width(), input_height()))
# def input_max_length() : return int(max(input_width(), input_height()))
# def input_mpx()        : return input_width() * input_height() / float(1000000)
#
# def main_factor() : return main_multiplier / settings.main.reduction
# def soft_factor() : return cast(int, settings.soft.multiplier) / settings.soft.divisor
# def hard_factor() : return cast(int, settings.hard.multiplier) / settings.hard.divisor
#
# def max_factor()   : return max(soft_factor(), hard_factor())
# def main_scaling() : return main_multiplier * settings.main.reduction
# def limit_factor() : return ( 1 if ( main_scaling()                     >=
#                                      cast(int, settings.soft.multiplier) )
#                                 else ( cast(int, settings.soft.multiplier) /
#                                        settings.main.reduction             ) )
# def total_factor() : return max(main_multiplier * max_factor(), limit_factor())
# def output_min_length() : return min(output_width, output_height)
# def output_max_length() : return max(output_width, output_height)
#
# def base_main_width()  : return int(input_width()  / settings.main.reduction)
# def base_main_height() : return int(input_height() / settings.main.reduction)
# def base_soft_width()  : return int(output_width   / settings.soft.divisor)
# def base_soft_height() : return int(output_height  / settings.soft.divisor)
# def base_hard_width()  : return int(output_width   / settings.hard.divisor)
# def base_hard_height() : return int(output_height  / settings.hard.divisor)

####################################################################################################
# Progress Bar Logging
####################################################################################################

progress_file_handle: TextIO

def create_progress_file() -> None:
    global progress_file_handle
    if loglevel >= LogLevel.debug:
        progress_file_handle = safe_open(PROGRESS_BAR_FILE_PATH)

####################################################################################################
# Run System
####################################################################################################

run_i: int
run_j: int
run_k: int

def init_run_system() -> None:

    global run_i
    global forward_j
    global forward_k

    run_i = 0
    forward_j = 0
    forward_k = 0

def step_forward(unit: StepForward, bar: ProgressBar):

    global forward_j
    global forward_k

    phase, steps = PHASES[run_i]
    step         = steps[forward_j]

    if unit.save:

        save_unit = Save(unit.width, unit.height)

        if step == "export":
            save(save_unit, current_image, output_file_path, bar)
        else:
            file_name = "_".join((f"{(forward_k + 1):02}",
                                  f"{phase}-phase",
                                  f"{step}-step",
                                  f"{unit.width}x{unit.height}.png"))
            file_path = SESSION_FOLDER_PATH / file_name
            save(save_unit, current_image, file_path, bar)

        forward_k += 1

    log(f"a{'n' if step[0] in 'aeiou' else ''} {step} "
        f"step in the {phase} phase has been completed "
        f"with output size {unit.width}x{unit.height}")

    if step == "export":
        log(f"the output image has been saved, {current_width()}x"
            f"{current_height()}px {output_mode}")

    forward_j = (forward_j + 1) % len(steps)

def phase_forward(unit : PhaseForward, bar: ProgressBar) -> None:

    global run_i
    global forward_j

    log(f"the {PHASES[run_i][0]} phase has been completed")

    run_i += 1
    forward_j = 0

########################################################################################
# Planners
########################################################################################

execution_plan: list[Unit]

def init_plan_system() -> None:

    global execution_plan

    execution_plan = []

def current_size() -> tuple[int, int]:

    for i in range(len(execution_plan) - 1, -1, -1):

        unit = execution_plan[i]

        if isinstance(unit, (Scale, Enhance)):
            return unit.out_width, unit.out_height
        else:
            continue

    return input_width(), input_height()


def plan_scale(scaler: Scaler, arg: float | tuple[int, int]) -> None:
    in_width, in_height = current_size()
    out_width, out_height = ( arg if isinstance(arg, tuple)
                                  else [ int(in_width * arg),
                                         int(in_height * arg) ] )
    multiplier =  float(out_width) / in_width
    unit = Scale(scaler, multiplier, in_width, in_height, out_width, out_height)
    execution_plan.append(unit)

def plan_enhance(model: str, multiplier: int) -> None:

    in_width  , in_height  = current_size()
    out_width , out_height = (int(in_width * multiplier), int(in_height * multiplier))
    unit = Enhance(model, multiplier, in_width, in_height, out_width, out_height)
    execution_plan.append(unit)

def plan_phase_forward() -> None:
    execution_plan.append(PhaseForward())

def plan_step_forward(save_level_: SaveLevel) -> None:
    width, height = current_size()
    execution_plan.append(StepForward(savelevel() >= save_level_, width, height))

########################################################################################
# Planning - Input Phase
########################################################################################

def plan_input_phase() -> None:

    plan_step_forward(SaveLevel.endpoints)

    plan_scale( Scaler[cast(str, settings.soft.scaler)] ,
                (base_main_width(), base_main_height()) )
    plan_step_forward(SaveLevel.research)

    plan_phase_forward()

########################################################################################
# Planning - Main Phase
########################################################################################

def plan_main_phase() -> None:

    main_iterations = math.ceil( math.log(main_scaling(),
                                 cast(int, settings.soft.multiplier)) )
    factor = ( (main_scaling() / settings.soft.multiplier ** main_iterations) **
               (1 / (main_iterations - 1) if main_iterations != 1 else 0)      )

    for _ in range(main_iterations - 1):
        plan_enhance(settings.soft.enhancer, cast(int,settings.soft.multiplier))
        plan_step_forward(SaveLevel.research)

        plan_scale(Scaler[cast(str, settings.soft.scaler)], factor)
        plan_step_forward(SaveLevel.research)

    if main_iterations != 0:
        plan_enhance(settings.soft.enhancer, cast(int, settings.soft.multiplier))
        plan_step_forward(SaveLevel.research)

    plan_phase_forward()

########################################################################################
# Planning - Soft Phase
########################################################################################

def plan_soft_phase() -> None:

    for _ in range(settings.soft.iterations):

        plan_scale( Scaler[cast(str, settings.soft.scaler)] ,
                    (base_soft_width(), base_soft_height()) )
        plan_step_forward(SaveLevel.research)

        plan_enhance(settings.soft.enhancer, cast(int, settings.soft.multiplier))
        plan_step_forward(SaveLevel.research)

    plan_phase_forward()

########################################################################################
# Planning - Hard Phase
########################################################################################

def plan_hard_phase() -> None:

    for _ in range(settings.hard.iterations):

        plan_scale( Scaler[cast(str, settings.hard.scaler)] ,
                    (base_hard_width(), base_hard_height()) )
        plan_step_forward(SaveLevel.research)

        plan_enhance(settings.hard.enhancer, cast(int, settings.hard.multiplier))
        plan_step_forward(SaveLevel.research)

    plan_phase_forward()

########################################################################################
# Planning - Output Phase
########################################################################################

def plan_output_phase() -> None:

    plan_scale( Scaler[cast(str, settings.main.closure)] ,
                (output_width, output_height)            )
    plan_step_forward(SaveLevel.endpoints)

    plan_step_forward(SaveLevel.nothing)

    plan_phase_forward()

########################################################################################
# Dry Check
########################################################################################

def dry_check() -> None:

    if savelevel() <= SaveLevel.dry and Flags.quiet not in flags:
        print("")
        print(f" tile size      : {cast(int, settings.main.tiling) * 64} px")
        print(f" input format   : {input_width()} x {input_height()} px")
        print(f" input mode     : {input_mode.replace('--', ', ')}")
        print(f" inversion      : {cast(str, settings.soft.scaler)} (" 
              f"{1 / cast(float, settings.main.reduction):.2f}".rstrip("0").rstrip(".")
              + "x)")
        print(f" normalization  : {settings.soft.enhancer} (" 
              f"{cast(float, settings.main.reduction) * main_multiplier:.2f}"
                  .rstrip("0").rstrip(".")
              + "x)")
        print(f" conservative")
        print(f"    downscaling : {cast(str, settings.soft.scaler)} ("
              f"{1 / cast(float, settings.soft.divisor):.2f}".rstrip("0").rstrip(".")
              + "x)")
        print(f"    upscaling   : {settings.soft.enhancer} "
              f"({cast(int, settings.soft.multiplier)}x)")
        print(f"    iterations  : {settings.soft.iterations}")
        print(f" strong")
        print(f"    downscaling : {cast(str, settings.hard.scaler)} ("
              f"{1 / cast(float, settings.hard.divisor):.2f}".rstrip("0").rstrip(".")
              + "x)")
        print(f"    upscaling   : {settings.hard.enhancer} "
              f"({cast(int, settings.hard.multiplier)}x)")
        print(f"    iterations  : {settings.hard.iterations}")
        print(f" finisher       : {cast(str, settings.main.closure)}")
        print(f" output format  : {output_width} x {output_height} px")
        print(f" output mode    : {output_mode.replace('--', ', ')}")
        print(f" total work     : "
              f"{sum([unit_cost(unit) for unit in execution_plan]):.2f} Mpx")
        print("")

    exit()


########################################################################################
# Execution
########################################################################################

def execute_plan() -> None:

    total_cost = sum([unit_cost(unit) for unit in execution_plan])

    with ProgressBar(total_cost) as bar:
        for unit in execution_plan:
            if isinstance(unit, Scale):
                scale(unit, bar)
            elif isinstance(unit, Enhance):
                enhance(unit, bar)
            elif isinstance(unit, StepForward):
                step_forward(unit,bar)
            elif isinstance(unit, PhaseForward):
                phase_forward(unit, bar)
        bar.complete()

########################################################################################
# Main
########################################################################################

def main():

    try:
        sort_options()
        create_session_folder()
        create_exit_file()
        create_log_file()

    except SystemExit as e:
        raise e

    except KeyboardInterrupt as e:
        early_fail(" └─→ Interrupted by user", False, e)

    except BaseException as e:
        early_fail("unexpected error", False, e)

    try:
        log( "options have been sorted" ,
             SaveLevel.text             ,
             sort_options_now           )
        log( "the session folder has been created" ,
             SaveLevel.text                        ,
             create_session_folder_now             )
        log( "the outcome record system is operative" ,
             SaveLevel.text                           ,
             create_exit_file_now                     )
        log( "the main logging system is operative" ,
             SaveLevel.text                         ,
             create_log_file_now                    )
        create_invocation_file()
        log("the invocation file has been written")
        create_temp_folder()
        log("the temporary file system is operative")
        create_scaling_file()
        log("the scaling's logging system is operative")
        create_scaling_ai_file()
        log("the AI scaling's logging system is operative")
        io_existence_checks()
        log("I/O existence checks have been passed")
        internal_existence_checks()
        log("internal existence checks have been passed")
        load_input_image()
        log("the input image has been loaded")
        load_settings()
        log("settings have been loaded")
        create_presets_file()
        log("the presets file has been written")
        resolve_defaults()
        log("defaults have been resolved")
        resolve_overrides()
        log("overrides have been resolved")
        compute_dimensions()
        log("output dimensions have been computed")
        create_session()
        log("the session has been created")
        create_session_file()
        log("the session file has been written")
        disjoint_settings_validation()
        log("settings have passed disjoint validation")
        combined_settings_validation()
        log("settings have passed combined validation")
        init_run_system()
        log("the run system is operative")
        create_progress_file()
        log("the progress logging system is operative")
        init_plan_system()
        log("the plan system is operative")
        plan_input_phase()
        log("the input phase has been planned")
        plan_main_phase()
        log("the main phase has been planned")
        plan_soft_phase()
        log("the soft phase has been planned")
        plan_hard_phase()
        log("the hard phase has been planned")
        plan_output_phase()
        log("the output phase has been planned")
        dry_check()
        log("the dry check has been performed")

    except SystemExit as e:
        raise e

    except KeyboardInterrupt as e:
        record_outcome("interrupt")
        fail(" └─→ Interrupted by user", False, e)

    except BaseException as e:
        record_outcome(traceback.format_exc())
        fail("unexpected error", False, e)

    try:
        execute_plan()
        log("the plan has been executed")

    except SystemExit as e:
        raise e

    except KeyboardInterrupt as e:
        if Flags.quiet not in flags: print()
        record_outcome("interrupt")
        fail(" └─→ Interrupted by user", False, e)

    except BaseException as e:
        if Flags.quiet not in flags: print()
        record_outcome(traceback.format_exc())
        fail("unexpected error", False, e)

    else:
        if Flags.quiet not in flags: print()
        record_outcome("success")

########################################################################################
# Help
########################################################################################

def print_help() -> None:

    def printer(s: str): print(textwrap.dedent(textwrap.dedent(s[1:])),end="")

    printer("""
        Anime-Ultrascale
        A Tool for Extreme Anime Upscaling.
    
        USAGE
    
        (1) anime-ultrascale 
              INPUT OUTPUT
              FORMAT REDUCTION CLOSURE TILING
              ENHANCER  ITERATIONS  MULTIPLIER  DIVISOR  SCALER
              ENHANCER_ ITERATIONS_ MULTIPLIER_ DIVISOR_ SCALER_
              [OPTIONS]
            
        (2) anime-ultrascale INPUT OUTPUT FORMAT PRESET [OPTIONS]
            anime-ultrascale INPUT OUTPUT PRESET FORMAT [OPTIONS]
        
        (3) anime-ultrascale INPUT OUTPUT FORMAT [OPTIONS]
            anime-ultrascale INPUT OUTPUT PRESET [OPTIONS]
        
        (4) anime-ultrascale INPUT OUTPUT [OPTIONS]
            anime-ultrascale INPUT OUTPUT [OPTIONS]
        
        (5) anime-ultrascale {-h│--help│-v│--version}
        
        (6) anime-ultrascale 
    
        EXAMPLES
        
        (1) anime-ultrascale 
              input.jpg output.png
              4k auto auto auto
              4xHFA2k 2 auto auto auto
              realesrgan-x4plus-anime 2 auto auto auto
              --log text
            
        (2) anime-ultrascale input.jpg output.png 4k quality
        
        (3) anime-ultrascale input.jpg output.png 4k
        
        (4) anime-ultrascale input.jpg output.png
        
        POSITIONAL ARGUMENTS
    
        INPUT (type: str)
            Input image in any of the following formats: PNG, JPG/JPEG, BMP, 
            TIF/TIFF, WEBP.
    
        OUTPUT.png (type: str)
            Output  image  in  any  of  the  following  formats: PNG (RGB[A]), 
            JPG/JPEG (RGB), BMP (RGB), TIF/TIFF (RGB[A]), WEBP (RGB[A]).
    
        FORMAT (type: str) (auto: 4k)
            Output  format,  all the following examples are accepted: (a) 2.0,
            (b)  200%, (c) w2160, (d)h2160, (e) 4k, 4kh, 4kv (f) 4K, 4KH, 4KV. 
            (a-b)  multiplies the input format. (c-d) fixes the output width /
            height (e) fits the input into a multiple of 960 x 540 px or 540 x
            960  px;  h  and v select the horizontal and vertical orientation, 
            and  when  absent  the input's orientation is chosen; for example,
            4kh  fits the input into 3840 x 2160 px (f) like the previous, but
            instead  of  producing  the largest image fitting into the box, it
            produces the smallest image filling the box.
            
        REDUCTION (type: float) (auto: automatic upscaling inversion)  
            The divisor of upscaling inversion.
           
        CLOSURE (type: str) (auto: bicubic)
            The algorithm to be used in the final downscaling.
            
        TILING (type: int) (auto: 4)
            The  size  of each tile, to be multiplied with 64 px. For example,
            4 leads to a tile size of 256 px.
            
        ENHANCER (type: str)
            The  name  of  the Real ESRGAN model to be used during preliminary
            upscaling  and  conservative  detail enhancement. It has be stored 
            in the 'models' folder as a '.bin'/'.param' file pair.
           
        ITERATIONS (type: str)
            The  number  of upscalings  performed  during  conservative detail 
            enhancement.
        
        MULTIPLIER (type: int) (auto: deduced by ENHANCER)
           The upscaling factor of ENHANCER.
        
        DIVISOR (type: float) (auto: sqrt(MULTIPLIER))
           The downscaling to be applied before upscalings during conservative
           detail enhancement.
           
        SCALER (type: str) (auto: bicubic)
           The  downscaling  algorithm  to  be used during conservative detail 
           enhancement.
           
        {ENHANCER_ │ ITERATIONS_ │ MULTIPLIER_ │ DIVISOR_ │ SCALER_}
           Just  as  their counterparts without underscore, but these apply to 
           strong detail enhancement. 
    
        PRESET (type: str) (auto: quality)
            The  name of a stored preset. It has to be stored in the 'presets' 
            folder  as  a '.preset' file. Each execution with log level 'text' 
            or higher saves its preset as part of session data.
    
        {-h│--help} (or no argument)
            Shows this help message.
    
        {-v│--version}
            Shows this program's version.
    
        CONSTRAINTS
    
            No  initial,  intermediate  or  final image can be either empty or 
            larger than 200 Mpx.
             
            REDUCTION  >= 1
            CLOSURE    in ['bilinear', 'bicubic', 'lanczos']
            TILING     >= 1 and <= 16
    
            iterations >= 0
            multiplier >= 2
            divisor    >= 1 and <= SOFT_MULTIPLIER
            scaler     in ['bilinear', 'bicubic', 'lanczos']
    
            ITERATIONS >= 0
            MULTIPLIER >= 2
            DIVISOR    >= 1 and <= HARD_MULTIPLIER
            SCALER     in ['bilinear', 'bicubic', 'lanczos']
            
        REGULAR OPTIONS
    
        {-l│--log} (type: str)
            Determines  which  session  data  is  saved: 
                'dry'       -> nothing (changes the output to terminal infos)
                'nothing'   -> nothing
                'text'      -> basic textual data, preset included
                'endpoints' -> as 'text'      + input/output images
                'debug'     -> as 'endpoints' + debug textual data
                'research'  -> as 'debug'     + intermediate images
        
        {-q│--quiet}
            No standard output.
            
        OVERRIDE OPTIONS
        
        Every  parameter  specified using positional arguments (possibly using
        the  default mechanic), except for INPUT and OUTPUT, can be overridden 
        with an option. Positional arguments are treated differently depending
        on  whether  they have been presented with an underscore or not. These
        two examples summarize the rules:
        
            ENHANCER  -> {-e│--enhancer}
            ENHANCER_ -> {-E│--Enhancer}
        
        DESCRIPTION
    
        Anime-Ultrascale  performs  extreme  image  enlargement  by controlled 
        alternation  of  downscaling  and  AI  upscaling, where downscaling is 
        performed  by  traditional  algorithms,  and AI upscaling is performed 
        using Real ESRGAN models.
    
        The program consists of four phases: 
            - upscaling  inversion:  detecting  and   applying  the  strongest 
              information-preserving  downscaling, as AI models will assume no 
              size inflation
            - preliminary upscaling: upscaling to the target format
            - conservative  detail enhancement: upscaling and downscaling back 
              the  image  zero or more times while preserving original details
              (adds detail moderately)
            - strong  detail  enhancement:  upscaling and downscaling back the 
              image  zero  or  more times while partly reinterpreting original
              details (adds detail considerably)
    
        PROGRESS
        
        A progress bar keeps track of the overall progress of the program. The 
        cost  unit  is  the Mpx, intended as the average time needed by a Real 
        ESRGAN model to process 1 Mpx of input data.
         
        DEPLOYMENT
    
        The  official Real ESRGAN executable, 'realesrgan-ncnn-vulkan', has to 
        be stored in the 'renv' folder.
        
        Real  ESRGAN  models  have to be stored in the 'models' folder. Such a
        model  consists  in  a  pair  of  '.bin'/'.param'  files with the same
        basename,  which  is  considered   to  be  the  model's  name.  When a
        model  multiplier  is  specified  as 'auto', it is searched for in the
        model name.
    
        Presets  have  to  be  stored  in  the 'presets' folder. A preset is a
        '.preset'  file  that  contains   every   detail   related   to  image 
        manipulation. The preset file's basename is considered to be its name.
        Unless  specified  otherwise,  every  execution  saves   its preset as 
        part  of  session  data.  The  preset  file  syntax is elementary, for 
        reference look at a generated preset.
        
        Session  files  are saved in the 'sessions' folder at the subdirectory
        'sessions/<date>/<time+pid>'. The most important session files are:
            - 'session.preset': image manipulation parameters
            - 'session.json': invocation details, I/O details, ground presets
            - 'log.txt': history of the execution with timestamps
            - <image with lowest counter>: input image in png format 
            - <image with highest counter>: output image in png format
        
        Temporary  files  are  created and deleted in the 'temp' folder, which 
        you don't need to care about.
    
        If  this  program has been downloaded from the official repository, it 
        will    include    the    models    '4xHFA2k'    (conservative)    and    
        'realesrgan-x4plus-anime'  (strong),  as well as the presets 'quality' 
        and 'speed'.
        
        If,   additionally,   the   program   has  been  installed  using  the
        repository's  'install.sh', the directory tree will be the following:
    
        ┌── anime-ultrascale.py
        ├── renv
        │     └── realesrgan-ncnn-vulkan
        ├── models
        │     ├── 4xHFA2k.bin
        │     ├── 4xHFA2k.param
        │     ├── realesrgan-x4plus-anime.bin
        │     └── realesrgan-x4plus-anime.param
        ├── presets
        │     ├── quality.preset
        │     └── speed.preset
        ├── sessions
        │     ├── <date>
        │     │      ├── <time+pid>
        │     │      │        └── ·······
        │     │      └── ·······
        │     └── ·······
        ├── temp
        │     ├── <date+time+pid>
        │     │      └── ·······
        │     └── ·······
        ├── LICENSE
        ├── README
        ├── README.md
        ├── pyproject.toml
        ├── install
        ├── third-party
        │     ├── LICENSES
        │     ├── realesrgan-ncnn-vulkan
        │     ├── 4xHFA2k.bin
        │     ├── 4xHFA2k.param
        │     ├── realesrgan-x4plus-anime.bin
        │     └── realesrgan-x4plus-anime.param        
        ├── setup-files
        │     ├── installer
        │     └── launcher
        ├── .bin
        │     └── anime-ultrascale
        ├── .venv
        │     └── ·······
        └── .gitignore      
        
        REPOSITORIES
    
        Concept -> https://github.com/michele-bizzoca/anime-upscaling
        Program -> https://github.com/michele-bizzoca/anime-ultrascale
    
        LICENSE
    
        Copyright (c) 2026 Michele Bizzoca
        Licensed under the MIT License.
""")

########################################################################################
# Main Call
########################################################################################

if __name__ == "__main__":
    main()

########################################################################################
# End
########################################################################################
