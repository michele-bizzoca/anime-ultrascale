####################################################################################################
# Imports
####################################################################################################

from __future__ import annotations

#---------------------------------------------------------------------------------------------------

import sys
import os
import time
import atexit
import re
import math
import json
import textwrap
import dataclasses
import subprocess
import traceback
import signal

#---------------------------------------------------------------------------------------------------

import dacite
import pyvips
import psutil
import numpy
import skimage
import scipy

#---------------------------------------------------------------------------------------------------

from pathlib import Path
from enum import IntEnum, Enum
from datetime import datetime
from dataclasses import dataclass, field
from threading import Event
from typing import NoReturn, Final, TextIO, Any, cast, Callable

####################################################################################################
# Constants
####################################################################################################

SOFTWARE_VERSION : Final = "2.0"
DEVELOPMENT_MODE : Final = False

#---------------------------------------------------------------------------------------------------

RENV_FOLDER        : Final = "runner"
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
UPSCALING_FILE     : Final = "upscaling.txt"
DESCALING_FILE     : Final = "descaling.txt"
BAR_FILE           : Final = "bar.txt"
EXIT_FILE          : Final = "exit.txt"
TEMP_INPUT_FILE    : Final = "input.png"
TEMP_OUTPUT_FILE   : Final = "output.png"

#---------------------------------------------------------------------------------------------------

DEFAULT_FORMAT        : Final = "4k"
DEFAULT_CLOSURE       : Final = "bicubic"
DEFAULT_DROP          : Final = "ssim95"
DEFAULT_REPAIR_MODEL  : Final = "ani2x"
DEFAULT_ENHANCE_MODEL : Final = "as2x"
DEFAULT_STYLIZE_MODEL : Final = "rpa4x"
DEFAULT_CYCLES        : Final = "1"
DEFAULT_PRESET        : Final = "quality"
DEFAULT_LOG_LEVEL     : Final = "text"
DEFAULT_TILE_SIZE     : Final = "4"

#---------------------------------------------------------------------------------------------------

MIN_SCALING_SCALE        : Final = 1
MAX_SCALING_SCALE        : Final = 99
MIN_DESCALING_SIMILARITY : Final = 80
MAX_DESCALING_SIMILARITY : Final = 95
MIN_UPSCALING_SCALE      : Final = 2
MAX_UPSCALING_SCALE      : Final = 16
MIN_CYCLES               : Final = 0
MAX_CYCLES               : Final = 4
MIN_TILE_SIZE            : Final = 1
MAX_TILE_SIZE            : Final = 16
MIN_WIDTH                : Final = 16
MIN_HEIGHT               : Final = 16

#---------------------------------------------------------------------------------------------------

PROMPT_WIDTH         : Final = 80
DESCALE_ITERATIONS   : Final = 6
OPAQUE_EXTENSIONS    : Final = ["jpg", "jpeg", "bmp"]
ALPHA_EXTENSIONS     : Final = ["png", "webp", "tif", "tiff"]
PRESET_EXTENSION     : Final = "preset"
OUTPUT_PRESET        : Final = "preset"
DEFAULT_KEYWORD      : Final = "base"
DESCALE_APPROX_RATIO : Final = 0.5
INTERNAL_SCALER      : Final = "lanczos"

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
DESCALING_FILE_PATH    : Final = SESSION_FOLDER_PATH / DESCALING_FILE
BAR_FILE_PATH          : Final = SESSION_FOLDER_PATH / BAR_FILE
EXIT_FILE_PATH         : Final = SESSION_FOLDER_PATH / EXIT_FILE
TEMP_INPUT_FILE_PATH   : Final = TEMP_FOLDER_PATH    / TEMP_INPUT_FILE
TEMP_OUTPUT_FILE_PATH  : Final = TEMP_FOLDER_PATH    / TEMP_OUTPUT_FILE
RENV_FILE_PATH         : Final = RENV_FOLDER_PATH    / RENV_FILE

####################################################################################################
# Shorthands
####################################################################################################

EXTENSIONS: Final = ALPHA_EXTENSIONS + OPAQUE_EXTENSIONS

#---------------------------------------------------------------------------------------------------

def extension(path: str | Path) -> str: return Path(path).suffix.lower()[1:]

#---------------------------------------------------------------------------------------------------

def timestring(dt: datetime): return dt.strftime('on %Y/%m/%d at %H:%M:%S and %f')

####################################################################################################
# Picture Size
####################################################################################################

@dataclass
class Size:
    width  : int
    height : int
    def __mul__(self, k: float)  -> Size:
        return Size(round(self.width * k), round(self.height * k))
    def __add__(self, x: Size) -> Size:
        return Size(self.width + x.width, self.height + x.height)

def get_size(image: pyvips.Image) -> Size:
    return Size(image.width, image.height)

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

log_level_map: Final = { LogLevel.dry       : ""       ,
                         LogLevel.nothing   : ""       ,
                         LogLevel.error     : "error"  ,
                         LogLevel.text      : "normal" ,
                         LogLevel.endpoints : "normal" ,
                         LogLevel.debug     : "debug"  ,
                         LogLevel.research  : "normal" }

####################################################################################################
# Enumerations
####################################################################################################

class Phase(str, Enum):
    repair  = "repair"
    enhance = "enhance"
    stylize = "stylize"

class Step(str, Enum):
    scale   = "scale"
    descale = "descale"
    upscale = "upscale"

# ---------------------------------------------------------------------------------------------------

class Scaler(str, Enum):
    bilinear = "bilinear"
    bicubic  = "bicubic"
    lanczos  = "lanczos"

class Comparer(str, Enum):
    ssim  = "ssim"
    psim  = "psim"
    gsim  = "gsim"

#---------------------------------------------------------------------------------------------------

scaler_map: Final = { Scaler.bilinear : "linear"   ,
                      Scaler.bicubic  : "cubic"    ,
                      Scaler.lanczos  : "lanczos3" }

####################################################################################################
# Arguments
####################################################################################################

class BasicArgument(IntEnum):
    program_path = 0
    input_path   = 1
    output_path  = 2

class ConfigArgument(IntEnum):
    main_format    = 0
    main_closure   = 1
    repair_drop    = 2
    repair_model   = 3
    repair_cycles  = 4
    enhance_drop   = 5
    enhance_model  = 6
    enhance_cycles = 7
    stylize_drop   = 8
    stylize_model  = 9
    stylize_cycles = 10

class QuickArgument(IntEnum):
    format_or_preset_a = 0
    format_or_preset_b = 1

class RegularOption(IntEnum):
    log  = 0
    tile = 1

class Flag(IntEnum):
    quiet = 0

#---------------------------------------------------------------------------------------------------

def long_opt(option: str, override: bool) -> str:
    if override:
        [x, y] = option.split("_")
        return "--" + (y if x == "main" else option.replace("_", "-"))
    else:
        return "--" + option.replace("_", "-")

def short_opt(option: str, override: bool = True) -> str:
    if override:
        [x, y] = option.split("_")
        return "-" + (y[0] if x == "main" else x[0] + y[0])
    else:
        return "-" + option[0]

#---------------------------------------------------------------------------------------------------

override_options_map: Final = ( { short_opt(x.name, True)  : x for x in list(ConfigArgument) } |
                                { long_opt(x.name, True)   : x for x in list(ConfigArgument) } )

regular_options_map: Final  = ( { short_opt(x.name, False) : x for x in list(RegularOption)  } |
                                { long_opt(x.name, False)  : x for x in list(RegularOption)  } )

flags_map: Final            = ( { short_opt(x.name, False) : x for x in list(Flag)           } |
                                { long_opt(x.name, False)  : x for x in list(Flag)           } )

####################################################################################################
# Settings
####################################################################################################

class SpecialDrop(str, Enum):
    unit = "unit"

class SpecialScale(str, Enum):
    root = "root"
    full = "full"

@dataclass
class UpscaleData:
    name  : str
    scale : int
    auto  : bool

@dataclass
class ScaleData:
    scaler : Scaler
    scale  : int | SpecialScale

@dataclass
class DescaleData:
    comparer   : Comparer
    similarity : int

#---------------------------------------------------------------------------------------------------

@dataclass
class UserMainSettings:
    format  : None | str    = None
    closure : None | Scaler = None

@dataclass
class UserStageSettings:
    drop    : None | SpecialDrop   | ScaleData | DescaleData = None
    model   : None | UpscaleData                             = None
    cycles  : None | int                                     = None

@dataclass
class UserSettings:
    main    : UserMainSettings  = field(default_factory = UserMainSettings)
    repair  : UserStageSettings = field(default_factory = UserStageSettings)
    enhance : UserStageSettings = field(default_factory = UserStageSettings)
    stylize : UserStageSettings = field(default_factory = UserStageSettings)

#---------------------------------------------------------------------------------------------------

def enrich_settings(base: UserSettings, extra: UserSettings) -> UserSettings:
    result = UserSettings()
    for arg in ConfigArgument:
        n, m = arg.name.split('_')
        x = getattr(getattr(base, n), m)
        y = getattr(getattr(extra, n), m)
        setattr(getattr(result, n), m, y if x is None else x)
    return result

#---------------------------------------------------------------------------------------------------

@dataclass
class GroundMainSettings:
    format  : str
    closure : Scaler

@dataclass
class GroundStageSettings:
    drop    : SpecialDrop | ScaleData | DescaleData
    model   : UpscaleData
    cycles  : int

@dataclass
class GroundSettings:
    main    : GroundMainSettings
    repair  : GroundStageSettings
    enhance : GroundStageSettings
    stylize : GroundStageSettings

#---------------------------------------------------------------------------------------------------

def freeze_settings(user: UserSettings) -> GroundSettings:
    return GroundSettings( GroundMainSettings (** vars(user.main))    ,
                           GroundStageSettings(** vars(user.repair))  ,
                           GroundStageSettings(** vars(user.enhance)) ,
                           GroundStageSettings(** vars(user.stylize)) )

####################################################################################################
# Sessions
####################################################################################################

@dataclass
class InvocationInfo:
    time    : str
    version : str
    log     : str
    quiet   : bool

@dataclass
class ImageInfo:
    mode   : str
    width  : int
    height : int

@dataclass
class ExtraInfo:
    tile : int

@dataclass
class Session:
    invocation : InvocationInfo
    input      : ImageInfo
    output     : ImageInfo
    settings   : GroundSettings
    extra      : ExtraInfo

####################################################################################################
# Work Units
####################################################################################################

@dataclass
class Unit:
    pass

@dataclass
class Scale(Unit):
    scaler : Scaler
    scale  : float

@dataclass
class Upscale(Unit):
    name  : str
    scale : int

@dataclass
class Descale(Unit):
    comparer   : Comparer
    similarity : float

@dataclass
class Save(Unit):
    path: Path

@dataclass
class Load(Unit):
    path: Path

#---------------------------------------------------------------------------------------------------

unit_weights: Final = { Unit : 0.00  , Upscale : 1.00 , Scale :  0.10 , Descale : 0.50 ,
                        Save : 0.30  , Load    : 0.05                                  }

unit_categories: Final = { Unit : 0 , Upscale : 1 , Scale: 2 , Descale : 3 ,
                           Save : 4 , Load    : 5                          }

#---------------------------------------------------------------------------------------------------

def unit_cost(size: Size, unit: Unit) -> float:
    result  = unit_weights[unit.__class__]
    result *= size.width * size.height
    if isinstance(unit, Scale):
        result *= unit.scale ** 2
    result /= 1000000
    return result

def unit_approx_scale(unit: Unit) -> float:
    if isinstance(unit, (Scale, Upscale)):
        return unit.scale
    elif isinstance(unit, Descale):
        return DESCALE_APPROX_RATIO
    else:
        return 1.00

####################################################################################################
# Time Recording
####################################################################################################

call_timestamps: dict[object, datetime] = {}

def register(function: object):
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

def early_fail( message   : str                         ,
                exception : BaseException | None = None ,
                suggest   : bool = True                 ) -> NoReturn:
    suggestion = " Run with --help for usage information."
    text = f"{message[:1].upper()}{message[1:]}.{suggestion if suggest else ''}"
    if exception is not None and DEVELOPMENT_MODE:
        raise RuntimeError(text) from exception
    else:
        raise SystemExit(text)

####################################################################################################
# Information Check
####################################################################################################

def information_check() -> None:
    if len(sys.argv) == 1 or len(sys.argv) == 2 and sys.argv[1] in ["-h", "--help"]:
        print_help(); sys.exit()
    if len(sys.argv) == 2 and sys.argv[1] in ["-v", "--version"]:
        print(SOFTWARE_VERSION); sys.exit()
    register(information_check)

####################################################################################################
# Early Checks
####################################################################################################

def early_checks() -> None:
    if len(sys.argv) <= 2:
        early_fail("invalid low-argument invocation")
    if not RENV_FILE_PATH.is_file():
        early_fail("missing upscaling runner")
    if not Path(sys.argv[1]).is_file():
        early_fail("non-existing input file")
    if not Path(sys.argv[2]).parent.is_dir():
        early_fail("non-existing output file's parent directory")
    if Path(sys.argv[2]).is_dir():
        early_fail("output file is a directory")
    input_ext = extension(Path(sys.argv[1]))
    if not input_ext in EXTENSIONS:
        early_fail(f"unrecognized input extension '{input_ext}'")
    output_ext = extension(Path(sys.argv[2]))
    if not output_ext in EXTENSIONS:
        early_fail(f"unrecognized output extension '{output_ext}'")
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

#---------------------------------------------------------------------------------------------------

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
                early_fail(f"positional argument '{arg}' preceded by options" )
            positional_arguments.append(arg)
            i += 1; continue

        if arg in flags_map:
            if flags_map[arg] in flags:
                early_fail(f"flag '{arg}' occurs more than once" )
            flags.add(flags_map[arg])
            i += 1; continue

        if i + 1 >= len(sys.argv) or sys.argv[i + 1].startswith("-"):
            early_fail(f"option '{arg}' has no value")

        if arg in regular_options_map:
            resolver  = regular_options_map
            collector = regular_options
        elif arg in override_options_map:
            resolver  = override_options_map
            collector = override_options
        else:
            early_fail(f"unrecognized option '{arg}'")

        option = resolver[arg]
        if option in collector:
            early_fail (f"option {arg} occurs more than once" )
        collector[option] = sys.argv[i + 1]
        i += 2

    register(sort_arguments)

####################################################################################################
# Regular Option Processing
####################################################################################################

log_level : LogLevel
tile_size : int

#---------------------------------------------------------------------------------------------------

def process_regular_options() -> None:
    global log_level
    global tile_size
    if RegularOption.log in regular_options:
        level_str = regular_options[RegularOption.log]
        if level_str not in LogLevel.__members__:
            early_fail(f"unrecognized log level '{level_str}'")
        log_level = LogLevel[level_str]
        if log_level == LogLevel.error:
            early_fail(f"unrecognized log level '{level_str}'")
    else:
        log_level = LogLevel[DEFAULT_LOG_LEVEL]
    if RegularOption.tile in regular_options:
        tile_str = regular_options[RegularOption.tile]
        try:
            tile_size = int(tile_str)
        except ValueError as e:
            early_fail(f"tile size '{tile_str}' not an integer", e)
        if tile_size < MIN_TILE_SIZE or tile_size > MAX_TILE_SIZE:
            early_fail( f"tile size '{tile_size}' out of range "
                        f"[{MIN_TILE_SIZE}, {MAX_TILE_SIZE}]" )
    else:
        tile_size = int(DEFAULT_TILE_SIZE)
    register(process_regular_options)

####################################################################################################
# Session Folder
####################################################################################################

def create_session_folder() -> None:
    if log_level >= LogLevel.text:
        SESSION_FOLDER_PATH.mkdir(parents = True)
    register(create_session_folder)

####################################################################################################
# Exiting
####################################################################################################

exit_file_handle:  TextIO
exit_file_written: bool

#---------------------------------------------------------------------------------------------------

def prepare_exit_file() -> None:
    global exit_file_handle
    global exit_file_written
    if log_level >= LogLevel.debug:
        exit_file_handle = safe_open(EXIT_FILE_PATH)
    exit_file_written = False
    register(prepare_exit_file)

def record_exit_message(success: bool, message: str) -> None:
    global exit_file_written
    if log_level >= LogLevel.debug and not exit_file_written:
        outcome = 'SUCCESS' if success else 'FAILURE'
        fast_print(exit_file_handle, f"{outcome}\n\n{message}")
        exit_file_written = True

####################################################################################################
# Logging
####################################################################################################

log_file_handle: TextIO

#---------------------------------------------------------------------------------------------------

def prepare_log_file() -> None:
    global log_file_handle
    if log_level >= LogLevel.text:
        log_file_handle = safe_open(LOG_FILE_PATH)
    register(prepare_log_file)

def log(message: str, level: LogLevel = LogLevel.text, now_ : str | None = None):
    if log_level >= LogLevel.text and log_level >= level:
        message = ( f"{now_ or timestring(datetime.now())}, "
                    f"level {log_level_map[level].upper()}: "
                    f"{message}" )
        fast_print(log_file_handle, message)

####################################################################################################
# Failing
####################################################################################################

def fail( message   : str                         ,
          exception : BaseException | None = None ,
          suggest   : bool = True                 ) -> NoReturn:
    log(message, LogLevel.error)
    record_exit_message(False, message)
    early_fail(message, exception, suggest)

####################################################################################################
# Invocation File
####################################################################################################

def create_invocation_file() -> None:
    if log_level >= LogLevel.text:
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
    TEMP_FOLDER_PATH.mkdir(parents = True)
    atexit.register(remove_temp_folder)
    atexit.register(clean_temp_folder)

####################################################################################################
# Progress Bar Logging
####################################################################################################

bar_file_handle: TextIO

#---------------------------------------------------------------------------------------------------

def create_progress_file() -> None:
    global bar_file_handle
    if log_level >= LogLevel.debug:
        bar_file_handle = safe_open(BAR_FILE_PATH)

def log_bar_progress(line: str) -> None:
    if log_level >= LogLevel.debug:
        message = f"{timestring(datetime.now())}: {line}"
        fast_print(bar_file_handle, message)

####################################################################################################
# Progress Bar
####################################################################################################

class ProgressBar:

    def __init__(self, cost: float, speed: float, log: Callable[[str], None]) -> None:
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

def start_unit(size: Size, unit: Unit, bar: ProgressBar) -> None:
    bar.start(unit_categories[unit.__class__], unit_cost(size, unit))

def create_bar(cost: float) -> ProgressBar:
    data   = numpy.random.bytes(2000 * 2000 * 3)
    image  = pyvips.Image.new_from_memory(data, 2000, 2000, 3, "uchar")
    start  = time.perf_counter()
    image.resize(0.5, kernel = scaler_map[Scaler(INTERNAL_SCALER)]).copy_memory()
    delta  = time.perf_counter() - start
    cost_  = unit_cost(Size(2000, 2000), Scale(Scaler(INTERNAL_SCALER), 0.5))
    mpxs   = cost_ / delta
    return ProgressBar(cost, mpxs, log_bar_progress)

####################################################################################################
# Scaling Logging
####################################################################################################

scaling_file_handle: TextIO

def prepare_scaling_file() -> None:
    global scaling_file_handle
    if log_level >= LogLevel.debug:
        scaling_file_handle = safe_open(SCALING_FILE_PATH)

def record_scaling_progress(job: str, event: str, progress: Any) -> None:
    if log_level >= LogLevel.debug:
        fast_print(scaling_file_handle, f"{timestring(datetime.now())}: " +
                                        f"job={job}, "                    +
                                        f"event={event}, "                +
                                        f"percent={progress.percent}, "   +
                                        f"run={progress.run}, "           +
                                        f"eta={progress.eta}, "           +
                                        f"npels={progress.npels}, "       +
                                        f"tpels={progress.tpels}"         )

####################################################################################################
# Loading
####################################################################################################

def load(unit: Load, bar: ProgressBar | None = None) -> pyvips.Image:

    interrupted    = Event()
    sigint_handler = signal.getsignal(signal.SIGINT)
    percentage     = -1
    loaded         = pyvips.Image.new_from_file(str(unit.path), access = "sequential")
    loaded         = loaded.colourspace("srgb")
    loaded         = loaded.cast("uchar")

    if loaded.hasalpha():
        if loaded.bands > 4:
            loaded = loaded[:3].bandjoin(loaded.extract_band(loaded.bands - 1))
    elif loaded.bands > 3:
        loaded = loaded[:3]

    if bar is not None: start_unit(Size(loaded.width, loaded.height), unit, bar)

    try:
        def update_interrupt(image: pyvips.Image, _) -> None:
            if interrupted.is_set(): image.set_kill(True)

        def update_progress(_, progress: Any) -> None:
            nonlocal percentage
            if bar is not None and percentage != progress.percent:
                bar.progress(float(progress.percent))
                percentage = progress.percent

        loaded.set_progress(True)
        loaded.signal_connect \
            ("preeval", lambda _, progress: record_scaling_progress("load", "preeval", progress))
        loaded.signal_connect("eval", update_interrupt)
        loaded.signal_connect("eval", update_progress)
        loaded.signal_connect \
            ("eval", lambda _, progress: record_scaling_progress("load", "eval", progress))
        loaded.signal_connect \
            ("posteval", lambda _, progress: record_scaling_progress("load", "posteval", progress))
        signal.signal(signal.SIGINT, lambda signum, frame: interrupted.set())

        if bar is not None: bar.progress(0.0)
        loaded = loaded.copy_memory()
        if bar is not None: bar.progress(100.0)
        if interrupted.is_set():
            raise KeyboardInterrupt
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

def save(unit: Save, image: pyvips.Image, bar: ProgressBar | None = None) -> None:

    if bar is not None: start_unit(Size(image.width, image.height), unit, bar)

    interrupted    = Event()
    sigint_handler = signal.getsignal(signal.SIGINT)
    percentage     = -1

    try:
        copied = image.copy()

        def update_interrupt(image: pyvips.Image, _) -> None:
            if interrupted.is_set(): image.set_kill(True)

        def update_progress(_, progress: Any) -> None:
            nonlocal percentage
            if bar is not None and percentage != progress.percent:
                bar.progress(float(progress.percent))
                percentage = progress.percent

        copied.set_progress(True)
        copied.signal_connect \
            ("preeval", lambda _, progress: record_scaling_progress("save", "preeval", progress))
        copied.signal_connect("eval", update_interrupt)
        copied.signal_connect("eval", update_progress)
        copied.signal_connect \
            ("eval", lambda _, progress: record_scaling_progress("save", "eval", progress))
        copied.signal_connect \
            ("posteval", lambda _, progress: record_scaling_progress("save", "posteval", progress))
        signal.signal(signal.SIGINT, lambda signum, frame: interrupted.set())

        kwargs = {}
        if extension(unit.path) == "webp":
            kwargs["lossless"] = True
            kwargs["effort"]   = 4
        if bar is not None: bar.progress(0.0)
        copied.write_to_file(str(unit.path), **kwargs)
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

def scale( unit: Scale                    ,
           image: pyvips.Image            ,
           vscale : float | None = None   ,
           bar: ProgressBar | None = None ) -> pyvips.Image:

    if bar is not None: start_unit(Size(image.width, image.height), unit, bar)

    interrupted    = Event()
    sigint_handler = signal.getsignal(signal.SIGINT)
    percentage     = -1

    try:
        scaled = image.resize( unit.scale                                            ,
                               vscale = vscale if vscale is not None else unit.scale ,
                               kernel = scaler_map[unit.scaler]                      )
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

        if bar is not None: bar.progress(0.0)
        scaled = scaled.copy_memory()
        if bar is not None: bar.progress(100.0)
        if interrupted.is_set():
            raise KeyboardInterrupt
    except pyvips.Error:
        if interrupted.is_set():
            raise KeyboardInterrupt from None
        raise
    finally:
        signal.signal(signal.SIGINT, sigint_handler)
        if bar is not None: bar.stop()

    return scaled

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

def format_to_str(format_: str | None) -> str:
    if format_ is None:
        return DEFAULT_KEYWORD
    return format_

def closure_to_str(closure: Scaler | None) -> str:
    if closure is None:
        return DEFAULT_KEYWORD
    elif isinstance(closure, Scaler):
        return closure.name
    raise ValueError

def drop_to_str(drop: SpecialDrop | ScaleData | DescaleData | None) -> str:
    if drop is None:
        return DEFAULT_KEYWORD
    elif isinstance(drop, SpecialDrop):
        return drop.name
    elif isinstance(drop, ScaleData):
        if isinstance(drop.scale, SpecialScale):
            return f"{drop.scaler.name}-{drop.scale.name}"
        elif isinstance(drop.scale, int):
            return f"{drop.scaler.name}{drop.scale}"
        else:
            raise ValueError
    elif isinstance(drop, DescaleData):
        return f"{drop.comparer.name}{drop.similarity}"
    raise ValueError

def model_to_str(model: UpscaleData | None) -> str:
    if model is None:
        return DEFAULT_KEYWORD
    elif isinstance(model, UpscaleData):
        return f"{model.name}{model.scale}x{'-auto' if model.auto else ''}"
    raise ValueError

def cycles_to_str(cycles: int | None) -> str:
    if cycles is None:
        return DEFAULT_KEYWORD
    elif isinstance(cycles, int):
        return str(cycles)
    raise ValueError

#---------------------------------------------------------------------------------------------------

def import_settings(s : str) -> UserSettings:
    s = re.sub(r'^\s*(#.*)?$\n?', '', s, flags = re.MULTILINE)
    s = re.sub(r'^\s*(\w+)\s*=([^#\n]*)(#.*)?$\n?', r'"\1": \2,', s, flags=re.MULTILINE)
    s = re.sub(r",\s*$", "", s)
    s = "{" + s + "}"
    return dacite.from_dict( data_class = UserSettings,
                             data = unflatten(json.loads(s)),
                             config = dacite.Config(check_types = True, cast = [Enum]) )

def export_settings(s: UserSettings) -> str:
    result: str = ""
    for key, value in flatten(dataclasses.asdict(s)).items():
        result += f"{key} = {json.dumps(value)}\n"
    return result

def rewind_settings(s: UserSettings) -> list[str]:
    return [ format_to_str(s.main.format)     ,
             closure_to_str(s.main.closure)   ,
             drop_to_str(s.repair.drop)       ,
             model_to_str(s.repair.model)     ,
             cycles_to_str(s.repair.cycles)   ,
             drop_to_str(s.enhance.drop)      ,
             model_to_str(s.enhance.model)    ,
             cycles_to_str(s.enhance.cycles)  ,
             drop_to_str(s.stylize.drop)      ,
             model_to_str(s.stylize.model)    ,
             cycles_to_str(s.stylize.cycles)  ]

####################################################################################################
# Session Import/Export
####################################################################################################

def import_session(s : str) -> Session:
    return dacite.from_dict( data_class = Session,
                             data       = json.loads(s),
                             config     = dacite.Config(check_types = True, cast = [Enum]) )


def export_session(s: Session) -> str:
    return json.dumps(dataclasses.asdict(s), indent = 4, sort_keys = False)

####################################################################################################
# Input Info Processing
####################################################################################################

input_mode: str
input_size: Size

#---------------------------------------------------------------------------------------------------

def process_input_info() -> None:
    global input_mode
    global input_size
    temp        = pyvips.Image.new_from_file(str(input_file_path))
    iext        = extension(input_file_path)
    imode       = str(cast(str, temp.interpretation))
    ialpha      = 'alpha' if temp.hasalpha() else 'opaque'
    input_mode  = f"{iext}--{imode}--{ialpha}"
    input_size  = Size(temp.width, temp.height)
    if input_size.width < MIN_WIDTH or input_size.height < MIN_HEIGHT:
        fail(f"input image too small ({input_size.width} x {input_size.height} px)")

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
    elif re.fullmatch("[0-9]+(kh|kv|KH|KV)", s):
        w, h = interpret_k(s[:-1], s[-1:] in "hH")
    else:
        return None
    return Size(w, h)

####################################################################################################
# Config Arguments -> Settings
####################################################################################################

def parse_format(s: str) -> str | None:
    if s == DEFAULT_KEYWORD: return None
    if interpret_format(s) is None:
        fail(f"unrecognized format '{s}'")
    return s

def parse_closure(s: str) -> Scaler | None:
    if s == DEFAULT_KEYWORD: return None
    if s not in Scaler.__members__:
        fail(f"unrecognized scaler '{s}'")
    return Scaler(s)

def parse_drop(s: str) -> SpecialDrop | ScaleData | DescaleData | None:
    if   s == DEFAULT_KEYWORD: return None
    elif s == SpecialDrop.unit.name : return SpecialDrop.unit
    elif s in [ f"{x.name}-{y.name}" for x in Scaler for y in SpecialScale]:
        [algorithm, scale] = s.split("-")
        return ScaleData(Scaler(algorithm), SpecialScale(scale))
    match = re.match(r"^([a-zA-Z-]+)([0-9]+)$", s)
    if match is None: fail(f"unrecognized drop '{s}'")
    algorithm, arg = match.group(1), int(match.group(2))
    if algorithm in Scaler.__members__:
        if arg < MIN_SCALING_SCALE or arg > MAX_SCALING_SCALE:
            fail( f"scale '{arg}' out of range "
                  f"[{MIN_SCALING_SCALE}, {MAX_SCALING_SCALE}]" )
        return ScaleData(Scaler(algorithm), arg)
    elif algorithm in Comparer.__members__:
        if arg < MIN_DESCALING_SIMILARITY or arg > MAX_DESCALING_SIMILARITY:
            fail(f"similarity '{arg}' out of range "
                  f"[{MIN_DESCALING_SIMILARITY}, {MAX_DESCALING_SIMILARITY}]" )
        return DescaleData(Comparer(algorithm), arg)
    fail(f"unrecognized algorithm '{algorithm}'")

def parse_model(s: str) -> UpscaleData | None:
    if s == DEFAULT_KEYWORD: return None
    match = re.match(r"^([a-zA-Z0-9-]*[a-zA-Z-])([0-9]+)x(-auto)?$", s)
    if match is None: fail(f"unrecognized model '{s}'")
    name, scale, auto = match.group(1), int(match.group(2)), bool(match.group(3))
    if scale < MIN_UPSCALING_SCALE or scale > MAX_UPSCALING_SCALE:
        fail( f"scale '{scale}' out of range "
              f"[{MIN_UPSCALING_SCALE}, {MAX_UPSCALING_SCALE}]" )
    if not (MODEL_FOLDER_PATH / f"{name}{scale}x.bin").is_file():
        fail(f"missing weights (.bin) of model '{name}{scale}x'")
    if not (MODEL_FOLDER_PATH / f"{name}{scale}x.param").is_file():
        fail(f"missing parameters (.param) of model '{name}{scale}x'")
    return UpscaleData(name, scale, auto)

def parse_cycles(s: str) -> int | None:
    if s == DEFAULT_KEYWORD: return None
    try: cycles = int(s)
    except ValueError as e: fail(f"unrecognized cycles '{s}'", e)
    if cycles < MIN_CYCLES or cycles > MAX_CYCLES:
        fail(f"cycles '{s}' out of range [{MIN_CYCLES}, {MAX_CYCLES}]")
    return cycles

#---------------------------------------------------------------------------------------------------

def settings_from_config_arguments(args: list[str]) -> UserSettings:
   return UserSettings \
        ( UserMainSettings  ( parse_format  ( args[ConfigArgument.main_format]    ) ,
                              parse_closure ( args[ConfigArgument.main_closure]   ) ) ,
          UserStageSettings ( parse_drop    ( args[ConfigArgument.repair_drop]    ) ,
                              parse_model   ( args[ConfigArgument.repair_model]   ) ,
                              parse_cycles  ( args[ConfigArgument.repair_cycles]  ) ) ,
          UserStageSettings ( parse_drop    ( args[ConfigArgument.enhance_drop]   ) ,
                              parse_model   ( args[ConfigArgument.enhance_model]  ) ,
                              parse_cycles  ( args[ConfigArgument.enhance_cycles] ) ) ,
          UserStageSettings ( parse_drop    ( args[ConfigArgument.stylize_drop]   ) ,
                              parse_model   ( args[ConfigArgument.stylize_model]  ) ,
                              parse_cycles  ( args[ConfigArgument.stylize_cycles] ) ) )

####################################################################################################
# Quick Arguments -> Settings
####################################################################################################

def settings_from_format_and_preset(arg1: str, arg2: str) -> UserSettings:

    b1 = interpret_format(arg1) is not None
    b2 = interpret_format(arg2) is not None

    if not b1 and not b2: fail("missing format")
    if     b1 and     b2: fail("format specified twice")

    format_ = arg1 if b1 else arg2
    preset  = arg2 if b1 else arg1

    if extension(preset) == "preset":
        preset_path = Path(preset)
        if not preset_path.is_file():
            fail(f"non-existing file of preset '{preset}'")
    else:
        preset_path = PRESET_FOLDER_PATH / f"{preset}.{PRESET_EXTENSION}"
        if not preset_path.is_file():
            fail(f"unrecognized preset '{preset}'")

    imported = import_settings(preset_path.read_text())
    rewind   = rewind_settings(imported)
    settings = settings_from_config_arguments(rewind)

    if interpret_format(format_) is None:
        fail(f"unrecognized format '{format_}'")
    settings.main.format = format_

    return settings

#---------------------------------------------------------------------------------------------------

def settings_from_two_arguments(quick_args: list[str]) -> UserSettings:
    fpa = quick_args[QuickArgument.format_or_preset_a]
    fpb = quick_args[QuickArgument.format_or_preset_b]
    return settings_from_format_and_preset(fpa, fpb)

def settings_from_one_argument(quick_args: list[str]) -> UserSettings:
    fpa = quick_args[QuickArgument.format_or_preset_a]
    fpb = DEFAULT_FORMAT if interpret_format(fpa) is None else DEFAULT_PRESET
    return settings_from_format_and_preset(fpa, fpb)

def settings_from_zero_arguments() -> UserSettings:
    return settings_from_format_and_preset(DEFAULT_FORMAT, DEFAULT_PRESET)

####################################################################################################
# User Settings
####################################################################################################

user_settings: UserSettings

#---------------------------------------------------------------------------------------------------

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
    override_args     = [override_options.get(arg, DEFAULT_KEYWORD) for arg in ConfigArgument]
    override_settings = settings_from_config_arguments(override_args)
    user_settings     = enrich_settings(override_settings, user_settings)

####################################################################################################
# Defaults Resolution
####################################################################################################

ground_settings: GroundSettings

#---------------------------------------------------------------------------------------------------

def resolve_defaults() -> None:
    global ground_settings
    default_args = [ DEFAULT_FORMAT        ,
                     DEFAULT_CLOSURE       ,
                     DEFAULT_DROP          ,
                     DEFAULT_REPAIR_MODEL  ,
                     DEFAULT_CYCLES        ,
                     DEFAULT_DROP          ,
                     DEFAULT_ENHANCE_MODEL ,
                     DEFAULT_CYCLES        ,
                     DEFAULT_DROP          ,
                     DEFAULT_STYLIZE_MODEL ,
                     DEFAULT_CYCLES        ]
    default_settings = settings_from_config_arguments(default_args)
    final_settings   = enrich_settings(user_settings, default_settings)
    ground_settings  = freeze_settings(final_settings)

####################################################################################################
# Output Info Processing
####################################################################################################

output_mode     : str
output_size     : Size

def process_output_info() -> None:
    global output_mode
    global output_size
    size = interpret_format(ground_settings.main.format)
    if size is None: fail(f"unrecognized format '{ground_settings.main.format}'")
    oext          = extension(output_file_path)
    omode         = "srgb"
    oalpha        = 'alpha' if input_mode.endswith('alpha') else 'opaque'
    output_mode   = f"{oext}--{omode}--{oalpha}"
    output_size   = size
    if output_size.width < MIN_WIDTH or output_size.height < MIN_HEIGHT:
        fail(f"output image too small ({output_size.width} x {output_size.height} px)")
    if input_mode.endswith('alpha') and extension(output_file_path) in OPAQUE_EXTENSIONS:
        fail(f"output can't carry input's alpha channel")

####################################################################################################
# Preset File
####################################################################################################

def create_preset_file() -> None:
    if log_level >= LogLevel.text:
        PRESET_FILE_PATH.write_text(export_settings(user_settings))

####################################################################################################
# Session File
####################################################################################################

def create_session_file() -> None:
    session = Session (
        InvocationInfo(INVOCATION_STAMP, SOFTWARE_VERSION, log_level.name, Flag.quiet in flags) ,
        ImageInfo(input_mode, input_size.width, input_size.height)                              ,
        ImageInfo(output_mode, output_size.width, output_size.height)                           ,
        ground_settings                                                                         ,
        ExtraInfo(tile_size)                                                                    )
    if log_level >= LogLevel.text:
        SESSION_FILE_PATH.write_text(export_session(session))

####################################################################################################
# Upscaling Logging
####################################################################################################

upscaling_file_handle: TextIO

#---------------------------------------------------------------------------------------------------

def create_upscaling_file() -> None:
    global upscaling_file_handle
    if log_level >= LogLevel.debug:
        upscaling_file_handle = safe_open(UPSCALING_FILE_PATH)

def record_upscaling_progress(line: str) -> None:
    if log_level >= LogLevel.debug:
        message = f"{timestring(datetime.now())}: {line}"
        fast_print(upscaling_file_handle, message)

####################################################################################################
# Descaling Logging
####################################################################################################

descaling_file_handle: TextIO

#---------------------------------------------------------------------------------------------------

def create_descaling_file() -> None:
    global descaling_file_handle
    if log_level >= LogLevel.debug:
        descaling_file_handle = safe_open(DESCALING_FILE_PATH)

def record_descaling_progress(line: str) -> None:
    if log_level >= LogLevel.debug:
        message = f"{timestring(datetime.now())}: {line}"
        fast_print(descaling_file_handle, message)

####################################################################################################
# Current Image
####################################################################################################

current_image   : pyvips.Image

####################################################################################################
# Upscaling
####################################################################################################

def upscale(unit: Upscale, bar: ProgressBar | None = None) -> None:
    size = Size(current_image.width, current_image.height)
    if bar is not None: start_unit(size, unit, bar)
    process = None
    try:
        process = subprocess.Popen( [ str(RENV_FILE_PATH)                   ,
                                      "-i", str(TEMP_INPUT_FILE_PATH)       ,
                                      "-o", str(TEMP_OUTPUT_FILE_PATH)      ,
                                      "-m", str(MODEL_FOLDER_PATH)          ,
                                      "-n", f"{unit.name}{unit.scale}x"     ,
                                      "-t", str(64 * tile_size)             ,
                                      "-g", "0"                             ,
                                      "-j", "1:1:1"                         ,
                                      "-s", str(unit.scale)                 ],
                                      stdout  = subprocess.PIPE             ,
                                      stderr  = subprocess.STDOUT           ,
                                      text    = True                        ,
                                      bufsize = 1                           )
        if process.stdout is None:
            fail("failed to capture upscaling runner's output")
        for line in process.stdout:
            line = line.rstrip("\n")
            record_upscaling_progress(line)
            if bar is not None:
                x = re.search(r"^([0-9]+(\.[0-9]+)?)%$", line)
                if x is not None: bar.progress(float(x.group(1)))
        exit_code = process.wait()
        if exit_code != 0:
            fail(f"upscaling runner failed with code {exit_code}")
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait()
        if bar is not None: bar.stop()

####################################################################################################
# Descaling
####################################################################################################

def gradient_similarity(reference: numpy.ndarray, candidate: numpy.ndarray) -> float:
    sigma = 1.0
    k     = 1e-6
    gamma = 0.5
    def gradient_magnitude(image: numpy.ndarray) -> numpy.ndarray:
        image = image.astype(numpy.float64, copy=False)
        gx = scipy.ndimage.gaussian_filter(image, sigma, order=(0, 1))
        gy = scipy.ndimage.gaussian_filter(image, sigma, order=(1, 0))
        return numpy.hypot(gx, gy)
    g_ref = gradient_magnitude(reference) ** gamma
    g_can = gradient_magnitude(candidate) ** gamma
    similarity_map = (2.0 * g_ref * g_can + k) / (g_ref ** 2 + g_can ** 2 + k)
    return numpy.mean(similarity_map)

def phase_similarity(reference: numpy.ndarray, candidate: numpy.ndarray) -> float:
    coefficient_sigma = 1.0
    window_sigma      = 3.0
    stabilizer        = 1e-6
    gf: Final = scipy.ndimage.gaussian_filter
    x, y = reference.astype(numpy.float64), candidate.astype(numpy.float64)
    def coefficients(image):
        return ( gf(image, coefficient_sigma, order=(0, 1))      +
                 1j * gf(image, coefficient_sigma, order=(1, 0)) )
    cx, cy = coefficients(x), coefficients(y)
    cross = cx * numpy.conj(cy)
    local_cross = ( gf(cross.real, window_sigma)      +
                    1j * gf(cross.imag, window_sigma) )
    energy = gf(numpy.abs(cross), window_sigma)
    coherence = (numpy.abs(local_cross) + stabilizer) / (energy + stabilizer)
    weight = numpy.sum(energy)
    if weight == 0: return 1.0
    return numpy.sum(coherence * energy) / weight

def structural_similarity(reference: numpy.ndarray, candidate: numpy.ndarray) -> float:
    return skimage.metrics.structural_similarity(reference, candidate, data_range = 255)

#---------------------------------------------------------------------------------------------------

def sim(reference: numpy.ndarray, candidate: numpy.ndarray, comparer: Comparer) -> float:
    if   comparer == Comparer.ssim:
        return structural_similarity(reference, candidate)
    elif comparer == Comparer.psim:
        return phase_similarity(reference, candidate)
    elif comparer == Comparer.gsim:
        return gradient_similarity(reference, candidate)
    raise ValueError

def ndarray(image: pyvips.Image) -> numpy.ndarray:
    return numpy.ndarray( buffer = image.write_to_memory()  ,
                          dtype  = numpy.uint8              ,
                          shape=(image.height, image.width) )

def roundtrip(image: pyvips.Image, div: float) -> pyvips.Image:
    forth = image.resize(1.0 / div, kernel = scaler_map[Scaler(INTERNAL_SCALER)])
    back  = forth.resize( image.width / forth.width                    ,
                          vscale = image.height / forth.height         ,
                          kernel = scaler_map[Scaler(INTERNAL_SCALER)] )
    return back

#---------------------------------------------------------------------------------------------------

def descale(unit: Descale, image: pyvips.Image, bar: ProgressBar | None = None) -> pyvips.Image:
    if bar is not None: start_unit(Size(image.width, image.height), unit, bar)
    try:
        bw = image.copy()
        if bw.hasalpha(): bw = bw[:-1]
        if bw.bands > 3: bw = bw[:3]
        bw = bw.colourspace("b-w").cast("uchar")
        ref  = ndarray(bw)
        max_div = min(bw.width / MIN_WIDTH, bw.height / MIN_HEIGHT)
        hi_div = min(2.0, max_div)
        bar_n = 0
        bar_p = 0
        if bw.width <= MIN_WIDTH or bw.height <= MIN_HEIGHT:
            if bar is not None: bar.progress(100.0)
            return image.copy_memory()
        lo_div = 1.0
        while round(bw.width / hi_div) >= MIN_WIDTH and round(bw.height / hi_div) >= MIN_HEIGHT:
            candidate = roundtrip(bw, hi_div)
            if candidate.width < MIN_WIDTH or candidate.height < MIN_HEIGHT: break
            similarity = sim(ref, ndarray(candidate), unit.comparer)
            record_descaling_progress(f"divisor = {hi_div:.2f}, similarity = {similarity}")
            if similarity < unit.similarity: break
            lo_div = hi_div
            next_div = min(hi_div * 2.0, max_div)
            if next_div == hi_div: break
            hi_div = next_div
            bar_p += 25.0 / 2 ** bar_n
            if bar is not None: bar.progress(bar_p)
            bar_n += 1
        else: fail("descaling search space exhausted")
        bar_p += 25.0 / 2 ** bar_n
        if bar is not None: bar.progress(bar_p)
        div = (lo_div + hi_div) / 2.0
        for i in range(DESCALE_ITERATIONS):
            similarity = sim(ref, ndarray(roundtrip(bw, div)), unit.comparer)
            record_descaling_progress(f"divisor = {div:.2f}, similarity = {similarity}")
            b = similarity >= unit.similarity
            lo_div = div if     b else lo_div
            hi_div = div if not b else hi_div
            div = (lo_div + hi_div) / 2.0
            delta_p = (90.0 - bar_p) * (i + 1) / DESCALE_ITERATIONS
            if bar is not None: bar.progress(bar_p + delta_p)
        kernel = scaler_map[Scaler(INTERNAL_SCALER)]
        result = image.resize(1.0 / lo_div, kernel = kernel).copy_memory()
        if bar is not None: bar.progress(100.0)
        return result
    finally:
        if bar is not None:
            bar.stop()

####################################################################################################
# Picture Logging
####################################################################################################

def log_input(size: Size, index: int, dry: bool, bar: ProgressBar | None = None) -> float:
    cost = 0
    if log_level >= LogLevel.endpoints:
        if not dry:
            filename = f"{index:02}_import_{current_image.width}x{current_image.height}.png"
            unit     = Save(SESSION_FOLDER_PATH / filename)
            save(unit, current_image, bar)
            cost += unit_cost(size, unit)
        else:
            cost += unit_cost(size, Save(Path()))
    if not dry: log( f"the import has been completed with output"
                     f" size {current_image.width}x{current_image.height}" )
    return cost

def log_step ( size  : Size                      ,
               index : int                       ,
               dry   : bool                      ,
               phase : Phase                     ,
               step  : Step                      ,
               bar   : ProgressBar | None = None ) -> float:
    cost = 0
    if log_level >= LogLevel.research:
        if not dry:
            filename = (f"{index:02}_"                f"{phase.name}-phase_"
                        f"{step.name}-step_"          f"{current_image.width}x"
                        f"{current_image.height}.png")
            unit = Save(SESSION_FOLDER_PATH / filename)
            save(unit, current_image, bar)
            cost += unit_cost(size, unit)
        else:
            cost += unit_cost(size, Save(Path()))
    if not dry:
        article = "an" if step.name[0] in 'aeiou' else 'a'
        log( f"{article} {step.name} step in the {phase.name} phase has been "
             f"completed with output size {current_image.width}x{current_image.height}" )
    return cost

def log_output(size: Size, index: int, dry: bool, bar: ProgressBar | None = None) -> float:
    cost = 0
    if log_level >= LogLevel.endpoints:
        if not dry:
            filename = f"{index:02}_export_{current_image.width}x{current_image.height}.png"
            unit = Save(SESSION_FOLDER_PATH / filename)
            save(unit, current_image, bar)
            cost += unit_cost(size, unit)
        else:
            cost += unit_cost(size, Save(Path()))
    if log_level >= LogLevel.nothing:
        if not dry:
            unit = Save(output_file_path)
            save(unit, current_image, bar)
            cost += unit_cost(size, unit)
        else:
            cost += unit_cost(size, Save(Path()))
    if not dry: log ( f"the export has been completed with output"
                      f" size {current_image.width}x{current_image.height}" )
    return cost

####################################################################################################
# Processing
####################################################################################################

def process(dry: bool, bar: ProgressBar | None = None) -> float:

    global current_image

    i              = 1
    cost           = 0
    estimated_size = input_size

    def index(): nonlocal i; i += 1; return i - 1

    unit = Load(input_file_path)
    if not dry: current_image = load(unit, bar)
    cost += unit_cost(estimated_size, unit)
    cost += log_input(estimated_size, index(), dry, bar)

    for phase in Phase:

        s: GroundStageSettings = getattr(ground_settings, phase.name)

        for _ in range(s.cycles):
            match s.drop:
                case SpecialDrop.unit:
                    pass
                case ScaleData():
                    step_size = estimated_size if dry else get_size(current_image)
                    match s.drop.scale:
                        case SpecialScale.root:
                            scale_ = 1.0 / math.sqrt(s.model.scale)
                        case SpecialScale.full:
                            scale_ = 1.0 / s.model.scale
                        case int():
                            scale_ = s.drop.scale / 100.0
                        case _:
                            raise ValueError
                    min_scale = max(MIN_WIDTH / step_size.width, MIN_HEIGHT / step_size.height)
                    unit = Scale(s.drop.scaler, max(scale_, min_scale))
                    if not dry: current_image = scale(unit, current_image, None, bar)
                    cost += unit_cost(estimated_size, unit)
                    cost += log_step(estimated_size, index(), dry, phase, Step.scale, bar)
                    estimated_size *= unit_approx_scale(unit)
                case DescaleData():
                    similarity_ = s.drop.similarity / 100.0
                    unit = Descale(s.drop.comparer, similarity_)
                    if not dry: current_image = descale(unit, current_image, bar)
                    cost += unit_cost(estimated_size, unit)
                    cost += log_step(estimated_size, index(), dry, phase, Step.descale, bar)
                    estimated_size *= unit_approx_scale(unit)
                case _:
                    raise ValueError

            match s.model:
                case UpscaleData():
                    w = estimated_size.width if dry else current_image.width
                    if s.model.auto and w >= output_size.width:
                        pass
                    elif not s.model.auto or w * s.model.scale >= output_size.width:
                        step_size = estimated_size
                        unit = Save(TEMP_INPUT_FILE_PATH)
                        if not dry: save(unit, current_image, bar)
                        cost += unit_cost(estimated_size, unit)
                        unit = Upscale(s.model.name, s.model.scale)
                        if not dry: upscale(unit, bar)
                        cost += unit_cost(estimated_size, unit)
                        estimated_size *= unit_approx_scale(unit)
                        unit = Load(TEMP_OUTPUT_FILE_PATH)
                        if not dry: current_image = load(unit, bar)
                        cost += unit_cost(estimated_size, unit)
                        cost += log_step(step_size, index(), dry, phase, Step.upscale, bar)
                        if s.model.auto:
                            k = output_size.width / (w * s.model.scale)
                            unit = Scale(Scaler(INTERNAL_SCALER), k)
                            if not dry: current_image = scale(unit, current_image, None, bar)
                            cost += unit_cost(estimated_size, unit)
                            estimated_size = output_size
                    else:
                        r = output_size.width / w
                        n = math.ceil(math.log(r, s.model.scale))
                        k = (r / s.model.scale **  n) ** (1.0 / (n - 1))
                        for j in range(n):
                            step_size = estimated_size
                            unit = Save(TEMP_INPUT_FILE_PATH)
                            if not dry: save(unit, current_image, bar)
                            cost += unit_cost(estimated_size, unit)
                            unit = Upscale(s.model.name, s.model.scale)
                            if not dry: upscale(unit, bar)
                            cost += unit_cost(estimated_size, unit)
                            estimated_size *= unit_approx_scale(unit)
                            unit = Load(TEMP_OUTPUT_FILE_PATH)
                            if not dry: current_image = load(unit, bar)
                            cost += unit_cost(estimated_size, unit)
                            cost += log_step(step_size, index(), dry, phase, Step.upscale, bar)
                            if j < n - 1:
                                unit = Scale(Scaler(INTERNAL_SCALER), k)
                                if not dry: current_image = scale(unit, current_image, None, bar)
                                cost += unit_cost(estimated_size, unit)
                                estimated_size *= unit_approx_scale(unit)
                        estimated_size = output_size
                case _:
                    raise ValueError

    w = estimated_size.width  if dry else current_image.width
    h = estimated_size.height if dry else current_image.height
    hscale = output_size.width  / w
    vscale = output_size.height / h
    unit = Scale(ground_settings.main.closure, hscale)
    if not dry: current_image = scale(unit, current_image, vscale, bar)
    cost += unit_cost(estimated_size, unit)
    cost += log_output(estimated_size, index(), dry, bar)

    return cost

####################################################################################################
# Dry Check
####################################################################################################

def dry_check(cost: float) -> None:
    if log_level <= LogLevel.dry:
        if Flag.quiet not in flags:
            print("")
            print(f" input format   : {input_size.width} x {input_size.height} px")
            print(f" input mode     : {input_mode.replace('--', ', ')}")
            print(f" closure        : {closure_to_str(ground_settings.main.closure)}")
            print(f" repair")
            print(f"    downscaling : {drop_to_str(ground_settings.repair.drop)}")
            print(f"    upscaling   : {model_to_str(ground_settings.repair.model)}")
            print(f"    cycles      : {cycles_to_str(ground_settings.repair.cycles)}")
            print(f" enhance")
            print(f"    downscaling : {drop_to_str(ground_settings.enhance.drop)}")
            print(f"    upscaling   : {model_to_str(ground_settings.enhance.model)}")
            print(f"    cycles      : {cycles_to_str(ground_settings.enhance.cycles)}")
            print(f" stylize")
            print(f"    downscaling : {drop_to_str(ground_settings.stylize.drop)}")
            print(f"    upscaling   : {model_to_str(ground_settings.stylize.model)}")
            print(f"    cycles      : {cycles_to_str(ground_settings.stylize.cycles)}")
            print(f" output format  : {output_size.width} x {output_size.height} px")
            print(f" output mode    : {output_mode.replace('--', ', ')}")
            print(f" tile size      : {int(tile_size) * 64} px")
            print(f" total work     : {cost:.2f} Mpx")
            print("")
        sys.exit()

####################################################################################################
# Main
####################################################################################################

def main():

    try:
        information_check()
        early_checks()
        sort_arguments()
        process_regular_options()
        create_session_folder()
        prepare_exit_file()
        prepare_log_file()
    except SystemExit as e:
        raise e
    except KeyboardInterrupt as e:
        early_fail(" └─→ keyboard interrupt", e, False)
    except BaseException as e:
        early_fail("unexpected error", e, False)

    try:
        log("information check performed", LogLevel.text, recall(information_check))
        log("early checks performed", LogLevel.text, recall(early_checks))
        log("arguments organized", LogLevel.text, recall(sort_arguments))
        log("regular options processed", LogLevel.text, recall(process_regular_options))
        log("session folder created", LogLevel.text, recall(create_session_folder))
        log("exit file prepared", LogLevel.text, recall(prepare_exit_file))
        log("log file prepared", LogLevel.text, recall(prepare_log_file))
        create_invocation_file()
        log("invocation file written")
        create_temp_folder()
        log("temporary folder created")
        create_progress_file()
        log("progress file created")
        prepare_scaling_file()
        log("scaling file prepared")
        process_input_info()
        log("input info processed")
        load_user_settings()
        log("user settings loaded")
        resolve_overrides()
        log("overrides resolved")
        resolve_defaults()
        log("defaults resolved")
        process_output_info()
        log("output info processed")
        create_preset_file()
        log("preset file created")
        create_session_file()
        log("session_file_created")
        create_upscaling_file()
        log("upscaling file created")
        create_descaling_file()
        log("descaling file created")
    except SystemExit as e:
        raise e
    except KeyboardInterrupt as e:
        record_exit_message(False, "keyboard interrupt")
        fail(" └─→ keyboard interrupt", e, False)
    except BaseException as e:
        record_exit_message(False, traceback.format_exc())
        fail("unexpected error", e, False)
    try:
        cost = process(True)
        log("execution plan created")
        dry_check(cost)
        log("dry check performed")
        with create_bar(cost) as bar:
            process(False, bar)
        log("execution plan executed")
    except SystemExit as e:
        raise e
    except KeyboardInterrupt as e:
        if Flag.quiet not in flags: print()
        record_exit_message(False, "keyboard interrupt")
        fail(" └─→ keyboard interrupt", e, False)
    except BaseException as e:
        if Flag.quiet not in flags: print()
        record_exit_message(False, traceback.format_exc())
        fail("unexpected error", e, False)
    else:
        if Flag.quiet not in flags: print()
        record_exit_message(True, "")

####################################################################################################
# Help
####################################################################################################

def print_help() -> None:

    def printer(s: str): print(textwrap.dedent(textwrap.dedent(s[1:])),end="")

    printer("""
        Under construction.
""")

####################################################################################################
# Main Call
####################################################################################################

if __name__ == "__main__":
    main()

####################################################################################################
# End
####################################################################################################
