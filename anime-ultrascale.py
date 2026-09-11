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
from dataclasses import dataclass
from threading import Event
from typing import NoReturn, Final, TextIO, Any, cast, Callable

####################################################################################################
# Constants
####################################################################################################

SOFTWARE_VERSION : Final = "1.0"
DEVELOPMENT_MODE : Final = False

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
DEFAULT_ENHANCE_MODEL : Final = "us4x"
DEFAULT_STYLIZE_MODEL : Final = "rpa4x"
DEFAULT_CYCLES        : Final = "1"
DEFAULT_PRESET        : Final = "quality"
DEFAULT_LOGLEVEL      : Final = "text"
DEFAULT_TILE_SIZE     : Final = "4"

#---------------------------------------------------------------------------------------------------

MIN_SCALE_SCALE    : Final = 1
MAX_SCALE_SCALE    : Final = 99
MIN_DESCALE_TARGET : Final = 80
MAX_DESCALE_TARGET : Final = 95
MIN_UPSCALE_SCALE  : Final = 1
MAX_UPSCALE_SCALE  : Final = 16
MIN_CYCLES         : Final = 0
MAX_CYCLES         : Final = 4
MIN_TILE_SIZE      : Final = 1
MAX_TILE_SIZE      : Final = 16

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

class Phase(Enum):
    repair  = "repair"
    enhance = "enhance"
    stylize = "stylize"

class Scaler(Enum):
    bilinear = "bilinear"
    bicubic  = "bicubic"
    lanczos  = "lanczos"

class Comparer(Enum):
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
    stylize_cycles = 1

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

class SpecialDrop(Enum):
    unit = "unit"

class SpecialScale(Enum):
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
    main    : UserMainSettings  = UserMainSettings()
    repair  : UserStageSettings = UserStageSettings()
    enhance : UserStageSettings = UserStageSettings()
    stylize : UserStageSettings = UserStageSettings()

#---------------------------------------------------------------------------------------------------

def enrich_settings(base: UserSettings, extra: UserSettings) -> UserSettings:
    result = UserSettings()
    for arg in ConfigArgument:
        n, m = arg.name.split('_')
        x = getattr(getattr(base, n), m)
        y = getattr(getattr(extra, n), m)
        setattr( getattr(result, n), m, x or y)
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

#---------------------------------------------------------------------------------------------------

class Step(Enum):
    open    = "open"
    scale   = "scale"
    descale = "descale"
    upscale = "upscale"

#---------------------------------------------------------------------------------------------------

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
    scale : float

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

def early_fail( message   : str                         ,
                suggest   : bool = True                 ,
                exception : BaseException | None = None ) -> NoReturn:
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
        print_help(); exit()
    if len(sys.argv) == 2 and sys.argv[1] in ["-v", "--version"]:
        print(SOFTWARE_VERSION); exit()
    register(information_check)

####################################################################################################
# Early Checks
####################################################################################################

def early_checks() -> None:
    if len(sys.argv) <= 2:
        early_fail("invalid low-argument invocation")
    if not RENV_FILE_PATH.is_file():
        early_fail("the upscaling runner is missing")
    if not Path(sys.argv[1]).is_file():
        early_fail("invalid input file path")
    if not Path(sys.argv[2]).parent.is_dir():
        early_fail("invalid output file path")
    if not extension(Path(sys.argv[1])) in EXTENSIONS:
        early_fail("invalid input file extension")
    if not extension(Path(sys.argv[2])) in EXTENSIONS:
        early_fail("invalid output file extension")
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

log_level : LogLevel
tile_size : int

def process_regular_options() -> None:
    global log_level
    global tile_size
    if RegularOption.log in regular_options.keys():
        values = [level.name for level in LogLevel]
        if regular_options[RegularOption.log] not in values:
            fail("invalid log level")
    if RegularOption.tile in regular_options.keys():
        try:
            n = int(regular_options[RegularOption.tile])
        except Exception as e:
            fail("tile size is not an integer", True, e)
        if n < MIN_TILE_SIZE:
            fail(f"tile size < {MIN_TILE_SIZE}")
        if n > MAX_TILE_SIZE:
            fail(f"tile size > {MAX_TILE_SIZE}")
    log_level = LogLevel(regular_options.get(RegularOption.log, DEFAULT_LOGLEVEL))
    tile_size = int(regular_options.get(RegularOption.tile, DEFAULT_TILE_SIZE))
    register(process_regular_options)

####################################################################################################
# Session Folder
####################################################################################################

def create_session_folder() -> None:
    if log_level >= LogLevel.text:
        SESSION_FOLDER_PATH.mkdir(exist_ok = True)
    register(create_session_folder)

####################################################################################################
# Exiting
####################################################################################################

exit_file_handle: TextIO

def prepare_exit_file() -> None:
    global exit_file_handle
    if log_level >= LogLevel.debug:
        exit_file_handle = safe_open(EXIT_FILE_PATH)
    register(prepare_exit_file)

def record_exit_message(success: bool, message: str) -> None:
    if log_level >= LogLevel.debug:
        outcome = 'SUCCESS' if success else 'FAILURE'
        fast_print(exit_file_handle, f"{outcome}\n\n{message}")

####################################################################################################
# Logging
####################################################################################################

log_file_handle: TextIO

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
          suggest   : bool = True                 ,
          exception : BaseException | None = None ) -> NoReturn:

    log(message, LogLevel.error)
    record_exit_message(False, message)
    early_fail(message, suggest, exception)

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
    TEMP_FOLDER_PATH.mkdir(exist_ok = True)
    atexit.register(remove_temp_folder)
    atexit.register(clean_temp_folder)

####################################################################################################
# Progress Bar Logging
####################################################################################################

bar_file_handle: TextIO

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
    scaler = Scaler(DEFAULT_CLOSURE)
    image.resize(0.5, kernel = scaler_map[scaler]).copy_memory()
    delta  = time.perf_counter() - start
    cost_  = unit_cost(Size(2000, 2000), Scale(scaler, 0.5))
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
    if loaded.bands > 4: loaded = loaded[:4]

    if bar is not None: start_unit(Size(loaded.width, loaded.height), unit, bar)

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
    copied         = image.copy()

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
    scaled         = image.resize( unit.scale                       ,
                                   vscale = vscale or unit.scale    ,
                                   kernel = scaler_map[unit.scaler] )

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
            return drop.scaler.name + str(drop.scale)
        else:
            raise ValueError
    elif isinstance(drop, DescaleData):
        return drop.comparer.name + str(drop.similarity)
    raise ValueError

def model_to_str(model: UpscaleData | None) -> str:
    if model is None:
        return DEFAULT_KEYWORD
    elif isinstance(model, UpscaleData):
        return model.name + str(model.scale) + ('-auto' if model.auto else '')
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
    s = "{" + s[:-1] + "}"
    return dacite.from_dict( data_class = UserSettings,
                             data = unflatten(json.loads(s)),
                             config = dacite.Config(check_types=True) )

def export_settings(s: UserSettings) -> str:
    result: str = ""
    for key, value in flatten(dataclasses.asdict(s)).items():
        result += f"{key} = {json.dumps(value)}\n"
    return result

def rewind_settings(s: UserSettings) -> list[str]:
    return [ s.main.format or DEFAULT_KEYWORD ,
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
# Config Arguments -> Settings
####################################################################################################

def parse_format(_: str, s: str) -> str | None:
    if s == DEFAULT_KEYWORD: return None
    if interpret_format(s) is None:
        fail("the format is invalid")
    return s

def parse_closure(name: str, s: str) -> ScaleData | None:
    if s == DEFAULT_KEYWORD: return None
    match = re.match(r"^([a-zA-Z]+)([0-9]+)$", s)
    if match is None:
        fail(f"the argument '{name}' is invalid")
    name = match.group(1)
    arg  = int(match.group(2))
    if arg < MIN_SCALE_SCALE or arg > MAX_SCALE_SCALE:
        fail(f"the argument '{name}' is invalid")
    elif name in Scaler.__members__():
        return ScaleData(Scaler[name], arg)
    fail(f"the argument '{name}' is invalid")

def parse_drop(name: str, s: str) -> SpecialDrop | ScaleData | DescaleData | None:
    if   s == DEFAULT_KEYWORD: return None
    elif s == SpecialDrop.unit.name : return SpecialDrop.unit
    elif s in [ f"{x.name}-{y.name}" for x in Scaler.__members__()
                                     for y in SpecialScale.__members__() ]:
        [algorithm, scale] = s.split("-")
        return ScaleData(Scaler[algorithm], SpecialScale[scale])
    match = re.match(r"^([a-zA-Z]+)([0-9]+)$", s)
    if match is None:
        fail(f"the argument '{name}' is invalid")
    algorithm = match.group(1)
    scale     = int(match.group(2))
    if algorithm in Scaler.__members__():
        if scale < MIN_SCALE_SCALE or scale > MAX_SCALE_SCALE:
            fail(f"the argument '{algorithm}' is invalid")
        return ScaleData(Scaler[algorithm], scale)
    elif algorithm in Comparer.__members__():
        if scale < MIN_DESCALE_TARGET or scale > MAX_DESCALE_TARGET:
            fail(f"the argument '{algorithm}' is invalid")
        return DescaleData(Comparer[algorithm], scale)
    fail(f"the argument '{algorithm}' is invalid")

def parse_model(name: str, s: str) -> UpscaleData | None:
    if s == DEFAULT_KEYWORD: return None
    match = re.match(r"^([a-zA-Z]+)([0-9]+)((\\-auto)?)$", s)
    if match is None:
        fail(f"the argument '{name}' is invalid")
    name = match.group(1)
    arg  = int(match.group(2))
    auto = bool(match.group(2))
    if arg < MIN_UPSCALE_SCALE or arg > MAX_UPSCALE_SCALE:
        fail(f"the argument '{name}' is invalid")
    if not (MODEL_FOLDER_PATH / f"{s}.bin").is_file():
        fail(f"the model {name}{arg}'s weights (.bin) are missing")
    if not (MODEL_FOLDER_PATH / f"{s}.param").is_file():
        fail(f"the model {name}{arg}'s parameters (.param) are missing")
    return UpscaleData(name, arg, auto)

def parse_cycles(name: str, s: str) -> int | None:
    if s == DEFAULT_KEYWORD: return None
    try: arg = int(s)
    except ValueError:
        fail(f"the argument '{name}' is invalid")
    if arg < MIN_CYCLES or arg > MAX_CYCLES:
        fail(f"the argument '{name}' is out of range")
    return arg

#---------------------------------------------------------------------------------------------------

def settings_from_config_arguments(config_args: list[str]) -> UserSettings:

   def feed(arg: ConfigArgument, parser):
       return parser(arg.name.replace("_", " "), config_args[arg])

   return UserSettings \
        ( UserMainSettings  ( feed(ConfigArgument.main_format    , parse_format   ) ,
                              feed(ConfigArgument.main_closure   , parse_closure  ) ) ,
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

    imported = import_settings(preset_path.read_text())
    rewind   = rewind_settings(imported)
    settings = settings_from_config_arguments(rewind)

    if interpret_format(format_) is None:
        fail("the format is invalid")
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
    override_args     = [override_options.get(arg, DEFAULT_KEYWORD) for arg in ConfigArgument]
    override_settings = settings_from_config_arguments(override_args)
    user_settings     = enrich_settings(override_settings, user_settings)

####################################################################################################
# Defaults Resolution
####################################################################################################

ground_settings: GroundSettings

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
# I/O Info Processing
####################################################################################################

input_mode      : str
input_size      : Size
input_image     : pyvips.Image
output_mode     : str
output_size     : Size
current_image   : pyvips.Image

def process_io_info() -> None:
    global input_mode
    global input_size
    global input_image
    global output_mode
    global output_size
    global current_image

    temp = pyvips.Image.new_from_file(str(input_file_path), access="sequential")

    iext        = extension(input_file_path)
    imode       = cast(str, temp.interpretation)
    ialpha      = 'alpha' if temp.hasalpha() else 'opaque'
    input_mode  = f"{iext}--{imode}--{ialpha}"
    input_size  = Size(temp.width, temp.height)
    input_image = load(Load(input_file_path)).copy_memory()

    if ialpha == 'alpha' and extension(output_file_path) in OPAQUE_EXTENSIONS:
        fail(f"the output can't carry input's alpha channel")

    size = interpret_format(ground_settings.main.format)
    if size is None: fail("the format is invalid")

    oext          = extension(output_file_path)
    omode         = "srgb"
    oalpha        = ialpha
    output_mode   = f"{oext}--{omode}--{oalpha}"
    output_size   = size
    current_image = input_image.copy_memory()

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
    session = Session ( InvocationInfo(INVOCATION_STAMP, SOFTWARE_VERSION, log_level.name) ,
                        ImageInfo(input_mode, input_size.width, input_size.height)         ,
                        ImageInfo(output_mode, output_size.width, output_size.height)      ,
                        ground_settings                                                    ,
                        ExtraInfo(tile_size)                                               )
    if log_level >= LogLevel.text:
        SESSION_FILE_PATH.write_text(export_session(session))

####################################################################################################
# Upscaling Logging
####################################################################################################

upscaling_file_handle: TextIO

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

def create_descaling_file() -> None:
    global descaling_file_handle
    if log_level >= LogLevel.debug:
        descaling_file_handle = safe_open(DESCALING_FILE_PATH)

def record_descaling_progress(line: str) -> None:
    if log_level >= LogLevel.debug:
        message = f"{timestring(datetime.now())}: {line}"
        fast_print(upscaling_file_handle, message)

####################################################################################################
# Upscaling
####################################################################################################

def upscale(unit: Upscale, bar: ProgressBar | None = None) -> None:

    size = Size(current_image.width, current_image.height)
    if bar is not None: start_unit(size, unit, bar)

    process = subprocess.Popen( [ str(RENV_FILE_PATH)              ,
                                  "-i", str(TEMP_INPUT_FILE_PATH)  ,
                                  "-o", str(TEMP_OUTPUT_FILE_PATH) ,
                                  "-m", str(MODEL_FOLDER_PATH)     ,
                                  "-n", unit.name                  ,
                                  "-t", str(64 * tile_size)        ,
                                  "-g", "0"                        ,
                                  "-j", "1:1:1"                    ,
                                  "-s", str(unit.scale)            ],
                                  stdout  = subprocess.PIPE         ,
                                  stderr  = subprocess.STDOUT       ,
                                  text    = True                    ,
                                  bufsize = 1                       )

    if process.stdout is None:
        fail("failed to capture the runner's output")

    for line in process.stdout:
        record_upscaling_progress(line)
        if bar is not None:
            x = re.search(r"^([0-9]+(\.[0-9]+)?)%$", line)
            if x is not None: bar.progress(float(x.group(1)))

    exit_code = process.wait()
    if exit_code != 0: fail(f"the runner failed with code {exit_code}" )

    if bar is not None: bar.stop()

####################################################################################################
# Descaling
####################################################################################################

def gsim(reference: numpy.ndarray, candidate: numpy.ndarray) -> float:
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

def psim(reference: numpy.ndarray, candidate: numpy.ndarray) -> float:
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
    return numpy.sum(coherence * energy) / numpy.sum(energy)

def ssim(reference: numpy.ndarray, candidate: numpy.ndarray) -> float:
    return skimage.metrics.structural_similarity(reference, candidate, data_range = 255)

#---------------------------------------------------------------------------------------------------

def sim(reference: numpy.ndarray, candidate: numpy.ndarray, comparer: Comparer) -> float:
    if   comparer == Comparer.ssim:
        return ssim(reference, candidate)
    elif comparer == Comparer.psim:
        return psim(reference, candidate)
    elif comparer == Comparer.gsim:
        return gsim(reference, candidate)
    raise ValueError

def ndarray(image: pyvips.Image) -> numpy.ndarray:
    return numpy.ndarray( buffer = image.write_to_memory()  ,
                          dtype  = numpy.uint8              ,
                          shape=(image.height, image.width) )

def roundtrip(image: pyvips.Image, div: float) -> pyvips.Image:
    forth = image.resize(1.0 / div, kernel = scaler_map[Scaler[INTERNAL_SCALER]])
    back  = forth.resize(div, kernel = scaler_map[Scaler[INTERNAL_SCALER]])
    return back

#---------------------------------------------------------------------------------------------------

def descale(unit: Descale, image: pyvips.Image, bar: ProgressBar | None = None) -> pyvips.Image:
    if bar is not None: start_unit(Size(image.width, image.height), unit, bar)
    bw = image.copy()
    bw = bw[:3] if bw.bands > 3 else bw
    bw = bw.colourspace("b-w").cast("uchar")
    ref  = ndarray(bw)
    hi_div = 2.0
    bar_n = 0
    bar_p = 0
    while ( sim(ref, ndarray(roundtrip(bw, hi_div)), unit.comparer) >= unit.similarity
            and (round(bw.width / hi_div) >= 1 or round(bw.height / hi_div) >= 1)        ):
        hi_div *= 2.0
        bar_p += 25.0 / 2 ** bar_n
        if bar is not None: bar.progress(bar_p)
        bar_n += 1
    bar_p += 25.0 / 2 ** bar_n
    if bar is not None: bar.progress(bar_p)
    lo_div = hi_div / 2.0
    div = (lo_div + hi_div) / 2.0
    for _ in range(DESCALE_ITERATIONS):
        b = sim(ref, ndarray(roundtrip(bw, div)), unit.comparer) >= unit.similarity
        lo_div = div if     b else lo_div
        hi_div = div if not b else hi_div
        div = (lo_div + hi_div) / 2.0
        if bar is not None: bar.progress(bar_p + (90.0 - bar_p) / DESCALE_ITERATIONS)
    result = image.resize(1.0 / div, kernel = scaler_map[Scaler["lanczos"]]).copy_memory()
    if bar is not None: bar.progress(100.0); bar.stop()
    return result

####################################################################################################
# Picture Logging
####################################################################################################

def log_input(size: Size, index: int, dry: bool, bar: ProgressBar | None = None) -> float:

    cost           = 0

    if log_level >= LogLevel.endpoints:
        filename = f"{index:02}_import_{current_image.width}x{current_image.height}.png"
        unit     = Save(SESSION_FOLDER_PATH / filename)
        if not dry: save(unit, current_image, bar)
        cost += unit_cost(size, unit)

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
        filename = ( f"{index:02}_"                f"{phase.name}-phase_"
                     f"{step.name}-step_"          f"{current_image.width}x"
                     f"{current_image.height}.png" )
        unit     = Save(SESSION_FOLDER_PATH / filename)
        if not dry: save(unit, current_image, bar)
        cost += unit_cost(size, unit)

    if not dry:
        article = "a" + "n" if step.name[0] in 'aeiou' else ''
        log( f"{article} {step.name} step in the {phase.name} phase has been "
             f"completed with output size {current_image.width}x{current_image.height}" )

    return cost

def log_output(size: Size, index: int, dry: bool, bar: ProgressBar | None = None) -> float:

    cost = 0

    if log_level >= LogLevel.endpoints:
        filename = f"{index:02}_export_{current_image.width}x{current_image.height}.png"
        unit = Save(SESSION_FOLDER_PATH / filename)
        if not dry: save(unit, current_image, bar)
        cost += unit_cost(size, unit)

    if log_level >= LogLevel.nothing:
        unit = Save(output_file_path)
        if not dry: save(unit, current_image, bar)
        cost += unit_cost(size, unit)

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

    cost += log_input(estimated_size, index(), dry, bar)

    for phase in Phase:

        s: GroundStageSettings = getattr(ground_settings, phase.name)

        for i in range(s.cycles):
            match s.drop:
                case SpecialDrop.unit:
                    continue
                case ScaleData():
                    match s.drop.scale:
                        case SpecialScale.root:
                            scale_ = math.sqrt(s.model.scale)
                        case SpecialScale.full:
                            scale_ = s.model.scale
                        case int():
                            scale_ = s.drop.scale / 100.0
                        case _:
                            raise ValueError
                    unit = Scale(s.drop.scaler, scale_)
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
                    if s.model.auto:
                        r = output_size.width / current_image.width
                        n = math.ceil(math.log(r, s.model.scale))
                        k = (r / s.model.scale **  n) ** (1.0 / (n - 1))
                        for _ in range(n):
                            unit = Upscale(s.model.name, s.model.scale)
                            if not dry: upscale(unit, bar)
                            cost += unit_cost(estimated_size, unit)
                            unit = Scale(Scaler[INTERNAL_SCALER], k)
                            if not dry: current_image = scale(unit, current_image, None, bar)
                            cost += unit_cost(estimated_size, unit)
                        cost += log_step(estimated_size, index(), dry, phase, Step.upscale, bar)
                        estimated_size = output_size
                    else:
                        unit = Save(TEMP_INPUT_FILE_PATH)
                        if not dry: save(unit, current_image, bar)
                        cost += unit_cost(estimated_size, unit)
                        unit = Upscale(s.model.name, s.model.scale)
                        if not dry: upscale(unit, bar)
                        cost += unit_cost(estimated_size, unit)
                        cost += log_step(estimated_size, index(), dry, phase, Step.upscale, bar)
                        estimated_size *= unit_approx_scale(unit)
                        unit = Load(TEMP_OUTPUT_FILE_PATH)
                        if not dry: current_image = load(unit, bar)
                        cost += unit_cost(estimated_size, unit)
                case _:
                    raise ValueError

    hscale = output_size.width  / current_image.width
    vscale = output_size.height / current_image.height
    unit = Scale(ground_settings.main.closure, hscale)
    if not dry: current_image = scale(unit, current_image, vscale, bar)
    cost += unit_cost(estimated_size, unit)
    cost += log_output(estimated_size, index(), dry, bar)
    estimated_size *= unit_approx_scale(unit)

    return cost

####################################################################################################
# Dry Check
####################################################################################################

def dry_check(cost: float) -> None:

    if log_level <= LogLevel.dry and Flag.quiet not in flags:
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
        print(f" tile size      : {int(regular_options[RegularOption.tile]) * 64} px")
        print(f" total work     : {cost:.2f} Mpx")
        print("")

    exit()

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
        early_fail(" └─→ Keyboard Interrupt", False, e)
    except BaseException as e:
        early_fail("Unexpected Error", False, e)

    try:
        log("information check performed", LogLevel.text, recall(information_check))
        log("early checks performed", LogLevel.text, recall(early_checks))
        log("early checks performed", LogLevel.text, recall(early_checks))
        log("arguments have been organized", LogLevel.text, recall(sort_arguments))
        log("regular options have been processed", LogLevel.text, recall(process_regular_options))
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
        load_user_settings()
        log("user settings loaded")
        resolve_overrides()
        log("overrides resolved")
        resolve_defaults()
        log("defaults resolved")
        process_io_info()
        log("io information processed")
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
        record_exit_message(False, "Keyboard Interrupt")
        fail(" └─→ Keyboard Interrupt", False, e)
    except BaseException as e:
        record_exit_message(False, traceback.format_exc())
        fail("Unexpected Error", False, e)

    try:
        cost = process(True)
        log("execution plan created")
        dry_check(cost)
        log("dry check performed")
        process(False, create_bar(cost))
        log("execution plan executed")
    except SystemExit as e:
        raise e
    except KeyboardInterrupt as e:
        if Flag.quiet not in flags: print()
        record_exit_message(False, "Keyboard Interrupt")
        fail(" └─→ Keyboard Interrupt", False, e)
    except BaseException as e:
        if Flag.quiet not in flags: print()
        record_exit_message(False, traceback.format_exc())
        fail("Unexpected Error", False, e)
    else:
        if Flag.quiet not in flags: print()
        record_exit_message(True, "")

####################################################################################################
# Help
####################################################################################################

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

####################################################################################################
# Main Call
####################################################################################################

if __name__ == "__main__":
    main()

####################################################################################################
# End
####################################################################################################
