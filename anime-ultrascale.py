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

from pathlib     import Path
from enum        import IntEnum, Enum
from datetime    import datetime
from dataclasses import dataclass, field
from threading   import Event
from typing      import NoReturn, Final, TextIO, Any, cast, Callable, Optional

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

DEFAULT_FORMAT         : Final = "1"
DEFAULT_CLOSURE        : Final = "cubic"
DEFAULT_REPAIR_DROP    : Final = "unit"
DEFAULT_REPAIR_MODEL   : Final = "unit"
DEFAULT_REPAIR_CYCLES  : Final = "1"
DEFAULT_ENHANCE_DROP   : Final = "unit"
DEFAULT_ENHANCE_MODEL  : Final = "unit"
DEFAULT_ENHANCE_CYCLES : Final = "1"
DEFAULT_STYLIZE_DROP   : Final = "unit"
DEFAULT_STYLIZE_MODEL  : Final = "unit"
DEFAULT_STYLIZE_CYCLES : Final = "1"
DEFAULT_PRESET         : Final = "quality"
DEFAULT_LOG_LEVEL      : Final = "text"
DEFAULT_TILE_SIZE      : Final = "4"

#---------------------------------------------------------------------------------------------------

MIN_SCALING_SCALE        : Final = 1
MAX_SCALING_SCALE        : Final = 100
MIN_DESCALING_SIMILARITY : Final = 80
MAX_DESCALING_SIMILARITY : Final = 100
MIN_UPSCALING_SCALE      : Final = 2
MAX_UPSCALING_SCALE      : Final = 16
MIN_CYCLES               : Final = 0
MAX_CYCLES               : Final = 4
MIN_TILE_SIZE            : Final = 1
MAX_TILE_SIZE            : Final = 16
MIN_WIDTH                : Final = 16
MIN_HEIGHT               : Final = 16

#---------------------------------------------------------------------------------------------------

OPAQUE_EXTENSIONS       : Final = ["jpg", "jpeg", "bmp"]
ALPHA_EXTENSIONS        : Final = ["png", "webp", "tif", "tiff"]
PRESET_EXTENSION        : Final = "preset"
OUTPUT_PRESET           : Final = "session"
DEFAULT_KEYWORD         : Final = "pass"
AUTO_KEYWORD            : Final = "auto"
FIX_KEYWORD             : Final = "fix"
FLEX_KEYWORD            : Final = "flex"
UNIT_KEYWORD            : Final = "unit"
PROMPT_WIDTH            : Final = 80
DEFAULT_SCALER          : Final = "lanczos"
DESCALING_APPROXIMATION : Final = 0.5
DESCALING_PRECISION     : Final = 0.01

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
    linear  = "linear"
    cubic   = "cubic"
    lanczos = "lanczos"

class Comparer(str, Enum):
    structure = "structure"
    phase     = "phase"
    gradient  = "gradient"

#---------------------------------------------------------------------------------------------------

scaler_map: Final = { Scaler.cubic : "cubic"    ,
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

@dataclass
class UnitData:
    pass

@dataclass
class DownscaleData:
    scaler     : Scaler
    scale      : int
    fixed      : bool

@dataclass
class DescaleData:
    comparer   : Comparer
    similarity : int
    scaler     : Scaler

@dataclass
class UpscaleData:
    model      : str
    scale      : int

@dataclass
class UltrascaleData:
    model      : str
    scale      : int
    scaler     : Scaler

@dataclass
class IterationData:
    count      : int

#---------------------------------------------------------------------------------------------------

@dataclass
class UserMainSettings:
    format  : Optional[str]    = None
    closure : Optional[Scaler] = None

@dataclass
class UserStageSettings:
    fall   : Optional[DownscaleData | DescaleData    | UnitData] = None
    rise   : Optional[UpscaleData   | UltrascaleData | UnitData] = None
    cycles : Optional[IterationData]                             = None

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
    fall    : DownscaleData | DescaleData    | UnitData
    rise    : UpscaleData   | UltrascaleData | UnitData
    cycles  : IterationData

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
    model : str
    scale : int

@dataclass
class Descale(Unit):
    comparer   : Comparer
    similarity : float
    scaler     : Scaler

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
        result *= unit.scale ** 2 if unit.scale != 1 else 0
    result /= 1000000
    return result

def unit_approx_scale(unit: Unit) -> float:
    if isinstance(unit, (Scale, Upscale)):
        return unit.scale
    elif isinstance(unit, Descale):
        return DESCALING_APPROXIMATION
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

        if arg in regular_options_map:
            resolver  = regular_options_map
            collector = regular_options
        elif arg in override_options_map:
            resolver  = override_options_map
            collector = override_options
        else:
            early_fail(f"unrecognized option '{arg}'")

        if i + 1 >= len(sys.argv) or sys.argv[i + 1].startswith("-"):
            early_fail(f"option '{arg}' has no value")

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
                    f"level {log_level_map[level].upper(): <6} -> "
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
        INVOCATION_FILE_PATH.write_text( f"PID       -> {INVOCATION_PID}\n"     +
                                         f"Timestamp -> {INVOCATION_STAMP}\n"   +
                                         f"PWD       -> {Path.cwd()}\n"         +
                                         f"Command   -> {' '.join(sys.argv)}\n" )

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
    image.resize(0.5, kernel = scaler_map[Scaler(DEFAULT_SCALER)]).copy_memory()
    delta  = time.perf_counter() - start
    cost_  = unit_cost(Size(2000, 2000), Scale(Scaler(DEFAULT_SCALER), 0.5))
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
        fast_print( scaling_file_handle                                         ,
                    f"{timestring(datetime.now())} -> "                         +
                    f"{{ "                                                      +
                    f"\"job\": \"{job + '",': <7} "                             +
                    f"\"event\": \"{event + '",': <10} "                        +
                    f"\"percent\": {f"{progress.percent},": >4} "               +
                    f"\"done\": {f"{progress.npels},": >9} "                    +
                    f"\"left\": {f"{progress.tpels - progress.npels}": >9}"     +
                    f" }}"                                                      )

####################################################################################################
# Loading
####################################################################################################

def load(unit: Load, bar: ProgressBar | None = None) -> pyvips.Image:

    loaded         = pyvips.Image.new_from_file(str(unit.path), access = "sequential")
    if bar is not None: start_unit(Size(loaded.width, loaded.height), unit, bar)
    sigint_handler = signal.getsignal(signal.SIGINT)
    interrupted    = Event()

    try:
        loaded         = loaded.colourspace("srgb")
        loaded         = loaded.cast("uchar")
        percentage     = -1

        if loaded.hasalpha():
            if loaded.bands > 4:
                loaded = loaded[:3].bandjoin(loaded.extract_band(loaded.bands - 1))
        elif loaded.bands > 3:
            loaded = loaded[:3]

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

        return loaded

    except pyvips.Error:
        if interrupted.is_set():
            raise KeyboardInterrupt from None
        raise

    finally:
        signal.signal(signal.SIGINT, sigint_handler)
        if bar is not None: bar.stop()


####################################################################################################
# Saving
####################################################################################################

def save(unit: Save, image: pyvips.Image, bar: ProgressBar | None = None) -> None:

    if bar is not None: start_unit(Size(image.width, image.height), unit, bar)
    sigint_handler = signal.getsignal(signal.SIGINT)
    interrupted    = Event()

    try:
        copied     = image.copy()
        percentage = -1

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
    sigint_handler = signal.getsignal(signal.SIGINT)
    interrupted = Event()

    try:
        if hscale == 1.0: return image.copy()

        scaled     = image.resize( unit.scale                                            ,
                                   vscale = vscale if vscale is not None else unit.scale ,
                                   kernel = scaler_map[unit.scaler]                      )
        percentage = -1
        def update_interrupt(image: pyvips.Image, _) -> None:
            if interrupted.is_set(): image.set_kill(True)

        def update_progress(_, progress: Any) -> None:
            nonlocal percentage
            if bar is not None and percentage != progress.percent:
                bar.progress(float(progress.percent))
                percentage = progress.percent

        scaled.set_progress(True)
        scaled.signal_connect \
            ("preeval", lambda _, progress: record_scaling_progress("scale", "preeval", progress))
        scaled.signal_connect("eval", update_interrupt)
        scaled.signal_connect("eval", update_progress)
        scaled.signal_connect \
            ("eval", lambda _, progress: record_scaling_progress("scale", "eval", progress))
        scaled.signal_connect \
            ("posteval", lambda _, progress: record_scaling_progress("scale", "posteval", progress))
        signal.signal(signal.SIGINT, lambda signum, frame: interrupted.set())

        if bar is not None: bar.progress(0.0)
        scaled = scaled.copy_memory()
        if bar is not None: bar.progress(100.0)
        if interrupted.is_set():
            raise KeyboardInterrupt

        return scaled

    except pyvips.Error:
        if interrupted.is_set():
            raise KeyboardInterrupt from None
        raise

    finally:
        signal.signal(signal.SIGINT, sigint_handler)
        if bar is not None: bar.stop()

####################################################################################################
# Nested Dictionary Un/Flattening
####################################################################################################

def flatten(data: dict[str, object]) -> dict[str, object]:
    def flatten_(data_: dict[str, object], prefix: str) -> dict[str, object]:
        result = {}
        for key, value in data_.items():
            joined_key = f"{prefix}_{key}" if prefix else key
            if isinstance(value, dict):
                if value:
                    result.update(flatten_(value, joined_key))
                else:
                    result[joined_key] = {}
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

def print_format(format_: str | None) -> str:
    if format_ is None:
        return DEFAULT_KEYWORD
    return format_

def print_closure(closure: Scaler | None) -> str:
    if closure is None:
        return DEFAULT_KEYWORD
    elif isinstance(closure, Scaler):
        return closure.name
    raise ValueError

def print_fall(fall: DownscaleData | DescaleData | UnitData | None) -> str:
    if fall is None:
        return DEFAULT_KEYWORD
    elif isinstance(fall, UnitData):
        return UNIT_KEYWORD
    elif isinstance(fall, DownscaleData):
        trail = f'-{FIX_KEYWORD if fall.fixed else FLEX_KEYWORD}'
        return f"{fall.scaler.name}{fall.scale}{trail}"
    elif isinstance(fall, DescaleData):
        trail = f'-{fall.scaler.name}' if fall.scaler else ''
        return f"{fall.comparer.name}{fall.similarity}{trail}"
    raise ValueError

def print_rise(rise: UpscaleData | UltrascaleData | UnitData | None) -> str:
    if rise is None:
        return DEFAULT_KEYWORD
    elif isinstance(rise, UnitData):
        return UNIT_KEYWORD
    elif isinstance(rise, UpscaleData):
        return f"{rise.model}{rise.scale}x"
    elif isinstance(rise, UltrascaleData):
        return f"{rise.model}{rise.scale}x-{rise.scaler.name}"
    raise ValueError

def print_cycles(cycles: IterationData | None) -> str:
    if cycles is None:
        return DEFAULT_KEYWORD
    elif isinstance(cycles, IterationData):
        return f"{cycles.count}"
    raise ValueError

#---------------------------------------------------------------------------------------------------

def import_settings(s : str) -> UserSettings:
    s = re.sub(r'^\s*(#.*)?$\n?', '', s, flags = re.MULTILINE)
    s = re.sub(r'^\s*(\w+)\s*=([^#\n]*)(#.*)?$\n?', r'"\1": \2,', s, flags=re.MULTILINE)
    s = re.sub(r",\s*$", "", s)
    s = "{" + s + "}"
    type_hooks = {Scaler: Scaler, Comparer: Comparer}
    config = dacite.Config(check_types = True, type_hooks = type_hooks)
    return dacite.from_dict( data       = unflatten(json.loads(s)) ,
                             data_class = UserSettings             ,
                             config     = config                   )

def export_settings(s: UserSettings) -> str:
    result: str = ""
    for key, value in flatten(dataclasses.asdict(s)).items():
        result += f"{key: <23} = {json.dumps(value)}\n"
    return result

def rewind_settings(s: UserSettings) -> list[str]:
    return [print_format(s.main.format)     ,
            print_closure(s.main.closure)   ,
            print_fall(s.repair.fall)       ,
            print_rise(s.repair.rise)       ,
            print_cycles(s.repair.cycles)   ,
            print_fall(s.enhance.fall)      ,
            print_rise(s.enhance.rise)      ,
            print_cycles(s.enhance.cycles)  ,
            print_fall(s.stylize.fall)      ,
            print_rise(s.stylize.rise)      ,
            print_cycles(s.stylize.cycles)]

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

def parse_fall(s: str) -> UnitData | DownscaleData | DescaleData | None:
    if s == DEFAULT_KEYWORD: return None
    elif s == UNIT_KEYWORD : return UnitData()
    match = re.match(r"^([a-zA-Z0-9-]*[a-zA-Z-])([0-9]+)(-([a-zA-Z-]+))$", s)
    if match is None: fail(f"unrecognized drop '{s}'")
    algorithm, arg1, arg2 = match.group(1), int(match.group(2)), match.group(4)
    if algorithm in Scaler.__members__ and arg2 in [FIX_KEYWORD, FLEX_KEYWORD]:
        if arg1 < MIN_SCALING_SCALE or arg1 > MAX_SCALING_SCALE:
            fail( f"scale '{arg1}' out of range "
                  f"[{MIN_SCALING_SCALE}, {MAX_SCALING_SCALE}]" )
        return DownscaleData(Scaler(algorithm), arg1, arg2 == FIX_KEYWORD)
    elif algorithm in Comparer.__members__ and arg2 in Scaler.__members__:
        if arg1 < MIN_DESCALING_SIMILARITY or arg1 > MAX_DESCALING_SIMILARITY:
            fail(f"similarity '{arg1}' out of range "
                  f"[{MIN_DESCALING_SIMILARITY}, {MAX_DESCALING_SIMILARITY}]" )
        return DescaleData(Comparer(algorithm), arg1, Scaler(arg2))
    fail(f"unrecognized drop '{s}'")

def parse_rise(s: str) -> UnitData | UpscaleData | UltrascaleData | None :
    if s == DEFAULT_KEYWORD: return None
    elif s == UNIT_KEYWORD : return UnitData()
    match = re.match(r"^([a-zA-Z0-9-]*[a-zA-Z-])([0-9]+)x(-([a-zA-Z-]+))?$", s)
    if match is None: fail(f"unrecognized model '{s}'")
    model, scale, scaler = match.group(1), int(match.group(2)), match.group(4)
    if scale < MIN_UPSCALING_SCALE or scale > MAX_UPSCALING_SCALE:
        fail( f"scale '{scale}' out of range "
              f"[{MIN_UPSCALING_SCALE}, {MAX_UPSCALING_SCALE}]" )
    if not (MODEL_FOLDER_PATH / f"{model}{scale}x.bin").is_file():
        fail(f"missing weights (.bin) of model '{model}{scale}x'")
    if not (MODEL_FOLDER_PATH / f"{model}{scale}x.param").is_file():
        fail(f"missing parameters (.param) of model '{model}{scale}x'")
    if scaler is None:
        return UpscaleData(model, scale)
    elif scaler in Scaler.__members__:
        return UltrascaleData(model, scale, Scaler(scaler))
    else:
        fail(f"unrecognized rise '{s}'")

def parse_cycles(s: str) -> IterationData | None:
    if s == DEFAULT_KEYWORD: return None
    try: cycles = int(s)
    except ValueError as e: fail(f"unrecognized cycles '{s}'", e)
    if cycles < MIN_CYCLES or cycles > MAX_CYCLES:
        fail(f"cycles '{cycles}' out of range [{MIN_CYCLES}, {MAX_CYCLES}]")
    return IterationData(cycles)

#---------------------------------------------------------------------------------------------------

def settings_from_config_arguments(args: list[str]) -> UserSettings:
   return UserSettings \
        ( UserMainSettings  ( parse_format  ( args[ConfigArgument.main_format]    ) ,
                              parse_closure ( args[ConfigArgument.main_closure]   ) ) ,
          UserStageSettings ( parse_fall    (args[ConfigArgument.repair_drop]     ) ,
                              parse_rise    (args[ConfigArgument.repair_model]    ) ,
                              parse_cycles  ( args[ConfigArgument.repair_cycles]  ) ) ,
          UserStageSettings ( parse_fall    (args[ConfigArgument.enhance_drop])   ,
                              parse_rise    (args[ConfigArgument.enhance_model])  ,
                              parse_cycles  ( args[ConfigArgument.enhance_cycles] ) ) ,
          UserStageSettings ( parse_fall    (args[ConfigArgument.stylize_drop])   ,
                              parse_rise    (args[ConfigArgument.stylize_model])  ,
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
    default_args = [ DEFAULT_FORMAT         ,
                     DEFAULT_CLOSURE        ,
                     DEFAULT_REPAIR_DROP    ,
                     DEFAULT_REPAIR_MODEL   ,
                     DEFAULT_REPAIR_CYCLES  ,
                     DEFAULT_ENHANCE_DROP   ,
                     DEFAULT_ENHANCE_MODEL  ,
                     DEFAULT_ENHANCE_CYCLES ,
                     DEFAULT_STYLIZE_DROP   ,
                     DEFAULT_STYLIZE_MODEL  ,
                     DEFAULT_STYLIZE_CYCLES ]
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
        message = f"{timestring(datetime.now())} -> {line}"
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

def record_descaling_progress(scale: float, similarity: float) -> None:
    if log_level >= LogLevel.debug:
        message = ( f"{timestring(datetime.now())} -> "  +
                    f"{{ "                               +
                    f"\"scale\": {scale:.4f}, "          +
                    f"\"similarity\" = {similarity:.4f}" +
                    f" }}"                               )
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
                                      "-n", f"{unit.model}{unit.scale}x"     ,
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
    gradient_scale      : Final = 1.0
    magnitude_exponent  : Final = 0.5
    stabilizer          : Final = 1e-6
    def gradient_magnitude(image: numpy.ndarray) -> numpy.ndarray:
        image = image.astype(numpy.float64, copy=False)
        gradient_x = scipy.ndimage.gaussian_filter(image, gradient_scale, order = (0, 1))
        gradient_y = scipy.ndimage.gaussian_filter(image, gradient_scale, order = (1, 0))
        return numpy.hypot(gradient_x, gradient_y)
    reference_magnitude = gradient_magnitude(reference) ** magnitude_exponent
    candidate_magnitude = gradient_magnitude(candidate) ** magnitude_exponent
    similarity_map = ( (2.0 * reference_magnitude * candidate_magnitude + stabilizer)     /
                       (reference_magnitude ** 2 + candidate_magnitude ** 2 + stabilizer) )
    return numpy.mean(similarity_map)

def phase_similarity(reference: numpy.ndarray, candidate: numpy.ndarray) -> float:
    gradient_scale  : Final = 1.0
    window_scale    : Final = 3.0
    stabilizer      : Final = 1e-6
    gaussian_filter : Final = scipy.ndimage.gaussian_filter
    reference = reference.astype(numpy.float64, copy = False)
    candidate = candidate.astype(numpy.float64, copy = False)
    def coefficients(image: numpy.ndarray) -> numpy.ndarray:
        gradient_x = gaussian_filter(image, gradient_scale, order = (0, 1))
        gradient_y = gaussian_filter(image, gradient_scale, order = (1, 0))
        return gradient_x + 1j * gradient_y
    reference_coefficients = coefficients(reference)
    candidate_coefficients = coefficients(candidate)
    phase_product = reference_coefficients * numpy.conj(candidate_coefficients)
    local_phase_product = ( gaussian_filter(phase_product.real, window_scale)      +
                            gaussian_filter(phase_product.imag, window_scale) * 1j )
    local_energy = gaussian_filter(numpy.abs(phase_product), window_scale)
    coherence = (numpy.abs(local_phase_product) + stabilizer) / (local_energy + stabilizer)
    total_energy = numpy.sum(local_energy)
    if total_energy == 0.0: return 1.0
    return numpy.sum(coherence * local_energy) / total_energy

def structural_similarity(reference: numpy.ndarray, candidate: numpy.ndarray) -> float:
    return skimage.metrics.structural_similarity(reference, candidate, data_range = 255)

#---------------------------------------------------------------------------------------------------

def similarity(reference: numpy.ndarray, candidate: numpy.ndarray, comparer: Comparer) -> float:
    if   comparer == Comparer.structure: return structural_similarity(reference, candidate)
    elif comparer == Comparer.gradient: return gradient_similarity(reference, candidate)
    elif comparer == Comparer.phase: return phase_similarity(reference, candidate)
    raise ValueError

def to_bytes(image: pyvips.Image) -> numpy.ndarray:
    return numpy.ndarray( buffer = image.write_to_memory()  ,
                          dtype  = numpy.uint8              ,
                          shape=(image.height, image.width) )

def roundtrip( image: pyvips.Image         ,
               scaler: Scaler              ,
               hscale: float               ,
               vscale: float | None = None ) -> pyvips.Image:
    small = image.resize( hscale                                            ,
                          vscale = vscale if vscale is not None else hscale ,
                          kernel = scaler_map[scaler]                       )
    return small.resize( image.width / small.width            ,
                         vscale = image.height / small.height ,
                         kernel = scaler_map[scaler]          )

#---------------------------------------------------------------------------------------------------

def descale(unit: Descale, image: pyvips.Image, bar: ProgressBar | None = None) -> pyvips.Image:
    if bar is not None:
        start_unit(Size(image.width, image.height), unit, bar)

    try:
        if image.width <= MIN_WIDTH or image.height <= MIN_HEIGHT:
            if bar is not None:
                bar.progress(100.0)
            return image.copy_memory()

        grayscale = image

        if grayscale.hasalpha():
            grayscale = grayscale[:-1]

        if grayscale.bands > 3:
            grayscale = grayscale[:3]

        grayscale = grayscale.colourspace("b-w").cast("uchar").copy_memory()

        scale_x = (grayscale.width  - 1.0) / grayscale.width
        scale_y = (grayscale.height - 1.0) / grayscale.height

        grayscale_r1 = roundtrip(
            grayscale,
            unit.scaler,
            scale_x,
            scale_y
        )

        reference_r1 = to_bytes(grayscale_r1)

        grayscale_r2 = roundtrip(
            grayscale_r1,
            unit.scaler,
            scale_x,
            scale_y
        )

        max_score = similarity(
            reference_r1,
            to_bytes(grayscale_r2),
            unit.comparer
        )

        min_output_scale = max(
            MIN_WIDTH  / grayscale.width,
            MIN_HEIGHT / grayscale.height
        )

        score_cache: dict[float, float] = {}

        def raw_similarity(scale: float) -> float:
            if scale not in score_cache:
                candidate = to_bytes(
                    roundtrip(grayscale_r1, unit.scaler, scale)
                )

                score_cache[scale] = similarity(
                    reference_r1,
                    candidate,
                    unit.comparer
                )

            return score_cache[scale]

        def relative_similarity(scale: float) -> float:
            return raw_similarity(scale) / max_score

        def search_from_below(
            scorer,
            target: float,
            minimum: float,
            log_divisor: float | None = None
        ) -> float:

            previous_scale = minimum
            previous_score = scorer(previous_scale)

            if log_divisor is not None:
                record_descaling_progress(
                    min(previous_scale / log_divisor, 1.0),
                    min(previous_score, 1.0)
                )

            if previous_score >= target:
                return previous_scale

            exponent = 1

            while True:
                scale = max(
                    1.0 - 1.0 / 2 ** exponent,
                    minimum
                )

                if scale <= previous_scale:
                    exponent += 1
                    continue

                score = scorer(scale)

                if log_divisor is not None:
                    record_descaling_progress(
                        min(scale / log_divisor, 1.0),
                        min(score, 1.0)
                    )

                if score >= target:
                    lower_scale = previous_scale
                    upper_scale = scale
                    break

                previous_scale = scale
                previous_score = score
                exponent += 1

            while upper_scale - lower_scale > DESCALING_PRECISION:
                scale = (
                    lower_scale + upper_scale
                ) / 2.0

                score = scorer(scale)

                if log_divisor is not None:
                    record_descaling_progress(
                        min(scale / log_divisor, 1.0),
                        min(score, 1.0)
                    )

                if score >= target:
                    upper_scale = scale
                else:
                    lower_scale = scale

            result = (
                lower_scale + upper_scale
            ) / 2.0

            if log_divisor is not None:
                result_score = scorer(result)

                record_descaling_progress(
                    min(result / log_divisor, 1.0),
                    min(result_score, 1.0)
                )

            return result

        max_scale = search_from_below(
            raw_similarity,
            max_score,
            min_output_scale
        )

        min_scale = (
            min_output_scale * max_scale
        )

        scale = search_from_below(
            relative_similarity,
            unit.similarity,
            min_scale,
            max_scale
        )

        output_scale = min(
            scale / max_scale,
            1.0
        )

        kernel = scaler_map[
            unit.scaler
        ]

        result = image.resize(
            output_scale,
            kernel=kernel
        ).copy_memory()

        if bar is not None:
            bar.progress(100.0)

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

        for _ in range(s.cycles.count):
            match s.fall:
                case UnitData():
                    pass
                case DownscaleData():
                    current_size = estimated_size if dry else get_size(current_image)
                    min_scale = max( MIN_WIDTH  / current_size.width   ,
                                     MIN_HEIGHT / current_size.height )
                    scale_ = s.fall.scale / 100.0
                    scale_ *= output_size.width / current_size.width if s.fall.fixed else 1
                    unit = Scale(s.fall.scaler, max(scale_, min_scale))
                    if not dry: current_image = scale(unit, current_image, None, bar)
                    cost += unit_cost(estimated_size, unit)
                    estimated_size *= unit_approx_scale(unit)
                    cost += log_step(estimated_size, index(), dry, phase, Step.scale, bar)
                case DescaleData():
                    similarity_ = s.fall.similarity / 100.0
                    scaler = s.fall.scaler or Scaler(DEFAULT_SCALER)
                    unit = Descale(s.fall.comparer, similarity_, scaler)
                    if not dry: current_image = descale(unit, current_image, bar)
                    cost += unit_cost(estimated_size, unit)
                    estimated_size *= unit_approx_scale(unit)
                    cost += log_step(estimated_size, index(), dry, phase, Step.descale, bar)
                case _:
                    raise ValueError
            w    = estimated_size.width if dry else current_image.width
            rise = s.rise
            if ( isinstance(s.rise, UltrascaleData)  and
                 w * s.rise.scale >= output_size.width ):
                rise = UpscaleData(s.rise.model, s.rise.scale)
            match rise:
                case UnitData():
                    pass
                case UpscaleData():
                    unit = Save(TEMP_INPUT_FILE_PATH)
                    if not dry: save(unit, current_image, bar)
                    cost += unit_cost(estimated_size, unit)
                    unit = Upscale(rise.model, rise.scale)
                    if not dry: upscale(unit, bar)
                    cost += unit_cost(estimated_size, unit)
                    estimated_size *= unit_approx_scale(unit)
                    unit = Load(TEMP_OUTPUT_FILE_PATH)
                    if not dry: current_image = load(unit, bar)
                    cost += unit_cost(estimated_size, unit)
                    cost += log_step(estimated_size, index(), dry, phase, Step.upscale, bar)
                case UltrascaleData():
                    r = output_size.width / w
                    n = math.ceil(math.log(r, rise.scale))
                    k = (r / rise.scale ** n) ** (1.0 / (n - 1))
                    for j in range(n):
                        unit = Save(TEMP_INPUT_FILE_PATH)
                        if not dry: save(unit, current_image, bar)
                        cost += unit_cost(estimated_size, unit)
                        unit = Upscale(rise.model, rise.scale)
                        if not dry: upscale(unit, bar)
                        cost += unit_cost(estimated_size, unit)
                        estimated_size *= unit_approx_scale(unit)
                        unit = Load(TEMP_OUTPUT_FILE_PATH)
                        if not dry: current_image = load(unit, bar)
                        cost += unit_cost(estimated_size, unit)
                        cost += log_step(estimated_size, index(), dry, phase, Step.upscale, bar)
                        if j < n - 1:
                            unit = Scale(rise.scaler, k)
                            if not dry: current_image = scale(unit, current_image, None, bar)
                            cost += unit_cost(estimated_size, unit)
                            estimated_size *= unit_approx_scale(unit)
                    estimated_size = output_size
                case _:
                    raise ValueError
            if isinstance(rise, UpscaleData) and isinstance(s.rise, UltrascaleData):
                k = output_size.width / (w * s.rise.scale)
                unit = Scale(Scaler(s.rise.scaler), k)
                if not dry: current_image = scale(unit, current_image, None, bar)
                cost += unit_cost(estimated_size, unit)
                estimated_size = output_size

    w = estimated_size.width  if dry else current_image.width
    h = estimated_size.height if dry else current_image.height
    hscale = output_size.width  / w
    vscale = output_size.height / h
    unit = Scale(ground_settings.main.closure, hscale)
    if not dry: current_image = scale(unit, current_image, vscale, bar)
    cost += unit_cost(estimated_size, unit)
    cost += log_output(output_size, index(), dry, bar)

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
            print(f" closure        : {print_closure(ground_settings.main.closure)}")
            print(f" repair")
            print(f"    downscaling : {print_fall(ground_settings.repair.fall)}")
            print(f"    upscaling   : {print_rise(ground_settings.repair.rise)}")
            print(f"    cycles      : {print_cycles(ground_settings.repair.cycles)}")
            print(f" enhance")
            print(f"    downscaling : {print_fall(ground_settings.enhance.fall)}")
            print(f"    upscaling   : {print_rise(ground_settings.enhance.rise)}")
            print(f"    cycles      : {print_cycles(ground_settings.enhance.cycles)}")
            print(f" stylize")
            print(f"    downscaling : {print_fall(ground_settings.stylize.fall)}")
            print(f"    upscaling   : {print_rise(ground_settings.stylize.rise)}")
            print(f"    cycles      : {print_cycles(ground_settings.stylize.cycles)}")
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
        Anime-Ultrascale
        A Tool for Extreme Anime Upscaling.

        USAGE

        (1) anime-ultrascale INPUT OUTPUT
                             FORMAT CLOSURE
                             DROP   MODEL   CYCLES
                             DROP_  MODEL_  CYCLES_
                             DROP__ MODEL__ CYCLES__
                             [OPTIONS]

        (2) anime-ultrascale INPUT OUTPUT FORMAT PRESET [OPTIONS]
            anime-ultrascale INPUT OUTPUT PRESET FORMAT [OPTIONS]

        (3) anime-ultrascale INPUT OUTPUT FORMAT [OPTIONS]
            anime-ultrascale INPUT OUTPUT PRESET [OPTIONS]

        (4) anime-ultrascale INPUT OUTPUT [OPTIONS]

        (5) anime-ultrascale {ε│-h│--help│-v│--version}

        EXAMPLES

        (1)  anime-ultrascale input.jpg output.png
                              pass pass
                              pass pass pass
                              pass pass pass
                              pass pass pass
                              --log debug

        (2a) anime-ultrascale input.jpg output.png my-preset 4k
        (2b) anime-ultrascale input.jpg output.png 4k my-preset

        (3a) anime-ultrascale input.jpg output.png my-preset
        (3b) anime-ultrascale input.jpg output.png 4k

        (4)  anime-ultrascale input.jpg output.png

        (1) Is  complete,  in  the  sense  that  it  specifies  every possible
        positional argument. It applies no transformation, saves the input and
        the  output  images,  and  logs  all  debug  information.  Use it as a
        template for your own invocation.

        (2-4) Revolve  around  'my-preset' which, when not specified, defaults
        to  the  built-in preset "quality". The argument 'format', if present,
        overrides  the preset's format. Preset files are always generated into
        the  "session" folder ("session.preset") as  long as "--log" is "text"
        or above, and can be invoked by basename if they are renamed and moved
        into the "presets" folder.

        (5) Prints  information,  where  "", "-h", and "--help" show this help
        message, while "-v" and "--version" show the program's version.

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


        CLOSURE (type: str) (auto: bicubic)
            The algorithm to be used in the final downscaling.

        DROP / DROP_ / DROP__ (type: str)
            The algorithm to be used in the preliminary downscaling of the
            repair / enhance / stylize phase.

        REDUCTION (type: float) (auto: automatic upscaling inversion)
            The divisor of upscaling inversion.

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
                'dry'       -> nothing (changes the output to infos)
                'nothing'   -> nothing
                'text'      -> basic textual data, preset included
                'endpoints' -> as 'text'      + input/output images
                'debug'     -> as 'endpoints' + debug textual data
                'research'  -> as 'debug'     + intermediate images

        {-q│--quiet}
            No standard output.
             TILING (type: int) (auto: 4)
            The  size  of each tile, to be multiplied with 64 px. For example,
            4 leads to a tile size of 256 px.

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

        Anime Ultrascale - Copyright (c) 2026 Michele Bizzoca
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
