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
import wcwidth

#---------------------------------------------------------------------------------------------------

from pathlib import Path
from enum import IntEnum, Enum
from datetime import datetime
from dataclasses import dataclass
from threading import Event, Thread, Lock
from typing import NoReturn, TypeAlias, Final, TextIO, Any, cast, NamedTuple

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
PRESET_FILE        : Final = "preset.preset"
LOG_FILE           : Final = "log.txt"
INVOCATION_FILE    : Final = "invocation.txt"
SCALING_FILE       : Final = "scaling.txt"
ENHANCEMENT_FILE   : Final = "enhancement.txt"
PROGRESS_FILE      : Final = "progress.txt"
EXIT_FILE          : Final = "exit.txt"
TEMP_INPUT_FILE    : Final = "input.png"
TEMP_OUTPUT_FILE   : Final = "output.png"
DEMO_FILE          : Final = "quality.preset"

#---------------------------------------------------------------------------------------------------

DEFAULT_PRESET     : Final = "quality"
DEFAULT_LOGLEVEL   : Final = "text"
DEFAULT_CLOSURE    : Final = "bicubic"
DEFAULT_TILING     : Final = 4
DEFAULT_ENHANCER   : Final = "bicubic"
DEFAULT_ENHANCER_  : Final = "lanczos"

#---------------------------------------------------------------------------------------------------

OUTPUT_MODE        : Final = "png--srgb--alpha"
ALPHA_EXTENSIONS   : Final = ["png", "webp", "tif", "tiff"]
OPAQUE_EXTENSIONS  : Final = ["jpg", "jpeg", "bmp"]

#---------------------------------------------------------------------------------------------------

MAX_MPX            : Final = 300
MAX_TILING         : Final = 16
MAX_ITERATIONS     : Final = 16

#---------------------------------------------------------------------------------------------------

PROMPT_WIDTH       : Final = 80
DESCALE_SIMILARITY : Final = 0.95
DESCALE_ITERATIONS : Final = 8

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
PRESET_FILE_PATH       : Final = SESSION_FOLDER_PATH / PRESET_FILE
SCALING_FILE_PATH      : Final = SESSION_FOLDER_PATH / SCALING_FILE
ENHANCING_FILE_PATH    : Final = SESSION_FOLDER_PATH / ENHANCEMENT_FILE
PROGRESS_BAR_FILE_PATH : Final = SESSION_FOLDER_PATH / PROGRESS_FILE
EXIT_FILE_PATH         : Final = SESSION_FOLDER_PATH / EXIT_FILE
TEMP_INPUT_FILE_PATH   : Final = TEMP_FOLDER_PATH    / TEMP_INPUT_FILE
TEMP_OUTPUT_FILE_PATH  : Final = TEMP_FOLDER_PATH    / TEMP_OUTPUT_FILE
RENV_FILE_PATH         : Final = RENV_FOLDER_PATH    / RENV_FILE
DEMO_FILE_PATH         : Final = PRESET_FOLDER_PATH  / DEMO_FILE

####################################################################################################
# Shorthands
####################################################################################################

EXTENSIONS: Final = ALPHA_EXTENSIONS + OPAQUE_EXTENSIONS

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

def loglevel_descriptor(loglevel: LogLevel):
    if   loglevel == LogLevel.error: return "error"
    elif loglevel == LogLevel.debug: return "debug"
    else:                            return "normal"

####################################################################################################
# Scalers and Enhancers
####################################################################################################

class Scaler(IntEnum):

    bilinear = 0
    bicubic  = 1
    lanczos  = 2

class Enhancer(IntEnum):

    conservative = 0
    strong = 1

#---------------------------------------------------------------------------------------------------

def scaler_descriptor(scaler: Scaler):
    if   scaler == Scaler.bilinear: return "linear"
    elif scaler == Scaler.bicubic:  return "cubic"
    else:                           return "lanczos3"

def enhancer_descriptor(enhancer: Enhancer):
    if enhancer == Enhancer.conservative : return "soft"
    else:                                : return "hard"

####################################################################################################
# Arguments
####################################################################################################

class BasicArgument(IntEnum):

    program_path = 0
    input_path   = 1
    output_path  = 2

class ConfigArgument(IntEnum):
    main_format     = 0
    main_reduction  = 1
    main_closure    = 2
    main_tiling     = 3
    soft_enhancer   = 4
    soft_iterations = 5
    soft_multiplier = 6
    soft_divisor    = 7
    soft_scaler     = 8
    hard_enhancer   = 9
    hard_iterations = 10
    hard_multiplier = 11
    hard_divisor    = 12
    hard_scaler     = 13

class QuickArgument(IntEnum):
    format_or_preset_a = 0
    format_or_preset_b = 0

class RegularOption(Enum):
    log = 0

class Flag(Enum):
    quiet = 0

#---------------------------------------------------------------------------------------------------

def option_string(option: str, override: bool = True) -> str:
    if override:
        [x, y] = option.split("_")
        return "--" + (y[0].upper() if x == "hard" else y[0]) + y[1:]
    else:
        return "--" + option

override_options_map = ( { option_string(x.name)             : x for x in list(ConfigArgument) } |
                         { option_string(x.name)[1:3]        : x for x in list(ConfigArgument) } )

regular_options_map  = ( { option_string(x.name, False)      : x for x in list(RegularOption) } |
                         { option_string(x.name, False)[1:3] : x for x in list(RegularOption) } )

flags_map            = ( { option_string(x.name, False)      : x for x in list(Flag)          } |
                         { option_string(x.name, False)[1:3] : x for x in list(Flag)          } )

####################################################################################################
# Sessions
####################################################################################################

@dataclass
class Invocation:

    time      : str
    version   : str
    savelevel : str

@dataclass
class ImageInfo:

    mode   : str
    width  : int
    height : int

@dataclass
class MainSettings:

    format    : str
    reduction : float | None
    closure   : str   | None
    tiling    : int   | None

@dataclass
class ModelSettings:

    enhancer   : str
    iterations : int
    multiplier : int   | None
    divisor    : float | None
    scaler     : str   | None

@dataclass
class Settings:

    main    : MainSettings
    soft    : ModelSettings
    hard    : ModelSettings

@dataclass
class Session:

    invocation : Invocation
    input      : ImageInfo
    output     : ImageInfo
    settings   : Settings

####################################################################################################
# Work Units
####################################################################################################

class Size(NamedTuple):
    width  : int
    height : int

@dataclass
class Unit:
    def __new__(cls, *args, **kwargs):
        if cls is Unit:
            raise TypeError
        return super().__new__(cls)

@dataclass
class Scale(Unit):
    scaler      : Scaler
    input_size  : Size
    output_size : Size

@dataclass
class Enhance(Unit):
    enhancer    : Enhancer
    input_size  : Size
    output_size : Size

@dataclass
class SoftEnhance(Unit):
    input_size  : Size
    output_size : Size

@dataclass
class HardEnhance(Unit):
    multiplier  : int
    input_size  : Size
    output_size : Size

@dataclass
class Save(Unit):
    size : Size

@dataclass
class Load(Unit):
    size : Size

@dataclass
class StepForward(Unit):
    save : bool
    size : Size

@dataclass
class PhaseForward(Unit):
    pass

#---------------------------------------------------------------------------------------------------

def unit_cost(unit: Unit) -> float:

    if   isinstance(unit, Scale) and not unit.input_size == unit.output_size:
        px = unit.output_size.width * unit.output_size.height * 0.10
    elif isinstance(unit, Enhance):
        px = unit.input_size.width * unit.input_size.height * 1.35
    elif isinstance(unit, SoftEnhance):
        px = unit.input_size.width * unit.input_size.height * 1.00
    elif isinstance(unit, HardEnhance):
        px = unit.input_size.width * unit.input_size.height * 1.00
    elif isinstance(unit, Save):
        px = unit.size.width * unit.size.height * 0.30
    elif isinstance(unit, Load):
        px = unit.size.width * unit.size.height * 0.05
    elif isinstance(unit, StepForward) and unit.save:
        px =  unit.size.width * unit.size.height * 0.30
    else:
        px = 0

    return px / 1000000

def unit_breakpoints(unit_class: type[Unit]) -> tuple[float, float]:

    begin = 10
    end   = 30 if unit_class in [SoftEnhance, HardEnhance, Enhance] else 10
    return begin, end

####################################################################################################
# Time Operations
####################################################################################################

def timestring(timestamp: datetime):
    return timestamp.strftime('on %Y/%m/%d at %H:%M:%S and %f')

#---------------------------------------------------------------------------------------------------

call_timestamps: dict[object, datetime]

def register(function: object):
    global call_timestamps
    call_timestamps[function] = datetime.now()

def recall(function: object) -> str:
    return timestring(call_timestamps[function])

def now() -> str:
    return timestring(datetime.now())

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
    if not RENV_FILE_PATH.is_file(): early_fail("missing Real ESRGAN runner")
    if not Path(sys.argv[1]).is_file(): early_fail("invalid input file path")
    if not Path(sys.argv[2]).parent.is_dir(): early_fail("invalid output file path")
    if not Path(sys.argv[1]).suffix[1:] in EXTENSIONS: early_fail("invalid input file extension")
    if not Path(sys.argv[2]).suffix[1:] in EXTENSIONS: early_fail("invalid output file extension")
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

        if i + 1 >= len(sys.argv) or sys.argv[i + 1].startswith["-"]:
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
# Log Level
####################################################################################################

loglevel: LogLevel

def compute_loglevel() -> None:
    global loglevel
    loglevel = LogLevel(regular_options.get(RegularOption.log, DEFAULT_LOGLEVEL))
    register(compute_loglevel)

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
# Enhancing Logging
####################################################################################################

enhancing_file_handle: TextIO

def create_enhancing_file() -> None:
    global enhancing_file_handle
    if loglevel >= LogLevel.debug:
        enhancing_file_handle = safe_open(ENHANCING_FILE_PATH)

####################################################################################################
# Progress Bar
####################################################################################################

class ProgressBar:

    def __init__(self, cost: float, mpxs: float, jumps: dict[int, tuple[float, float]]) -> None:
        pass

    def __enter__(self) -> "ProgressBar":
        pass

    def __exit__(self, *args) -> None:
        pass

    def __del__(self) -> None:
        pass

    def start(self, kind: int, cost: float) -> None:
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
    kind = Unit.__subclasses__().index(unit.__class__())
    cost = unit_cost(unit)
    bar.start(kind, cost)

def create_bar(cost: float) -> ProgressBar:

    data  = numpy.random.bytes(2000 * 2000 * 3)
    image = pyvips.Image.new_from_memory(data, 2000, 2000, 3, "uchar")
    start = time.perf_counter()
    image.resize(0.5, kernel = scaler_descriptor(Scaler.bicubic)).copy_memory()
    delta = time.perf_counter() - start
    cost_ = unit_cost(Scale(Scaler.bicubic, Size(2000, 2000), Size(1000, 1000)))
    mpxs  = cost_ / delta
    jumps = { i: unit_breakpoints(unit_class)
                 for i, unit_class in enumerate(Unit.__subclasses__()) }
    return ProgressBar(cost, mpxs, jumps)

####################################################################################################
# Scaling / Saving / Loading
####################################################################################################

def load(unit: Load, path: Path, bar: ProgressBar | None = None) -> pyvips.Image:

    if bar is not None:
        start_unit(unit, bar)

    image = pyvips.Image.new_from_file(str(path), access = "sequential")
    image = image.colourspace("srgb")
    if image.bands > 4: image = image[:4]
    image = image.cast("uchar")

    interrupted = Event()
    sigint_handler = signal.getsignal(signal.SIGINT)

    def update_interrupt(image: pyvips.Image, progress: Any) -> None:
        if interrupted.is_set(): image.set_kill(True)

    def update_progress(image: pyvips.Image, progress: Any) -> None:
        if bar is not None: bar.progress(float(progress.percent))

    image.set_progress(True)

    image.signal_connect \
        ("preeval", lambda image, progress: record_scaling_progress("load", "preeval", progress))

    image.signal_connect("eval", update_interrupt)

    image.signal_connect("eval", update_progress)

    image.signal_connect \
        ("eval", lambda image, progress: record_scaling_progress("load", "eval", progress))

    image.signal_connect \
        ("posteval", lambda image, progress: record_scaling_progress("load", "posteval", progress))

    signal.signal(signal.SIGINT, lambda signum, frame: interrupted.set())

    try:
        if bar is not None: bar.progress(0.0)
        image = image.copy_memory()
        if bar is not None: bar.progress(100.0)
        if interrupted.is_set(): raise KeyboardInterrupt

    except pyvips.Error:
        if interrupted.is_set():
            raise KeyboardInterrupt from None
        raise

    finally:
        signal.signal(signal.SIGINT, sigint_handler)

    return image

def save(unit: Save, image: pyvips.Image, path: Path, bar: ProgressBar) -> None:

    bar.new_unit(unit)

    image           = image.copy()
    last_percentage = 0.0
    started         = False
    interrupted     = Event()
    previous_sigint = signal.getsignal(signal.SIGINT)

    def request_interrupt(signum, frame) -> None:

        interrupted.set()

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

        if interrupted.is_set():
            image.set_kill(True)

    image.set_progress(True)
    image.signal_connect("preeval", start_bar)
    image.signal_connect("eval", update_bar)
    image.signal_connect("preeval",
        lambda image, progress: log_scaling("save", "preeval", progress))
    image.signal_connect("eval",
         lambda image, progress: log_scaling("save", "eval", progress))
    image.signal_connect("posteval",
         lambda image, progress: log_scaling("save", "posteval", progress))

    signal.signal(signal.SIGINT, request_interrupt)

    kwargs = {}

    if str(path).endswith("webp"):
        kwargs["lossless"] = True
        kwargs["effort"] = 4

    image.write_to_file(str(path), )
    try:
        image.write_to_file(str(path), **kwargs)

        if interrupted.is_set():
            raise KeyboardInterrupt

    except pyvips.Error:
        if interrupted.is_set():
            raise KeyboardInterrupt from None
        raise

    finally:
        signal.signal(signal.SIGINT, previous_sigint)

def scale(unit: Scale, bar: ProgressBar) -> None:

    global current_image

    if unit.out_width == unit.in_width and unit.out_height == unit.in_height:
        return

    bar.new_unit(unit)

    kernel          = scaler_descriptor(unit.scaler)
    hscale          = unit.out_width / unit.in_width
    vscale          = unit.out_height / unit.in_height
    image           = current_image.resize(hscale, vscale = vscale, kernel = kernel)
    last_percentage = 0.0
    started         = False
    interrupted     = Event()
    previous_sigint = signal.getsignal(signal.SIGINT)

    def request_interrupt(signum, frame) -> None:

        interrupted.set()

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

        if interrupted.is_set():
            image.set_kill(True)

    image.set_progress(True)
    image.signal_connect("preeval", start_bar)
    image.signal_connect("eval", update_bar)
    image.signal_connect("preeval",
        lambda image, progress: log_scaling("scale", "preeval", progress))
    image.signal_connect("eval",
        lambda image, progress: log_scaling("scale", "eval", progress))
    image.signal_connect("posteval",
        lambda image, progress: log_scaling("scale", "posteval", progress))

    signal.signal(signal.SIGINT, request_interrupt)

    try:
        scaled_image = image.copy_memory()

        if interrupted.is_set():
            raise KeyboardInterrupt

    except pyvips.Error:
        if interrupted.is_set():
            raise KeyboardInterrupt from None
        raise

    finally:
        signal.signal(signal.SIGINT, previous_sigint)

    current_image = scaled_image

def enhance(unit: Enhance, bar: ProgressBar) -> None:

    global current_image

    save_unit    = Save(unit.in_width, unit.in_height)
    pure_ai_unit = Realesrgan(** vars(unit))
    load_unit    = Load(unit.out_width, unit.out_height)

    save(save_unit, current_image, TEMP_INPUT_FILE_PATH, bar)

    bar.new_unit(pure_ai_unit)

    process = subprocess.Popen(

        [ str(RENV_RUNNER_PATH)                ,
          "-i", str(TEMP_INPUT_FILE_PATH)      ,
          "-o", str(TEMP_OUTPUT_FILE_PATH)     ,
          "-m", str(MODEL_FOLDER_PATH)         ,
          "-n", unit.enhancer                  ,
          "-t", str(64 * settings.main.tiling) ,
          "-g", "0"                            ,
          "-j", "1:1:1"                        ,
          "-s", str(unit.multiplier)           ],

        stdout  = subprocess.PIPE   ,
        stderr  = subprocess.STDOUT ,
        text    = True              ,
        bufsize = 1
    )

    if process.stdout is None:
        fail("failed to capture Real ESRGAN's output")

    for line in process.stdout:
        if savelevel() >= SaveLevel.debug:
            scaling_ai_file_handle.write(timestring(datetime.now()) + ": " + line)
            scaling_ai_file_handle.flush()
        line = "".join(line.split())
        if re.search(r"^[0-9]+(\.[0-9]+)?%$", line):
            bar.progress(float(line[:-1]))

    return_code = process.wait()
    assume( return_code == 0                              ,
            f"Real ESRGAN failed with code {return_code}" )

    current_image = load(load_unit, TEMP_OUTPUT_FILE_PATH, bar)

####################################################################################################
# Input Loading
####################################################################################################

input_image   : pyvips.Image
input_mode    : str
input_size    : Size
output_image  : pyvips.Image
output_mode   : str

def load_input() -> None:

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

    oformat = str(output_file_path.suffix.lower()[1:])
    omode   = "srgb"
    oalpha  = 'alpha' if temp.hasalpha() else 'opaque'
    output_mode = f"{oformat}--{omode}--{oalpha}"
    output_image = input_image.copy_memory()

####################################################################################################
# Image Descaling
####################################################################################################

def luminance(image: pyvips.Image) -> pyvips.Image:
    image = image[:3] if image.bands > 3 else image

    if image.bands == 1:
        return image.cast("uchar")

    return image.colourspace("b-w").cast("uchar")


def to_numpy(image: pyvips.Image) -> numpy.ndarray:
    return numpy.ndarray(
        buffer=image.write_to_memory(),
        dtype=numpy.uint8,
        shape=(image.height, image.width),
    )

def roundtrip(image: pyvips.Image, factor: float) -> pyvips.Image:
    width = max(1, round(image.width / factor))
    height = max(1, round(image.height / factor))

    reduced = image.resize(
        width / image.width,
        vscale=height / image.height,
        kernel="lanczos3",
    )

    return reduced.resize(
        image.width / reduced.width,
        vscale=image.height / reduced.height,
        kernel="lanczos3",
    )

def similarity(
    reference: numpy.ndarray,
    image: pyvips.Image,
    factor: float,
) -> float:
    reconstructed = to_numpy(roundtrip(image, factor))

    return float(
        skimage.metrics.structural_similarity(
            reference,
            reconstructed,
            data_range=255,
        )
    )

def descale(image: pyvips.Image) -> float:
    img = image.copy()
    img = luminance(img)
    reference = to_numpy(img)

    low = 1.0
    high = 2.0

    while similarity(reference, img, high) >= DESCALE_TARGET:
        low = high
        high *= 2.0

        if (
            round(img.width / high) <= 1
            or round(img.height / high) <= 1
        ):
            break

    for _ in range(DESCALE_ITERATIONS):
        middle = (low + high) / 2.0

        if similarity(reference, img, middle) >= DESCALE_TARGET:
            low = middle
        else:
            high = middle

    return (low + high) / 2.0

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
    s = re.sub(r'^\s*(\w+)\s*=([^#\n]*)(#.*)?$\n?', r'"\1": \2,', s, flags=re.MULTILINE)
    s = "{" + s[:-1] + "}"

    return dacite.from_dict( data_class = Settings,
                             data = unflatten(json.loads(s)),
                             config = dacite.Config(check_types=True) )

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
# Format Validation and Interpretation
########################################################################################

def validate_format(s:str) -> bool:
    regex = ( "[wh][0-9+]|[0-9]+%|[0-9]+"
             "(\\.[0-9]+)?|[0-9]+(k|kh|kv|K|KH|KV)" )
    return re.fullmatch(regex, s) is not None

def interpret_format(s: str) -> tuple[int, int] | None:

    def parse_k(s: str, horizontal: bool) -> tuple[int, int]:
        f = int(s[:-1])
        k1 = (960.0 if horizontal else 540.0) * f / input_width()
        k2 = (540.0 if horizontal else 960.0) * f / input_height()
        k = min(k1, k2) if s[-1:] == "k" else max(k1, k2)
        w = round(input_width() * k)
        h = round(input_height() * k)
        return w, h

    if re.fullmatch("w[0-9]+", s):
        w = int(s[1:])
        h = round((float(w) / input_width()) * input_height())
    elif re.fullmatch("h[0-9]+", s):
        h = int(s[1:])
        w = round((float(h) / input_height()) * input_width())
    elif re.fullmatch("[0-9]+%", s):
        k = float(s[:-1]) / 100
        w = round(input_width() * k)
        h = round(input_height() * k)
    elif re.fullmatch("[0-9]+(\\.[0-9]+)?", s):
        k = float(s)
        w = round(input_width() * k)
        h = round(input_height() * k)
    elif re.fullmatch("[0-9]+[kK]", s):
        w, h = parse_k(s, input_width() > input_height())
    elif re.fullmatch("[0-9]+(kh|kv|KH|KV)?", s):
        w, h = parse_k(s[:-1], s[-1:] in "hH")
    else:
        return None

    return w, h

########################################################################################
# 0 / 1 / 2 Options -> Settings
########################################################################################

def settings_from_format_and_presets(o1: str, o2: str) -> Settings:

    b1 = validate_format(o1)
    b2 = validate_format(o2)

    assume(not (not b1 and not b2), "unrecognized format")
    assume(not (b1 and b2), "format specified twice")

    format_ = o1 if b1 else o2
    preset  = o2 if b1 else o1

    if preset.endswith(".preset"):
        preset_path = Path(preset)
        assume( preset_path.is_file()             ,
                "the preset file does not exists" )
    else:
        preset_path = PRESET_FOLDER_PATH / (preset + ".preset")
        assume( preset_path.is_file()                 ,
                "the specified preset is unavailable" )

    settings = import_settings(preset_path.read_text())
    settings.main.format = format_

    return settings

def settings_from_zero_options() -> Settings:

   o1 = DEFAULT_MAIN_FORMAT
   o2 = AUTO_PRESET_FILE.split('.')[0]

   return settings_from_format_and_presets(o1, o2)

def settings_from_one_option() -> Settings:

    o1 = positional_options[OneOption.format_or_preset]
    o2 = ( DEFAULT_MAIN_FORMAT
              if interpret_format(o1) is None
              else AUTO_PRESET_FILE.split('.')[0] )

    return settings_from_format_and_presets(o1, o2)

def settings_from_two_options() -> Settings:

   o1 = positional_options[TwoOptions.format_or_preset_1]
   o2 = positional_options[TwoOptions.format_or_preset_2]

   return settings_from_format_and_presets(o1, o2)

########################################################################################
# Basic / Full Options -> Settings
########################################################################################

def parse_int(name: str, s: str) -> int:
    try: return int(s)
    except BaseException as e:
        fail(f"the argument '{name}' is not an integer", True, e)

def parse_float(name: str, s: str) -> float:
    try: return float(s)
    except BaseException as e:
        fail(f"the argument '{name}' is not a real", True, e)

def parse_str(name: str, s: str) -> str:
    return s

def with_auto(parser):
    def f(name: str, s: str):
        return None if s == "auto" else parser(name, s)
    return f

def settings_from_basic_options() -> Settings:

    return Settings (

        MainSettings(
            parse_str(FullOptions.main_format),
            None,
            None,
            None
        ),

        ModelSettings(
            parse_str(FullOptions.soft_enhancer),
            parse_int(FullOptions.soft_iterations),
            None,
            None,
            None
        ),

        ModelSettings(
            parse_str(FullOptions.hard_enhancer),
            parse_int(FullOptions.hard_iterations),
            None,
            None,
            None
        )
    )

def settings_from_full_options() -> Settings:

   def feed(index: FullOptions, parser):
       return parser(index.name.replace("_", " "), positional_options[index])

   return Settings (

        MainSettings(
            feed(FullOptions.main_format, parse_str),
            feed(FullOptions.main_reduction, with_auto(parse_int)),
            feed(FullOptions.main_closure, with_auto(parse_str)),
            feed(FullOptions.main_tiling, with_auto(parse_int))
        ),

        ModelSettings(
            feed(FullOptions.soft_enhancer, parse_str),
            feed(FullOptions.soft_iterations, parse_int),
            feed(FullOptions.soft_multiplier, with_auto(parse_int)),
            feed(FullOptions.soft_divisor, with_auto(parse_float)),
            feed(FullOptions.soft_scaler, with_auto(parse_str))
        ),

        ModelSettings(
            feed(FullOptions.hard_enhancer, parse_str),
            feed(FullOptions.hard_iterations, parse_int),
            feed(FullOptions.hard_multiplier, with_auto(parse_int)),
            feed(FullOptions.hard_divisor, with_auto(parse_float)),
            feed(FullOptions.hard_scaler, with_auto(parse_str))
        )
    )

########################################################################################
# Options -> Settings
########################################################################################

settings: Settings

def load_settings() -> None:

    global settings

    if len(positional_options) == len(ZeroOptions):
        settings = settings_from_zero_options()
    elif len(positional_options) == len(OneOption):
        settings = settings_from_one_option()
    elif len(positional_options) == len(TwoOptions):
        settings = settings_from_two_options()
    elif len(positional_options) == len(BasicOptions):
        settings = settings_from_basic_options()
    elif len(positional_options) == len(OneOption):
        settings = settings_from_full_options()
    else:
        fail( "incorrect parameter count")

########################################################################################
# Settings Recording
########################################################################################

def create_presets_file() -> None:

    if savelevel() >= SaveLevel.text:
        PRESET_FILE_PATH.write_text(export_settings(settings))

########################################################################################
# Multiplier Deduction
########################################################################################

def deduce_multiplier(enhancer: str,  hardness: str) -> int:

    m = re.search("([2-8])[xX]", enhancer) or re.search("[xX]([2-8])", enhancer)

    if m is not None:
        return int(m.group(1))
    else:
        fail(f"cannot deduce {hardness} multiplier")

########################################################################################
# Defaults Resolution
########################################################################################

def resolve_defaults() -> None:

    if settings.main.reduction is None:
        settings.main.reduction = descale(input_image)
    if settings.main.closure is None:
        settings.main.closure = DEFAULT_MAIN_CLOSURE
    if settings.main.tiling is None:
        settings.main.tiling = DEFAULT_MAIN_TILING
    if settings.soft.multiplier is None:
        settings.soft.multiplier = deduce_multiplier(settings.soft.enhancer, "soft")
    if settings.soft.divisor is None:
        settings.soft.divisor = math.sqrt(cast(int, settings.soft.multiplier))
    if settings.soft.scaler is None:
        settings.soft.scaler = "bicubic"
    if settings.hard.multiplier is None:
        settings.hard.multiplier = deduce_multiplier(settings.hard.enhancer, "hard")
    if settings.hard.divisor is None:
        settings.hard.divisor = math.sqrt(cast(int, settings.hard.multiplier))
    if settings.hard.scaler is None:
        settings.hard.scaler = "lanczos"

########################################################################################
# Overrides Resolution
########################################################################################

def resolve_overrides() -> None:

    def feed(index: FullOptions, parser):
        x = full_options.get(index, None)
        if x is None: return None
        return parser(index.name.replace("_", " "), x)

    settings.main.format = ( feed(FullOptions.main_format, parse_str) or
                             settings.main.format                      )

    settings.main.reduction = ( feed(FullOptions.main_reduction,with_auto(parse_int)) or
                                settings.main.reduction                                )

    settings.main.closure = ( feed(FullOptions.main_closure, with_auto(parse_str)) or
                              settings.main.closure                                 )

    settings.main.tiling = ( feed(FullOptions.main_tiling, with_auto(parse_int)) or
                             settings.main.tiling                                 )

    settings.soft.enhancer = ( feed(FullOptions.soft_enhancer, parse_str) or
                               settings.soft.enhancer                      )

    settings.soft.iterations = ( feed(FullOptions.soft_iterations, parse_int) or
                                 settings.soft.iterations                      )

    settings.soft.multiplier = (feed(FullOptions.soft_multiplier,with_auto(parse_int))or
                                 settings.soft.multiplier                              )

    settings.soft.divisor = ( feed(FullOptions.soft_divisor, with_auto(parse_float)) or
                              settings.soft.divisor                                   )

    settings.soft.scaler = ( feed(FullOptions.soft_scaler, with_auto(parse_str)) or
                             settings.soft.scaler                                 )

    settings.hard.enhancer = ( feed(FullOptions.hard_enhancer, parse_str) or
                               settings.hard.enhancer                      )

    settings.hard.iterations = ( feed(FullOptions.hard_iterations, parse_int) or
                                 settings.hard.iterations                      )

    settings.hard.multiplier = (feed(FullOptions.hard_multiplier,with_auto(parse_int))or
                                 settings.hard.multiplier                              )

    settings.hard.divisor = ( feed(FullOptions.hard_divisor, with_auto(parse_float)) or
                              settings.hard.divisor                                   )

    settings.hard.scaler = ( feed(FullOptions.hard_scaler, with_auto(parse_str)) or
                             settings.hard.scaler                                 )

########################################################################################
# Dimensions Computation
########################################################################################

output_width    : int
output_height   : int
main_multiplier : float

def compute_dimensions() -> None:

    global output_width
    global output_height
    global main_multiplier

    f = interpret_format(settings.main.format)

    if f is None:
        fail("unrecognized format")

    output_width = f[0]
    output_height = f[1]
    main_multiplier = output_width / input_width()

########################################################################################
# Session Construction
########################################################################################

session: Session

def create_session() -> None:

    global session

    stamp = f"{INVOCATION_DATE}--{INVOCATION_TIME}--{INVOCATION_USEC}"

    session = Session (

        Invocation(stamp, SOFTWARE_VERSION, savelevel().name)   ,
        ImageInfo(input_mode, input_width(), input_height())  ,
        ImageInfo(output_file_path.suffix.lower()[1:] , output_width, output_height) ,
        settings
    )

########################################################################################
# Session Recording
########################################################################################

def create_session_file() -> None:

    if savelevel() >= SaveLevel.text:

        with open(SESSION_FILE_PATH, "w") as session_handle:
            session_handle.write(export_session(session))

########################################################################################
# Disjoint Settings Validation
########################################################################################

def disjoint_settings_validation() -> None:

    assume( settings.main.reduction  >= 1.0  , "main reduction < 1       " )
    assume( settings.soft.multiplier >= 2    , "soft-phase multiplier < 2" )
    assume( settings.soft.divisor    >= 1.0  , "soft-phase divisor < 1"    )
    assume( settings.soft.iterations >= 0    , "soft-phase iterations < 0" )
    assume( settings.hard.multiplier >= 2    , "hard-phase multiplier < 2" )
    assume( settings.hard.divisor    >= 1.0  , "hard-phase divisor < 1"    )
    assume( settings.hard.iterations >= 0    , "hard-phase iterations < 0" )

    assume( settings.soft.iterations <= MAX_ITERATIONS ,
            f"soft-phase iterations > {MAX_ITERATIONS}" )

    assume( settings.hard.iterations <= MAX_ITERATIONS ,
            f"hard-phase iterations > {MAX_ITERATIONS}" )

    assume( (MODEL_FOLDER_PATH/(settings.soft.enhancer + ".bin")).is_file()      ,
            "missing soft-phase model weights (.bin)"                            )
    assume( (MODEL_FOLDER_PATH / (settings.soft.enhancer + ".param")).is_file()  ,
            "missing soft-phase model parameters (.param)"                       )

    assume( (MODEL_FOLDER_PATH / (settings.hard.enhancer + ".bin")).is_file()    ,
            "missing hard-phase model weights (.bin)"                            )
    assume( (MODEL_FOLDER_PATH / (settings.hard.enhancer + ".param")).is_file()  ,
            "missing hard-phase model parameters (.param)"                       )

    assume ( settings.main.closure in Scaler.__members__ ,
            "unknown scaling algorithm"                   )

    assume ( settings.soft.scaler in Scaler.__members__   ,
            "unknown scaling algorithm"                   )

    assume ( settings.hard.scaler in Scaler.__members__   ,
            "unknown scaling algorithm"                   )

########################################################################################
# Shorthands
########################################################################################

def input_min_length() : return int(min(input_width(), input_height()))
def input_max_length() : return int(max(input_width(), input_height()))
def input_mpx()        : return input_width() * input_height() / float(1000000)

def main_factor() : return main_multiplier / settings.main.reduction
def soft_factor() : return cast(int, settings.soft.multiplier) / settings.soft.divisor
def hard_factor() : return cast(int, settings.hard.multiplier) / settings.hard.divisor

def max_factor()   : return max(soft_factor(), hard_factor())
def main_scaling() : return main_multiplier * settings.main.reduction
def limit_factor() : return ( 1 if ( main_scaling()                     >=
                                     cast(int, settings.soft.multiplier) )
                                else ( cast(int, settings.soft.multiplier) /
                                       settings.main.reduction             ) )
def total_factor() : return max(main_multiplier * max_factor(), limit_factor())
def output_min_length() : return min(output_width, output_height)
def output_max_length() : return max(output_width, output_height)

def base_main_width()  : return int(input_width()  / settings.main.reduction)
def base_main_height() : return int(input_height() / settings.main.reduction)
def base_soft_width()  : return int(output_width   / settings.soft.divisor)
def base_soft_height() : return int(output_height  / settings.soft.divisor)
def base_hard_width()  : return int(output_width   / settings.hard.divisor)
def base_hard_height() : return int(output_height  / settings.hard.divisor)

########################################################################################
# Combined Settings Validation
########################################################################################

def combined_settings_validation() -> None:

    assume ( settings.soft.multiplier >= cast(int, settings.soft.divisor) ,
             "soft-phase divisor exceeds multiplier"           )

    assume ( settings.hard.multiplier >= cast(int, settings.hard.divisor) ,
             "hard-phase divisor exceeds multiplier"            )

    assume ( input_min_length()  >= settings.main.reduction and
             output_min_length() >= settings.soft.divisor   and
             output_min_length() >= settings.hard.divisor     ,
             "attempt to generate an empty intermediate image")

    assume( input_mpx() * total_factor() ** 2 < MAX_MPX                            ,
            f"attempt to generate an intermediate image larger than {MAX_MPX} Mpx" )

    assume( not ( "alpha" in input_mode and
                  output_file_path.suffix.lower()[1:] in OPAQUE_FORMATS) ,
            f"the output format can't carry the input's alpha channel")

########################################################################################
# Progress Bar Logging
########################################################################################

progress_file_handle: TextIO

def create_progress_file() -> None:

    global progress_file_handle

    if savelevel() >= SaveLevel.debug:
        progress_file_handle = open(PROGRESS_FILE_PATH, "w")
        def close_progress_file(): progress_file_handle.close()
        atexit.register(close_progress_file)

########################################################################################
# Run System
########################################################################################

forward_i: int
forward_j: int
forward_k: int

def init_run_system() -> None:

    global forward_i
    global forward_j
    global forward_k

    forward_i = 0
    forward_j = 0
    forward_k = 0

def step_forward(unit: StepForward, bar: ProgressBar):

    global forward_j
    global forward_k

    phase, steps = PHASES[forward_i]
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

    global forward_i
    global forward_j

    log(f"the {PHASES[forward_i][0]} phase has been completed")

    forward_i += 1
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
