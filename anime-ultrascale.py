########################################################################################
# Imports
########################################################################################

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
import traceback
import numpy
import signal
import skimage
import wcwidth

from pathlib import Path
from enum import IntEnum, Enum
from datetime import datetime
from dataclasses import dataclass
from threading import Event, Thread, Lock
from typing import TypeVar, NoReturn, TypeAlias, Final, TextIO, Any, cast

########################################################################################
# Type Variables
########################################################################################

T = TypeVar("T")

########################################################################################
# Constants
########################################################################################

DEVELOPMENT_MODE     : Final = False

SOFTWARE_VERSION     : Final = "1.0"
OUTPUT_MODE          : Final = "png--4bands--srgb--alpha"
SUPPORTED_FORMATS    : Final = ["png", "jpg", "jpeg", "bmp", "webp", "tif", "tiff"]
OPAQUE_FORMATS       : Final = ["jpg", "jpeg", "bmp"]

MAX_MPX              : Final = 200
MAX_TILING           : Final = 16
MAX_ITERATIONS       : Final = 16

TEMP_FOLDER          : Final = "temp"
MODEL_FOLDER         : Final = "models"
SESSION_FOLDER       : Final = "sessions"
PRESET_FOLDER        : Final = "presets"
RENV_FOLDER          : Final = "renv"

SESSION_FILE         : Final = "session.json"
PRESET_FILE          : Final = "session.preset"
LOG_FILE             : Final = "log.txt"
INVOCATION_FILE      : Final = "invocation.txt"
SCALING_FILE         : Final = "scaling.txt"
SCALING_AI_FILE      : Final = "scaling_ai.txt"
PROGRESS_FILE        : Final = "progress.txt"
TEMP_INPUT_FILE      : Final = "in.png"
TEMP_OUTPUT_FILE     : Final = "output.png"
EXIT_FILE            : Final = "exit.txt"
AUTO_PRESET_FILE     : Final = "quality.preset"
RENV_FILE            : Final = "realesrgan-ncnn-vulkan"

DEFAULT_SAVE_LEVEL   : Final = "text"
DEFAULT_MAIN_FORMAT  : Final = "4k"
DEFAULT_MAIN_CLOSURE : Final = "bicubic"
DEFAULT_MAIN_TILING  : Final = 4
DEFAULT_SOFT_SCALER  : Final = "bicubic"
DEFAULT_HARD_SCALER  : Final = "lanczos"

PROGRESS_BAR_SIZE    : Final = 35
DESCALE_TARGET       : Final = 0.95
DESCALE_ITERATIONS   : Final = 8

########################################################################################
# Phases
########################################################################################

PHASES : Final = [

    ( "input"     , ["import"      , "downscaling" ]) ,
    ( "main"      , ["upscaling"   , "downscaling" ]) ,
    ( "soft"      , ["downscaling" , "upscaling"   ]) ,
    ( "hard"      , ["downscaling" , "upscaling"   ]) ,
    ( "output"    , ["downscaling" , "export"      ])
]

########################################################################################
# Invocation Data
########################################################################################

INVOCATION_INSTANT : Final = datetime.fromtimestamp(psutil.Process().create_time())
INVOCATION_PATH    : Final = Path(__file__).resolve().parent
INVOCATION_PID     : Final = os.getpid()

INVOCATION_DATE  : Final = INVOCATION_INSTANT.strftime('%Y-%m-%d')
INVOCATION_TIME  : Final = INVOCATION_INSTANT.strftime('%H-%M-%S')
INVOCATION_USEC  : Final = INVOCATION_INSTANT.strftime('%f')
INVOCATION_STAMP : Final = f"{INVOCATION_DATE}--{INVOCATION_TIME}--{INVOCATION_USEC}"

########################################################################################
# Paths
########################################################################################

MODEL_FOLDER_PATH   : Final = INVOCATION_PATH / MODEL_FOLDER
RENV_FOLDER_PATH    : Final = INVOCATION_PATH / RENV_FOLDER
PRESET_FOLDER_PATH  : Final = INVOCATION_PATH / PRESET_FOLDER

SESSION_FOLDER_PATH : Final = ( INVOCATION_PATH        /
                                SESSION_FOLDER         /
                                INVOCATION_DATE        /
                                f"{INVOCATION_TIME}--"  
                                f"{INVOCATION_USEC}--"  
                                f"{INVOCATION_PID}"    )

TEMP_FOLDER_PATH : Final = ( INVOCATION_PATH        /
                             TEMP_FOLDER            /
                             f"{INVOCATION_DATE}--"      
                             f"{INVOCATION_TIME}--"      
                             f"{INVOCATION_USEC}--"     
                             f"{INVOCATION_PID}"    )

INVOCATION_FILE_PATH     : Final = SESSION_FOLDER_PATH / INVOCATION_FILE
LOG_FILE_PATH            : Final = SESSION_FOLDER_PATH / LOG_FILE
SESSION_FILE_PATH        : Final = SESSION_FOLDER_PATH / SESSION_FILE
PRESET_FILE_PATH         : Final = SESSION_FOLDER_PATH / PRESET_FILE
SCALING_FILE_PATH        : Final = SESSION_FOLDER_PATH / SCALING_FILE
SCALING_AI_FILE_PATH     : Final = SESSION_FOLDER_PATH / SCALING_AI_FILE
PROGRESS_FILE_PATH       : Final = SESSION_FOLDER_PATH / PROGRESS_FILE
EXIT_FILE_PATH           : Final = SESSION_FOLDER_PATH / EXIT_FILE
TEMP_INPUT_FILE_PATH     : Final = TEMP_FOLDER_PATH    / TEMP_INPUT_FILE
TEMP_OUTPUT_FILE_PATH    : Final = TEMP_FOLDER_PATH    / TEMP_OUTPUT_FILE
RENV_RUNNER_PATH         : Final = RENV_FOLDER_PATH    / RENV_FILE
AUTO_PRESET_PATH         : Final = PRESET_FOLDER_PATH  / AUTO_PRESET_FILE

########################################################################################
# Save Levels
########################################################################################

class SaveLevel(IntEnum):

    dry       = 0
    nothing   = 1
    text      = 2
    error     = 3
    endpoints = 4
    debug     = 5
    research  = 6

def savelevel_descriptor(savelevel: SaveLevel):
    if   savelevel == SaveLevel.error: return "error"
    elif savelevel == SaveLevel.debug: return "debug"
    else:                              return "normal"

########################################################################################
# Scalers
########################################################################################

class Scaler(IntEnum):

    bilinear = 0
    bicubic  = 1
    lanczos  = 2

def scaler_descriptor(scaler: Scaler):
    if   scaler == Scaler.bilinear: return "linear"
    elif scaler == Scaler.bicubic:  return "cubic"
    else:                           return "lanczos3"

########################################################################################
# Arguments
########################################################################################

class Arguments(IntEnum):

    program_path = 0
    input_path   = 1
    output_path  = 2

class FullOptions(IntEnum):
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

class BasicOptions(IntEnum):
    main_format     = 1
    soft_enhancer   = 2
    soft_iterations = 3
    hard_enhancer   = 4
    hard_iterations = 5

class TwoOptions(IntEnum):
    format_or_preset_1 = 0
    format_or_preset_2 = 0

class OneOption(IntEnum):
    format_or_preset = 0

class ZeroOptions(IntEnum):
    pass

class FlexOptions(Enum):
    log = 0

class Flags(Enum):
    quiet = 0

def full_option_name(core_option: FullOptions) -> str:
    [x, y] = core_option.name.split("_")
    return "--" + (y[0].upper() if x == "hard" else y[0]) + y[1:]

def full_option_letter(core_option: FullOptions) -> str:
    [x, y] = core_option.name.split("_")
    return "-" + (y[0].upper() if x == "hard" else y[0])

def extra_option_name(extra_option: FlexOptions) -> str:
    return "--" + extra_option.name

def extra_option_letter(extra_option: FlexOptions) -> str:
    return "-" + extra_option.name[0]

full_options_map = ( { full_option_name(x)   : x for x in list(FullOptions) } |
                     { full_option_letter(x) : x for x in list(FullOptions) } )

flex_options_map = ( { extra_option_name(x)   : x for x in list(FlexOptions) } |
                     { extra_option_letter(x) : x for x in list(FlexOptions) } )

flags_map = ( { extra_option_name(x)   : x for x in list(Flags) } |
              { extra_option_letter(x) : x for x in list(Flags) } )

Options: TypeAlias = FullOptions | BasicOptions | TwoOptions | OneOption | ZeroOptions

########################################################################################
# Session
########################################################################################

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

########################################################################################
# Work Units
########################################################################################

@dataclass
class Scale:
    scaler     : Scaler
    multiplier : float
    in_width   : int
    in_height  : int
    out_width  : int
    out_height : int

@dataclass
class Enhance:
    enhancer   : str
    multiplier : int
    in_width   : int
    in_height  : int
    out_width  : int
    out_height : int

@dataclass
class Realesrgan:
    enhancer   : str
    multiplier : int
    in_width   : int
    in_height  : int
    out_width  : int
    out_height : int

@dataclass
class Save:
    width  : int
    height : int

@dataclass
class Load:
    width  : int
    height : int

@dataclass
class StepForward:
    save   : bool
    width  : int
    height : int

@dataclass
class PhaseForward:
    pass

Unit: TypeAlias = ( Scale | Enhance | Realesrgan  |
                    Save  | Load    | StepForward | PhaseForward )

UNIT_CLASSES : Final = [ Scale , Enhance , Realesrgan  ,
                         Save  , Load    , StepForward , PhaseForward ]

def unit_cost(unit: Unit) -> float:

    if   isinstance(unit, Scale) and not ( unit.in_width  == unit.out_width and
                                           unit.in_height == unit.out_height  ):
        return unit.out_width * unit.out_height * 0.1  / 1000000
    elif isinstance(unit, Enhance):
        return unit.in_width  * unit.in_height  * 1.33 / 1000000
    elif isinstance(unit, Realesrgan):
        return unit.in_width  * unit.in_height  * 1.00 / 1000000
    elif isinstance(unit, Save):
        return unit.width     * unit.height     * 0.33 / 1000000
    elif isinstance(unit, Load):
        return unit.width     * unit.height     * 0.05 / 1000000
    elif isinstance(unit, StepForward) and unit.save:
        return unit.width     * unit.height     * 0.33 / 1000000
    return 0

########################################################################################
# Time String
########################################################################################

def timestring(now: datetime) -> str:

    return now.strftime('on %Y/%m/%d at %H:%M:%S and %f')

########################################################################################
# File Opening with Automatic Closure
########################################################################################

def open_and_close_at_exit(path: Path) -> TextIO:
    handle = open(path, "w+")
    def close_handle():handle.close()
    atexit.register(close_handle)
    return handle

########################################################################################
# Early Error Reporting
########################################################################################

def early_fail( message   : str                         ,
                suggest   : bool = True                 ,
                exception : BaseException | None = None ) -> NoReturn:

    suggestion = " Run with --help for usage information." if suggest else ""
    text = message[:1].upper() + message[1:] + "." + suggestion
    if exception is None or not DEVELOPMENT_MODE:
        raise SystemExit(text)
    else:
        raise RuntimeError(text) from exception

def early_assume( condition : bool                        ,
                  message   : str                         ,
                  suggest   : bool = True                 ,
                  exception : BaseException | None = None ) -> None:

    if not condition:
        early_fail(message, suggest, exception)

########################################################################################
# Options Sorting
########################################################################################

input_file_path    : Path
output_file_path   : Path
positional_options : list[str]
flex_options       : dict[FlexOptions, str]
full_options       : dict[FullOptions, str]
flags              : set[Flags]
sort_options_now   : datetime

def sort_options() -> None:

    global input_file_path
    global output_file_path
    global positional_options
    global flex_options
    global full_options
    global flags
    global sort_options_now

    if len(sys.argv) == 1:
        print(HELP, end="")
        exit()

    if len(sys.argv) == 2:
        if sys.argv[1] in ["-h", "--help"]:
            print(HELP, end = "")
            exit()
        elif sys.argv[1] in ["-v", "--version"]:
            print(SOFTWARE_VERSION)
            exit()

    input_file_path = Path(sys.argv[Arguments.input_path])
    output_file_path = Path(sys.argv[Arguments.output_path])
    positional_options = []
    flex_options       = {}
    full_options       = {}
    flags              = set()

    early_assume( len(sys.argv) >= len(Arguments) ,
                  "incomplete I/O specification"  )

    i = len(Arguments)
    while i < len(sys.argv):

        arg = sys.argv[i]

        if not arg.startswith("-"):
            early_assume( not flex_options                         ,
                          "positional option following a flex one" )
            positional_options.append(arg)
            i += 1; continue

        if arg in flags_map.keys():
            early_assume( flags_map[arg] not in flags,
                          f"multiple instances of flag {arg}" )
            flags.add(flags_map[arg])
            i += 1; continue

        early_assume(i + 1 < len(sys.argv) and
                     not sys.argv[i + 1].startswith("-"),
                     f"missing value for flex option {arg}")

        early_assume( arg in flex_options_map.keys() or
                      arg in full_options_map.keys()  ,
                      f"unknown option {arg}"   )

        if arg in flex_options_map.keys():
            omap  = flex_options_map
            olist = flex_options
        elif arg in full_options_map.keys():
            omap  = full_options_map
            olist = full_options
        else:
            early_fail(f"Unknown option {arg}")

        option = omap[arg]

        early_assume( option.name not in olist            ,
                      f"multiple values for option {arg}" )

        olist[option] = sys.argv[i + 1]

        i += 2

    sort_options_now = datetime.now()

########################################################################################
# Save Level
########################################################################################

def savelevel() -> SaveLevel:

    return SaveLevel[flex_options.get(FlexOptions.log, DEFAULT_SAVE_LEVEL)]

########################################################################################
# Session Folder Creation
########################################################################################

create_session_folder_now : datetime

def create_session_folder() -> None:

    global create_session_folder_now

    if savelevel() >= SaveLevel.text:
        SESSION_FOLDER_PATH.mkdir(parents = True, exist_ok = True)

    create_session_folder_now = datetime.now()

########################################################################################
# Exit Message
########################################################################################

exit_file_handle     : TextIO
create_exit_file_now : datetime

def create_exit_file() -> None:

    global exit_file_handle
    global create_exit_file_now

    if savelevel() >= SaveLevel.debug:
        exit_file_handle = open_and_close_at_exit(EXIT_FILE_PATH)

    create_exit_file_now = datetime.now()

def record_outcome(message: str) -> None:

    if savelevel() >= SaveLevel.debug:
        exit_file_handle.write(message + "\n")
        exit_file_handle.flush()

########################################################################################
# Logging
########################################################################################

log_file_handle     : TextIO
create_log_file_now : datetime

def create_log_file() -> None:

    global log_file_handle
    global create_log_file_now

    if savelevel() >= SaveLevel.text:
        log_file_handle = open_and_close_at_exit(LOG_FILE_PATH)

    create_log_file_now = datetime.now()

def log ( message: str                      ,
          level: SaveLevel = SaveLevel.text ,
          now : datetime | None = None      ):

    if savelevel() >= SaveLevel.text and savelevel() >= level:
        now = now or datetime.now()
        message = ( f"{timestring(now)}, "
                    f"level {savelevel_descriptor(level).upper()}: "
                    f"{message}" )
        log_file_handle.write(message + "\n")
        log_file_handle.flush()

########################################################################################
# Error Reporting
########################################################################################

def fail( message   : str                         ,
          suggest   : bool = True                 ,
          exception : BaseException | None = None ) -> NoReturn:

    log(message, SaveLevel.error)
    early_fail(message, suggest, exception)

def assume( condition : bool                        ,
            message   : str                         ,
            suggest   : bool = True                 ,
            exception : BaseException | None = None ) -> None:

    if not condition:
        fail(message, suggest, exception)

########################################################################################
# Invocation File Creation
########################################################################################

def create_invocation_file() -> None:

    if savelevel() >= SaveLevel.text:
        stamp = INVOCATION_STAMP
        INVOCATION_FILE_PATH.write_text( f"PID: {INVOCATION_PID}\n"         +
                                         f"Timestamp: {stamp}\n"            +
                                         f"PWD: {Path.cwd()}\n"             +
                                         f"Command: {' '.join(sys.argv)}\n" )

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

########################################################################################
# Scaling Logging
########################################################################################

scaling_file_handle: TextIO

def create_scaling_file() -> None:

    global scaling_file_handle

    if savelevel() >= SaveLevel.debug:
        scaling_file_handle = open_and_close_at_exit(SCALING_FILE_PATH)

def log_scaling(job: str, event: str, progress: Any) -> None:

    if savelevel() >= SaveLevel.debug:
        scaling_file_handle.write( f"{timestring(datetime.now())}: "
                                   f"job={job}, "
                                   f"event={event}, "
                                   f"percent={progress.percent}, "
                                   f"run={progress.run}, "
                                   f"eta={progress.eta}, "
                                   f"npels={progress.npels}, "
                                   f"tpels={progress.tpels}\n" )
        scaling_file_handle.flush()

########################################################################################
# AI Scaling Logging
########################################################################################

scaling_ai_file_handle: TextIO

def create_scaling_ai_file() -> None:

    global scaling_ai_file_handle

    if savelevel() >= SaveLevel.debug:
        scaling_ai_file_handle = open_and_close_at_exit(SCALING_AI_FILE_PATH)

########################################################################################
# I/O Existence Checks
########################################################################################

def io_existence_checks() -> None:
    assume(input_file_path.is_file(), "input file does not exist")
    assume( input_file_path.suffix.lower()[1:] in SUPPORTED_FORMATS ,
            "input extension is not supported"                      )
    assume(output_file_path.parent.is_dir(), "output folder does not exist")
    assume( output_file_path.suffix.lower()[1:] in SUPPORTED_FORMATS ,
            "output extension is not supported"                      )

########################################################################################
# Internal Existence Checks
########################################################################################

def internal_existence_checks() -> None:
    assume(RENV_RUNNER_PATH.is_file(), "missing Real ESRGAN runner")

########################################################################################
# Progress Bar
########################################################################################

# TODO 'ProgressBar' refactoring (and turning into zero the first percentage of a unit)
#                                (and moving out breakpoints estimation)

class ProgressBar:

    def __init__(self, total_cost: float) -> None:
        current_time = time.perf_counter()

        self.total_cost = total_cost
        self.completed_cost = 0.0

        self.current_width = input_width()
        self.current_height = input_height()
        self.current_unit = PhaseForward
        self.current_unit_type = PhaseForward.__name__
        self.current_unit_cost = 0.0
        self.current_unit_completed_cost = 0.0
        self.estimated_cost = 0.0
        self.last_reported_percentage = 0.0
        self.current_unit_zero_reported = False
        self.current_unit_zero_breakpoint_reached = False
        self.current_unit_tail_breakpoint_reached = False
        self.current_unit_zero_percentage = 0.0
        self.current_unit_last_percentage = 0.0

        self.estimated_speed = 0.0
        self.displayed_speed = 0.0
        self.last_report_time = current_time
        self.last_refresh_time = current_time
        self.last_render_time = current_time
        self.last_rendered_percentage = 0.0
        self.last_logged_percentage_text: str | None = None
        self.progress_completed = False
        self.cursor_hidden = False

        unit_types = [unit_class.__name__ for unit_class in UNIT_CLASSES]
        self.total_elapsed_time_by_type = {name: 0.0 for name in unit_types}
        self.total_completed_cost_by_type = {name: 0.0 for name in unit_types}
        self.pre_zero_elapsed_time_by_type = {name: 0.0 for name in unit_types}
        self.pre_zero_completed_cost_by_type = {name: 0.0 for name in unit_types}
        self.post_zero_elapsed_time_by_type = {name: 0.0 for name in unit_types}
        self.post_zero_completed_cost_by_type = {name: 0.0 for name in unit_types}
        self.tail_elapsed_time_by_type = {name: 0.0 for name in unit_types}
        self.tail_completed_cost_by_type = {name: 0.0 for name in unit_types}

        self.lock = Lock()
        self.stop_event = Event()
        self.refresh_thread = Thread(
            target=self._refresh_loop,
            daemon=True,
        )
        self.refresh_thread.start()

        if Flags.quiet not in flags:
            print("\033[?25l", end="", flush=True)
            self.cursor_hidden = True
        self._render()

    def new_unit(self, unit: Unit) -> None:
        with self.lock:
            if self.current_unit_cost > 0.0:
                self._complete_current_unit()

            unit_type = type(unit).__name__

            current_time = time.perf_counter()

            if isinstance(self.current_unit, (Scale, Enhance)):
                self.current_width = self.current_unit.out_width
                self.current_height = self.current_unit.out_height
            self.current_unit = unit
            self.current_unit_type = unit_type
            self.current_unit_cost = unit_cost(unit)
            self.current_unit_completed_cost = 0.0
            self.estimated_cost = 0.0
            self.last_reported_percentage = 0.0
            self.current_unit_zero_reported = False
            self.current_unit_zero_breakpoint_reached = False
            self.current_unit_tail_breakpoint_reached = False
            self.current_unit_zero_percentage = (
                self._zero_percentage_for_current_unit()
            )
            self.current_unit_last_percentage = (
                self._last_percentage_for_current_unit()
            )
            self.estimated_speed = self._current_integral_speed()
            self.last_report_time = current_time
            self.last_refresh_time = current_time

    def progress(self, percentage: float) -> None:
        with self.lock:
            self._progress(percentage)

    def _progress(self, percentage: float) -> None:
        remapped_percentage = self._remap_percentage(percentage)
        percentage_delta = max(
            0.0,
            remapped_percentage - self.last_reported_percentage,
        )
        reported_cost = self.current_unit_cost * percentage_delta / 100.0

        estimated_overlap = min(self.estimated_cost, reported_cost)
        newly_completed_cost = reported_cost - estimated_overlap

        current_time = time.perf_counter()
        elapsed_time = current_time - self.last_report_time

        use_post_zero_average = percentage > 0.0

        self.last_reported_percentage = remapped_percentage
        self.last_report_time = current_time

        if elapsed_time > 0.0:
            self.total_elapsed_time_by_type[
                self.current_unit_type
            ] += elapsed_time
            self.total_completed_cost_by_type[
                self.current_unit_type
            ] += reported_cost

            if use_post_zero_average:
                self.post_zero_elapsed_time_by_type[
                    self.current_unit_type
                ] += elapsed_time
                self.post_zero_completed_cost_by_type[
                    self.current_unit_type
                ] += reported_cost
            else:
                self.pre_zero_elapsed_time_by_type[
                    self.current_unit_type
                ] += elapsed_time
                self.pre_zero_completed_cost_by_type[
                    self.current_unit_type
                ] += reported_cost

        if percentage == 0.0:
            self.current_unit_zero_reported = True
        elif percentage == 100.0:
            self.current_unit_tail_breakpoint_reached = True

        self.estimated_cost -= max(0.0, estimated_overlap)
        self._add_cost(newly_completed_cost)
        self._update_current_unit_breakpoints()
        self.estimated_speed = self._current_integral_speed()
        self._render()

    def _complete_current_unit(self) -> None:
        current_time = time.perf_counter()
        elapsed_time = current_time - self.last_report_time
        percentage_delta = max(0.0, 100.0 - self.last_reported_percentage)
        tail_cost = self.current_unit_cost * percentage_delta / 100.0

        if elapsed_time > 0.0:
            self.total_elapsed_time_by_type[
                self.current_unit_type
            ] += elapsed_time
            self.total_completed_cost_by_type[
                self.current_unit_type
            ] += tail_cost
            self.tail_elapsed_time_by_type[
                self.current_unit_type
            ] += elapsed_time
            self.tail_completed_cost_by_type[
                self.current_unit_type
            ] += tail_cost

        remaining_cost = max(
            0.0,
            self.current_unit_cost - self.current_unit_completed_cost,
        )

        self.estimated_cost = 0.0
        self.last_reported_percentage = 100.0
        self.last_report_time = current_time
        self.current_unit_tail_breakpoint_reached = True
        self._add_cost(remaining_cost)

    def _add_cost(self, cost: float) -> None:
        self.completed_cost = min(
            self.completed_cost + cost,
            self.total_cost,
        )
        self.current_unit_completed_cost = min(
            self.current_unit_completed_cost + cost,
            self.current_unit_cost,
        )

    def refresh(self) -> None:
        with self.lock:
            current_time = time.perf_counter()
            elapsed_time = current_time - self.last_refresh_time
            self.last_refresh_time = current_time

            estimated_cost_limit = self.current_unit_cost

            estimated_cost = max(
                0.0,
                min(
                    elapsed_time * self.estimated_speed,
                    estimated_cost_limit - self.current_unit_completed_cost,
                ),
            )

            self.estimated_cost += estimated_cost
            self._add_cost(estimated_cost)
            self._update_current_unit_breakpoints()
            self.estimated_speed = self._current_integral_speed()
            self._render()

    def _current_integral_speed(self) -> float:
        if self.current_unit_tail_breakpoint_reached:
            elapsed_time = self.tail_elapsed_time_by_type[
                self.current_unit_type
            ]
            completed_cost = self.tail_completed_cost_by_type[
                self.current_unit_type
            ]
        elif (
            self.current_unit_zero_breakpoint_reached
            or self.current_unit_zero_reported
        ):
            elapsed_time = self.post_zero_elapsed_time_by_type[
                self.current_unit_type
            ]
            completed_cost = self.post_zero_completed_cost_by_type[
                self.current_unit_type
            ]
        else:
            elapsed_time = self.pre_zero_elapsed_time_by_type[
                self.current_unit_type
            ]
            completed_cost = self.pre_zero_completed_cost_by_type[
                self.current_unit_type
            ]

        return completed_cost / elapsed_time if elapsed_time > 0.0 else 0.0

    def _update_current_unit_breakpoints(self) -> None:
        zero_breakpoint_cost = (
            self.current_unit_cost
            * self.current_unit_zero_percentage
            / 100.0
        )
        tail_breakpoint_cost = (
            self.current_unit_cost
            * self.current_unit_last_percentage
            / 100.0
        )

        if self.current_unit_completed_cost >= zero_breakpoint_cost:
            self.current_unit_zero_breakpoint_reached = True

        if self.current_unit_completed_cost >= tail_breakpoint_cost:
            self.current_unit_tail_breakpoint_reached = True

    def _zero_percentage_for_current_unit(self) -> float:
        elapsed_time = self.total_elapsed_time_by_type[
            self.current_unit_type
        ]

        if elapsed_time == 0.0: # change
            return 10.0

        zero_percentage = 100.0 * self.pre_zero_elapsed_time_by_type[
            self.current_unit_type
        ] / elapsed_time

        return max(0.0, min(90.0, zero_percentage))

    def _last_percentage_for_current_unit(self) -> float:
        elapsed_time = self.total_elapsed_time_by_type[
            self.current_unit_type
        ]

        if elapsed_time == 0.0: # change
            if self.current_unit_type == Realesrgan.__name__:
                return 50.0

            return 90.0

        last_percentage = 100.0 * (
            elapsed_time
            - self.tail_elapsed_time_by_type[self.current_unit_type]
        ) / elapsed_time
        minimum_percentage = self.current_unit_zero_percentage + 1.0

        return max(
            minimum_percentage,
            min(99.0, last_percentage),
        )

    def _remap_percentage(self, percentage: float) -> float:
        zero_percentage = self.current_unit_zero_percentage
        report_span = (
            self.current_unit_last_percentage
            - zero_percentage
        )

        return zero_percentage + percentage * report_span / 100.0

    def complete(self) -> None:
        self._stop_refresh_thread()

        with self.lock:
            if self.current_unit_cost > 0.0:
                self._complete_current_unit()

            self.completed_cost = self.total_cost
            self.progress_completed = True
            self.displayed_speed = 0.0
            self._render()

            if self.cursor_hidden:
                print("\033[?25h", end="", flush=True)
                self.cursor_hidden = False

    def _render(self) -> None:
        current_time = time.perf_counter()
        elapsed_time = current_time - self.last_render_time
        self.last_render_time = current_time

        raw_percentage = (
            100.0
            if self.total_cost == 0.0
            else 100.0 * self.completed_cost / self.total_cost
        )

        percentage = raw_percentage
        if not self.progress_completed:
            percentage = min(99.99, percentage)

        displayed_completed_cost = (
            self.total_cost
            if self.progress_completed
            else min(
                self.completed_cost,
                self.total_cost * percentage / 100.0,
            )
        )

        if self.progress_completed:
            self.displayed_speed = 0.0
        elif elapsed_time > 0.0:
            current_speed = (
                (percentage - self.last_rendered_percentage)
                * self.total_cost
                / (100.0 * elapsed_time)
            )
            decay = math.exp(-elapsed_time / 0.9)
            self.displayed_speed = (
                current_speed
                if self.displayed_speed == 0.0
                else (
                    decay * self.displayed_speed
                    + (1.0 - decay) * current_speed
                )
            )

        self.last_rendered_percentage = percentage

        percentage_text = f"{percentage:‥>5.{2 if percentage < 100 else 1}f}"

        bar_percentage = percentage
        if not self.progress_completed:
            bar_percentage = min(
                bar_percentage,
                100.0 * (PROGRESS_BAR_SIZE - 1) / PROGRESS_BAR_SIZE,
            )

        partial_width = PROGRESS_BAR_SIZE * bar_percentage / 100.0
        filled_width = int(partial_width)
        partial_index = int((partial_width - filled_width) * 8)
        partial = " ▏▎▍▌▋▊▉"[partial_index]
        empty_width = PROGRESS_BAR_SIZE - filled_width - (partial_index > 0)
        bar = "█" * filled_width + partial.strip() + " " * empty_width

        cost_digits = 3 + math.ceil(math.log10(math.floor(self.total_cost)))

        line = (
            f"【{bar}】"
            f"{percentage_text}% | "
            f"{displayed_completed_cost:‥>{cost_digits}.2f} / "
            f"{self.total_cost:{cost_digits}.2f} Mpx | "
            f"{self.displayed_speed:.2f} Mpx/s"
        )

        if percentage == 100:
            msg = "complete"
        elif isinstance(self.current_unit, Scale):
            msg = ( f"{self.current_unit.in_width} x {self.current_unit.in_height} px -> "
                    f"{self.current_unit.out_width} x {self.current_unit.out_height} px " 
                    f"with {self.current_unit.scaler} ...")
        elif isinstance(self.current_unit, Realesrgan):
            msg = (f"{self.current_unit.in_width} x {self.current_unit.in_height} px -> "
                   f"{self.current_unit.out_width} x {self.current_unit.out_height} px "
                   f"with {self.current_unit.enhancer} ...")
        else:
            msg = f"collateral work at {self.current_width} x {self.current_height} px ..."

        if Flags.quiet not in flags:
            print("\033[1A\033[2K\r", end="")
            print("\033[1A\033[2K\r", end="")
            print("\n", end="")
            print(line, end="")
            print("\033[1B\033[2K\r", end="")
            print(" " * (wcwidth.wcswidth(line) - wcwidth.wcswidth(msg)) + msg, end="", flush=True)

        if (
            savelevel() >= SaveLevel.debug
            and percentage_text != self.last_logged_percentage_text
        ):
            progress_file_handle.write(timestring(datetime.now()) + ": " + line + "\n")
            progress_file_handle.flush()
            self.last_logged_percentage_text = percentage_text

    def _refresh_loop(self) -> None:
        while not self.stop_event.wait(0.1):
            self.refresh()

    def _stop_refresh_thread(self) -> None:
        self.stop_event.set()
        self.refresh_thread.join()

    def __enter__(self) -> "ProgressBar":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._stop_refresh_thread()
        if self.cursor_hidden:
            print("\033[?25h", end="", flush=True)
            self.cursor_hidden = False

    def __del__(self) -> None:
        if getattr(self, "cursor_hidden", False):
            print("\033[?25h", end="", flush=True)
            self.cursor_hidden = False

########################################################################################
# Image Loading
########################################################################################

input_mode    : str
input_image   : pyvips.Image
current_image : pyvips.Image
output_mode   : str

def current_width()  -> int: return current_image.width
def current_height() -> int: return current_image.height
def input_width()    -> int: return input_image.width
def input_height()   -> int: return input_image.height

# TODO 'load' refactoring --------------------------------------------------------------

def load(unit: Load, path: Path, bar: ProgressBar | None = None) -> pyvips.Image:

    if bar is not None:
        bar.new_unit(unit)

    image = pyvips.Image.new_from_file(str(path), access="sequential")

    image = image.colourspace("srgb")

    if image.bands > 4:
        image = image[:4]

    image = image.cast("uchar")

    interrupted = Event()
    previous_sigint = signal.getsignal(signal.SIGINT)

    def request_interrupt(signum, frame) -> None:
        interrupted.set()

    def update_interrupt(image: pyvips.Image, progress: Any) -> None:
        if interrupted.is_set():
            image.set_kill(True)

    def update_progress(image: pyvips.Image, progress: Any) -> None:
        if bar is not None:
            bar.progress(float(progress.percent))

    image.set_progress(True)
    image.signal_connect("eval", update_interrupt)

    image.signal_connect(
        "preeval",
        lambda image, progress:
            log_scaling("load", "preeval", progress),
    )
    image.signal_connect(
        "eval",
        lambda image, progress:
            log_scaling("load", "eval", progress),
    )
    image.signal_connect(
        "eval",
        update_progress,
    )
    image.signal_connect(
        "posteval",
        lambda image, progress:
            log_scaling("load", "posteval", progress),
    )

    signal.signal(signal.SIGINT, request_interrupt)

    try:
        if bar is not None:
            bar.progress(0.0)

        loaded_image = image.copy_memory()

        if bar is not None:
            bar.progress(100.0)

        if interrupted.is_set():
            raise KeyboardInterrupt

    except pyvips.Error:
        if interrupted.is_set():
            raise KeyboardInterrupt from None
        raise

    finally:
        signal.signal(signal.SIGINT, previous_sigint)

    return loaded_image

def load_input_image() -> None:

    global input_mode
    global input_image
    global current_image
    global output_mode

    initial_image = pyvips.Image.new_from_file( str(input_file_path)  ,
                                                access = "sequential" )

    ifmt = f"{input_file_path.suffix.lower()[1:]}"
 #   ibands = f"{initial_image.bands}bands"
    imode = initial_image.interpretation
    ialpha = 'alpha' if initial_image.hasalpha() else 'opaque'

    ofmt = f"{output_file_path.suffix.lower()[1:]}"
 #   obands = ( str( initial_image.bands -
 #                   (initial_image.hasalpha() and ofmt in OPAQUE_FORMATS) ) + "bands" )
    omode = "srgb"
    oalpha = ialpha

    input_mode  = f"{ifmt}--{imode}--{ialpha}"
    output_mode = f"{ofmt}--{omode}--{oalpha}"

    input_image = load( Load(initial_image.width, initial_image.height) ,
                        input_file_path                                 )
    current_image = input_image.copy()

########################################################################################
# Image Descaling
########################################################################################

# TODO refactoring of this section -----------------------------------------------------

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

# TODO 'save' refactoring --------------------------------------------------------------

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

# TODO 'scale' refactoring -------------------------------------------------------------

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

    if savelevel() <= SaveLevel.dry:
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

HELP = textwrap.dedent("""\
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
        Output image in any of the following formats: PNG (RGBA), JPG/JPEG
        (RGB), BMP (RGB), TIF/TIFF (RGBA), WEBP (RGBA).

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
    
    {-q│--quiet} (ignored by --log dry)
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
