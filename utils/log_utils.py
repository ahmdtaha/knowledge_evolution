import logging
import os.path as osp
# from utils import os_utils

# tensorflow_logger = logging.getLogger('tensorflow')
# tensorflow_logger.setLevel(logging.DEBUG)
#
# # create formatter and add it to the handlers
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# # create file handler which logs even debug messages
# fh = logging.FileHandler(os.path.join(config.model_save_path ,'tensorflow.txt'))
# fh.setLevel(logging.DEBUG)
# fh.setFormatter(formatter)
# tensorflow_logger.addHandler(fh)


# Copyright (c) 2014 Markus Pointner
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

class _AnsiColorStreamHandler(logging.StreamHandler):
    DEFAULT = '\x1b[0m'
    RED     = '\x1b[31m'
    GREEN   = '\x1b[32m'
    YELLOW  = '\x1b[33m'
    CYAN    = '\x1b[36m'

    CRITICAL = RED
    ERROR    = RED
    WARNING  = YELLOW
    INFO     = DEFAULT  # GREEN
    DEBUG    = CYAN

    @classmethod
    def _get_color(cls, level):
        if level >= logging.CRITICAL:  return cls.CRITICAL
        elif level >= logging.ERROR:   return cls.ERROR
        elif level >= logging.WARNING: return cls.WARNING
        elif level >= logging.INFO:    return cls.INFO
        elif level >= logging.DEBUG:   return cls.DEBUG
        else:                          return cls.DEFAULT

    def __init__(self, stream=None):
        logging.StreamHandler.__init__(self, stream)

    def format(self, record):
        text = logging.StreamHandler.format(self, record)
        color = self._get_color(record.levelno)
        return (color + text + self.DEFAULT) if self.is_tty() else text

    def is_tty(self):
        isatty = getattr(self.stream, 'isatty', None)
        return isatty and isatty()


class _WinColorStreamHandler(logging.StreamHandler):
    # wincon.h
    FOREGROUND_BLACK     = 0x0000
    FOREGROUND_BLUE      = 0x0001
    FOREGROUND_GREEN     = 0x0002
    FOREGROUND_CYAN      = 0x0003
    FOREGROUND_RED       = 0x0004
    FOREGROUND_MAGENTA   = 0x0005
    FOREGROUND_YELLOW    = 0x0006
    FOREGROUND_GREY      = 0x0007
    FOREGROUND_INTENSITY = 0x0008 # foreground color is intensified.
    FOREGROUND_WHITE     = FOREGROUND_BLUE | FOREGROUND_GREEN | FOREGROUND_RED

    BACKGROUND_BLACK     = 0x0000
    BACKGROUND_BLUE      = 0x0010
    BACKGROUND_GREEN     = 0x0020
    BACKGROUND_CYAN      = 0x0030
    BACKGROUND_RED       = 0x0040
    BACKGROUND_MAGENTA   = 0x0050
    BACKGROUND_YELLOW    = 0x0060
    BACKGROUND_GREY      = 0x0070
    BACKGROUND_INTENSITY = 0x0080 # background color is intensified.

    DEFAULT  = FOREGROUND_WHITE
    CRITICAL = BACKGROUND_YELLOW | FOREGROUND_RED | FOREGROUND_INTENSITY | BACKGROUND_INTENSITY
    ERROR    = FOREGROUND_RED | FOREGROUND_INTENSITY
    WARNING  = FOREGROUND_YELLOW | FOREGROUND_INTENSITY
    INFO     = FOREGROUND_GREEN
    DEBUG    = FOREGROUND_CYAN

    @classmethod
    def _get_color(cls, level):
        if level >= logging.CRITICAL:  return cls.CRITICAL
        elif level >= logging.ERROR:   return cls.ERROR
        elif level >= logging.WARNING: return cls.WARNING
        elif level >= logging.INFO:    return cls.INFO
        elif level >= logging.DEBUG:   return cls.DEBUG
        else:                          return cls.DEFAULT

    def _set_color(self, code):
        import ctypes
        ctypes.windll.kernel32.SetConsoleTextAttribute(self._outhdl, code)

    def __init__(self, stream=None):
        logging.StreamHandler.__init__(self, stream)
        # get file handle for the stream
        import ctypes, ctypes.util
        # for some reason find_msvcrt() sometimes doesn't find msvcrt.dll on my system?
        crtname = ctypes.util.find_msvcrt()
        if not crtname:
            crtname = ctypes.util.find_library("msvcrt")
        crtlib = ctypes.cdll.LoadLibrary(crtname)
        self._outhdl = crtlib._get_osfhandle(self.stream.fileno())

    def emit(self, record):
        color = self._get_color(record.levelno)
        self._set_color(color)
        logging.StreamHandler.emit(self, record)
        self._set_color(self.FOREGROUND_WHITE)

# select ColorStreamHandler based on platform
import platform
if platform.system() == 'Windows':
    ColorStreamHandler = _WinColorStreamHandler
else:
    ColorStreamHandler = _AnsiColorStreamHandler

def get_logging_dict(name,mode='w'):
    return {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            },
        },
        'handlers': {
            'stderr': {
                'level': 'INFO',
                'formatter': 'standard',
                'class': 'utils.log_utils.ColorStreamHandler',
                'stream': 'ext://sys.stderr',
            },
            'logfile': {
                'level': 'DEBUG',
                'formatter': 'standard',
                'class': 'logging.FileHandler',
                'filename': name,
                'mode': mode,
            }
        },
        'loggers': {
            '': {
                'handlers': ['stderr', 'logfile'],
                'level': 'DEBUG',
                'propagate': True
            },

            # extra ones to shut up.
            'tensorflow': {
                'handlers': ['stderr', 'logfile'],
                'level': 'INFO',
            },
        }
    }

def create_logger(log_file):
    import logging.config
    logging.config.dictConfig(get_logging_dict(log_file))
    filename = osp.basename(log_file)
    logger = logging.getLogger(filename)
    return logger

def get_logger_by_tag(tag):
    logger = logging.getLogger(tag)
    return logger


