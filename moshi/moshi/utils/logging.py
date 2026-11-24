# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import logging
import sys
import random
import string
from typing import Optional
from .logging_interface import LoggingProvider


_provider: Optional[LoggingProvider] = None


def configure_logging(provider: LoggingProvider):
    """Explicitly wire the logging provider implementation."""
    global _provider
    _provider = provider


def _get_provider() -> LoggingProvider:
    """Internal helper to get the active logging provider."""
    if _provider is None:
        raise RuntimeError(
            "Logging provider not configured. Call configure_logging() "
            "during application startup."
        )
    return _provider


def random_id(n=4):
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=n))


def setup_logger(name: str, log_file=None, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def print_log(level: str, msg: str, prefix: Optional[str] = None, info_color: Optional[str] = None):
    provider = _get_provider()
    colorized_msg = provider.make_log(level, msg) if info_color is None or level != "info" else provider.colorize(msg, info_color)
    if prefix is None:
        print(colorized_msg)
    else:
        print(prefix + colorized_msg)


class ColorizedLog(object):
    def __init__(self, prefix: str, info_color: str):
        self.prefix = prefix
        self.info_color = info_color

    def log(self, level: str, msg: str):
        print_log(level, msg, prefix=self.prefix, info_color=self.info_color)

    @classmethod
    def randomize(cls):
        cid = random_id()
        color = random.choice(["91", "92", "93", "94", "95", "96", "97"])
        prefix = _get_provider().colorize(f"[{cid}] ", color)
        return cls(prefix=prefix, info_color=color)
