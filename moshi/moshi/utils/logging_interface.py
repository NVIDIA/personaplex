# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

from typing import Protocol

class LoggingProvider(Protocol):
    """Abstraction for logging capabilities required by low-level modules."""
    
    def colorize(self, text: str, color: str) -> str:
        """Colorize text with ANSI codes."""
        ...

    def make_log(self, level: str, msg: str) -> str:
        """Create a colorized log message with a level prefix."""
        ...
