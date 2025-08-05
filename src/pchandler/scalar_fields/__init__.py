__all__ = [
    "ScalarField",
    "RGBFields",
    "NormalFields",
    "ScalarFieldManager",
]

__author__ = "Nicholas Meyer"
__email__ = "meyernic@ethz.ch"

import logging
from logging import config as logconfig

from .scalar_fields import ScalarField, RGBFields, NormalFields
from .scalar_field_manager import ScalarFieldManager