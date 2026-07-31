from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict

######################################################################
class UnitCell(str, Enum):
  SQUARE = 'square'
  HEXAGON = 'hexagon'
  
class HoleShape(str, Enum):
  CIRCLE = 'circle'
  HEXAGON = 'hexagon'
  OCTAGON = 'octagon'
  TRIANGLE = 'triangle'
  NESTEGGS = 'nesteggs'
  TRIVIALPURSUIT = 'trivialpursuit'

class PadShape(str, Enum):
  SQUARE = 'square'
  HEXAGON = 'hexagon'
  OCTAGON = 'octagon'

class ScaleOption(str, Enum):
  CORNER = 'corner'
  SINGLE = 'single'
  HALF = 'half'
  SURROUNDING = 'surrounding'
######################################################################

@dataclass
class GeometryConfiguration:
  """
  Defines and validates the shape and scale of the simulation geometry.
  """

  unitCell: UnitCell
  scale: ScaleOption
  holeShape: HoleShape
  padShape: PadShape

  def __post_init__(self):
    self.unitCell = self._validateEnum(self.unitCell, UnitCell, "unitCell")
    self.scale = self._validateEnum(self.scale, ScaleOption, "scale")
    self.holeShape = self._validateEnum(self.holeShape, HoleShape, "holeShape")
    self.padShape = self._validateEnum(self.padShape, PadShape, "padShape")

  @staticmethod
  def _validateEnum(value: Any, enumClass: type[Enum], fieldName: str) -> Enum:
    """
    Validates and coerces a value into a specific Enum class.
    """
    if isinstance(value, enumClass):
      return value

    cleanValue = str(value).lower()
    try:
      return enumClass(cleanValue)
    except ValueError:
      validOptions = [e.value for e in enumClass]
      raise ValueError(
        f"Invalid {fieldName}: '{value}'. Must be one of {validOptions}."
      ) from None

  @classmethod
  def fromDict(geoClass, dataDict: Dict[str, Any]) -> "GeometryConfiguration":
    """
    Instantiate directly from a raw input dictionary.
    """
    requiredKeys = {'unitCell', 'scale', 'holeShape', 'padShape'}
    missingKeys = requiredKeys - dataDict.keys()
    
    if missingKeys:
      raise ValueError(f"Error - Missing required configuration keys: {missingKeys}")

    classObject = geoClass(
      unitCell=dataDict['unitCell'],
      scale=dataDict['scale'],
      holeShape=dataDict['holeShape'],
      padShape=dataDict['padShape'],
    )
    
    return classObject
######################################################################


