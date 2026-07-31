from enum import Enum

class SignalTypeMapping(Enum):
    BUY: int = 1
    SELL: int = -1
    HOLD: int = 0
