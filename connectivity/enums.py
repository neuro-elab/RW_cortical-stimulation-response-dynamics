from enum import Enum


class SleepStage(Enum):
    UNKNOWN = 0
    N3 = 1
    N2 = 2
    N1 = 3
    REM = 4
    QWAKE = 5
    AWAKE = 6
    ICTAL = 7
    OTHER = 8
    ARTEFACT = 9


class TimeGrade(Enum):
    NONE = 0
    UNKNOWN = 1
    NOISY = 2
    IED = 3
    ICTAL = 4
    IED_PS = 5
    IED_RB = 6
    IED_RS = 7
    IED_GS = 8
    IED_ST = 9
    IED_KE = 10
    IED_WS = 11
    IED_SSSS = 12
    IED_K = 13
    IED_SW = 14


class TraceGrade(Enum):
    UNKNOWN = 0
    NOISY = 1
    IED = 2
    ICTAL = 3
    PREVIEW = 4
    NORMAL = 5
