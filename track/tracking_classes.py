from dataclasses import dataclass, field
from typing import List, Any
import math


@dataclass
class Track1Params:
    inputVideoPath: str = ""
    outputMatPath: str = ""
    min_area: int = 0
    max_area: float = 0.0
    minSat_color: float = 0.0
    minVal_color: float = 0.0
    redHueLow1: float = 0.0
    redHueHigh1: float = 0.0
    redHueLow2: float = 0.0
    redHueHigh2: float = 0.0
    greenHueLow: float = 0.0
    greenHueHigh: float = 0.0
    whiteMaxSat: float = 0.0
    whiteMinVal: float = 0.0
    colorOpenRadius: int = 0
    colorCloseRadius: int = 0
    whiteCloseRadius: int = 0
    ringInnerRadius: int = 0
    ringOuterRadius: int = 0
    minWhiteCoverageFraction: float = 0.0
    rejectNearImageBorder: bool = False
    borderMarginPx: int = 0
    ccConnectivity: int = 8
    assumedInputType: str = ""
    createdOn: str = ""


@dataclass
class DetectionRecord:
    x: float
    y: float
    color: str
    area: float
    angle: float = float("nan")


@dataclass
class FrameDetections:
    frame_number: int
    frame_time_s: float
    detections: List[DetectionRecord]


@dataclass
class VideoCentroids:
    filepath: str
    frames: List[FrameDetections]
    params: Track1Params
    nFrames: int = 0
    fps: float = 0.0
    passedVerification: bool = False
    meanBlockDistance: float = 0.0

    @classmethod
    def from_dict(cls, data: dict) -> 'VideoCentroids':
        p_data = data.get('params', {})
        params = Track1Params(**p_data)

        frames = []
        for f_data in data.get('frames', []):
            dets = []
            for d in f_data.get('detections', []):
                dd = dict(d)
                if 'angle' not in dd:
                    dd['angle'] = float("nan")
                dets.append(DetectionRecord(**dd))

            frames.append(FrameDetections(
                frame_number=f_data['frame_number'],
                frame_time_s=f_data['frame_time_s'],
                detections=dets
            ))

        return cls(
            filepath=data.get('filepath', ''),
            frames=frames,
            params=params,
            nFrames=data.get('nFrames', 0),
            fps=data.get('fps', 0.0),
            passedVerification=data.get('passedVerification', False),
            meanBlockDistance=data.get('meanBlockDistance', 0.0)
        )


@dataclass
class Track2XPermanence:
    originalVideoPath: str
    trackingResultsPath: str
    blockColors: List[str]          # Row vector of chars (List of strings in Python)
    xPositions: List[List[float]]   # Matrix: nFrames rows x nBlocks columns (NaN where invisible)
    frameTimes_s: List[float]       # nFrames x 1
    frameNumbers: List[int]         # nFrames x 1


@dataclass
class Track3Analysis:
    track2_source_path: str

    # Metadata
    pair_colors: List[str]          # e.g., ["rg", "gr", "rg"] for neighbors

    # 1. Spacing (The "Block Differences")
    # Matrix: nFrames x (nBlocks - 1).
    # Value at [t, i] is distance between Block(i) and Block(i+1) at time t.
    spacing_matrix: List[List[float]]

    # 2. Velocity (Derived from time_per_frame)
    # Matrix: nFrames x nBlocks.
    # Value at [t, i] is instantaneous velocity of Block(i) compared to t-1.
    velocity_matrix: List[List[float]]

    # 3. Time Deltas
    # Vector: nFrames. dt[t] = time[t] - time[t-1]. (dt[0] is NaN)
    time_deltas: List[float]
