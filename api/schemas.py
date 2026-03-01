"""Pydantic response models for the DeepShield API."""

from typing import List, Optional
from pydantic import BaseModel


class FrameResult(BaseModel):
    frame_idx: int
    label: str
    confidence: float
    prob_real: float


class PredictionResponse(BaseModel):
    label: str
    confidence: float
    prob_real: float


class VideoPredictionResponse(BaseModel):
    label: str
    confidence: float
    prob_real: float
    frames_analyzed: int
    frame_results: List[FrameResult]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
