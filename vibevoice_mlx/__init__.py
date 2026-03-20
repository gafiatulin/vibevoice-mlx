"""VibeVoice MLX — text-to-speech inference on Apple Silicon."""

from .generate import GenerationMetrics, GenerationOptions, generate
from .load_weights import load_model, resolve_model_path
from .model import VibeVoiceConfig, VibeVoiceModel
