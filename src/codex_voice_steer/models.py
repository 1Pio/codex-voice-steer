from __future__ import annotations


STT_MODELS = [
    {
        "id": "macparakeet-parakeet",
        "engine": "macparakeet",
        "label": "MacParakeet Parakeet",
        "default": True,
        "quality": "high",
        "latency": "very-low",
        "languages": "english+european",
        "command": "macparakeet-cli",
    },
    {
        "id": "macparakeet-whisper",
        "engine": "macparakeet",
        "label": "MacParakeet WhisperKit",
        "quality": "high",
        "latency": "medium",
        "languages": "multilingual",
    },
    {
        "id": "mlx-whisper-large-v3-turbo",
        "engine": "mlx-whisper",
        "repo": "mlx-community/whisper-large-v3-turbo",
        "quality": "good",
        "latency": "medium-low",
    },
    {
        "id": "mlx-whisper-large-v3",
        "engine": "mlx-whisper",
        "repo": "mlx-community/whisper-large-v3-mlx",
        "quality": "highest",
        "latency": "medium-high",
    },
]


def render_models() -> str:
    lines = ["Compatible STT models:"]
    for model in STT_MODELS:
        suffix = " (default)" if model.get("default") else ""
        lines.append(f"- {model['id']}{suffix}")
        lines.append(f"  engine: {model['engine']}")
        if "repo" in model:
            lines.append(f"  repo: {model['repo']}")
        if "command" in model:
            lines.append(f"  command: {model['command']}")
        lines.append(f"  quality: {model.get('quality', 'unknown')}  latency: {model.get('latency', 'unknown')}")
    return "\n".join(lines)
