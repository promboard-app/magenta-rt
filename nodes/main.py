"""promboard.magenta-rt — Magenta RealTime 2(Google DeepMind) 연주곡 노드.

Google 공식 구현은 JAX(리눅스 · TPU) · MLX(맥)이라, 원본과 토큰 단위로 같다고 검증된 PyTorch 판
(magenta-community/magenta-realtime-2, Apache-2.0)의 모델 코드를 nodes/mrt2 로 옮겨 쓴다(NOTICE). 실시간 스트리밍 대신
'길이를 정해 한 번에 만들기'다 — 같은 상태(state)를 이어 받으며 2초씩 만들고, 그 사이마다 진행률 · 취소를 확인한다.
스타일 글은 MusicCoCa(TorchScript)로 12개 토큰이 된다.
"""
from __future__ import annotations

import time

FPS = 25  # 한 프레임 = 40ms
CHUNK_FRAMES = 50  # 2초씩
MODELS = {
    "small": ("magenta-community/magenta-realtime-2-small", "config.json"),
    "base": ("magenta-community/magenta-realtime-2", "config.json"),
}
COCA = "magenta-community/magenta-rt-musiccoca-torch"
OOM_HELP = "그래픽카드 메모리가 모자랍니다. 판을 'small' 로 바꾸거나, 다른 프로그램이 그래픽카드를 쓰고 있으면 닫아 주세요."


def make_music(ctx, inputs, settings):
    import os

    import numpy as np
    import soundfile as sf
    import torch

    from .mrt2.modeling_magenta_rt2 import MagentaRT2ForConditionalGeneration
    from .mrt2.processing_musiccoca import MusicCoCaProcessor

    style = str(inputs.get("style") or "").strip()
    if not style:
        raise ValueError("스타일 글을 이어 주세요(예: lo-fi hip hop, mellow piano).")
    seconds = int(settings["duration"])
    temperature = float(settings["temperature"])
    cfg = float(settings["cfg"])
    seed = int(settings["seed"])
    repo, cfg_file = MODELS[str(settings["model"])]
    device = ctx.device
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    started = time.monotonic()

    model = None
    try:
        ctx.progress(0.02, "load-model")
        model_dir = os.path.dirname(ctx.model_path(f"{repo}/{cfg_file}"))
        model = MagentaRT2ForConditionalGeneration.from_pretrained(model_dir, dtype=dtype).to(device).eval()
        coca_dir = os.path.dirname(ctx.model_path(f"{COCA}/text_encoder.pt"))
        model.set_processor(MusicCoCaProcessor.from_pretrained(coca_dir, device=device))
        ctx.log(f"{settings['model']} loaded on {device} in {time.monotonic() - started:.1f}s")
        ctx.check_cancel()

        # 스타일 글 → 12개 토큰은 한 번만 만든다(조각마다 다시 만들지 않게).
        style_tokens = model.processor(style)
        total = seconds * FPS
        made = 0
        state = None
        parts = []
        gen_started = time.monotonic()
        while made < total:
            n = min(CHUNK_FRAMES, total - made)
            last = made + n >= total
            audio, state = model.generate(style=style_tokens, frames=n, temperature=temperature, cfg_musiccoca=cfg,
                                          seed=seed, state=state, flush=last)
            parts.append(audio)
            made += n
            ctx.check_cancel()
            ctx.progress(0.05 + 0.93 * made / total, "compose")
        ctx.log(f"{total} frames in {time.monotonic() - gen_started:.1f}s")
    except Exception as exc:  # noqa: BLE001
        if isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower():
            raise RuntimeError(OOM_HELP) from exc
        raise
    finally:
        if model is not None:
            del model
        if device == "cuda":
            torch.cuda.empty_cache()

    wav = np.concatenate(parts, axis=0) if parts else np.zeros((0, 2), np.float32)
    dst = ctx.out_path("music.wav")
    sf.write(dst, wav, 48000, subtype="PCM_16")
    ctx.log(f"music {wav.shape[0] / 48000:.1f}s done in {time.monotonic() - started:.1f}s")
    ctx.progress(1.0, "done")
    return {"audio": dst}


NODES = {"music": make_music}
