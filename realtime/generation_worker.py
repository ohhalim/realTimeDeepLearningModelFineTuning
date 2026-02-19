import threading
import time
from queue import Empty, Queue
from typing import Optional, Tuple

import torch

from models.midi_dataset import MIDITokenizer
from models.music_transformer import MusicTransformer
from realtime.types import GenerationRequest, GenerationResult


class GenerationWorker(threading.Thread):
    """
    Async generation worker.

    KV cache (Phase 1):
    - Prompt reuse cache by primer hash for consecutive similar requests.
    - Reduces repeated generation calls in chord-stable sections.
    - Tensor-level KV caching is planned in ONNX step path later.
    """

    def __init__(
        self,
        request_queue: Queue,
        result_queue: Queue,
        checkpoint_path: str,
        max_inference_step_ms: float = 60.0,
        device: Optional[str] = None,
        cache_enabled: bool = True,
    ):
        super().__init__(name="generation-worker", daemon=True)
        self.request_queue = request_queue
        self.result_queue = result_queue
        self.checkpoint_path = checkpoint_path
        self.max_inference_step_ms = float(max_inference_step_ms)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_enabled = cache_enabled
        self.stop_event = threading.Event()

        self.model, self.tokenizer = self._load_model_and_tokenizer()
        self._cache_key: Optional[Tuple[int, ...]] = None
        self._cache_generated: Optional[list] = None

    def _load_model_and_tokenizer(self):
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        config = checkpoint["config"]
        model = MusicTransformer(
            vocab_size=config["model"]["vocab_size"],
            d_model=config["model"]["d_model"],
            num_heads=config["model"]["num_heads"],
            num_layers=config["model"]["num_layers"],
            d_ff=config["model"]["d_ff"],
            max_seq_len=config["data"]["max_length"],
            dropout=0.0,
        ).to(self.device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        tokenizer = MIDITokenizer(vocab_size=config["model"]["vocab_size"])
        return model, tokenizer

    def stop(self) -> None:
        self.stop_event.set()

    def _make_cache_key(self, primer_tokens: list) -> Tuple[int, ...]:
        return tuple(primer_tokens[-128:])

    def _generate(self, req: GenerationRequest) -> Tuple[list, bool]:
        cache_hit = False
        if self.cache_enabled:
            key = self._make_cache_key(req.primer_tokens)
            if self._cache_key == key and self._cache_generated is not None:
                if len(self._cache_generated) >= req.max_length:
                    return self._cache_generated[: req.max_length], True

        generated = self.model.generate(
            primer=req.primer_tokens,
            max_length=req.max_length,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            device=self.device,
            stop_at_eos=True,
            context_window=req.context_window,
            eos_token_id=self.tokenizer.eos_token,
        ).tolist()

        if self.cache_enabled:
            self._cache_key = self._make_cache_key(req.primer_tokens)
            self._cache_generated = generated
        return generated, cache_hit

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                req: GenerationRequest = self.request_queue.get(timeout=0.05)
            except Empty:
                continue

            start_event = torch.cuda.Event(enable_timing=True) if self.device == "cuda" else None
            end_event = torch.cuda.Event(enable_timing=True) if self.device == "cuda" else None
            start_ms = req.created_at_ms

            try:
                wall_start = time.perf_counter()
                if start_event is not None:
                    start_event.record()

                generated, cache_hit = self._generate(req)

                if end_event is not None:
                    end_event.record()
                    torch.cuda.synchronize()
                    inference_ms = float(start_event.elapsed_time(end_event))
                else:
                    inference_ms = max(0.0, (time.perf_counter() - wall_start) * 1000.0)

                continuation = generated[len(req.primer_tokens) :]
                status = "ok"
                if inference_ms > self.max_inference_step_ms:
                    status = "slow"

                result = GenerationResult(
                    request_id=req.request_id,
                    created_at_ms=req.created_at_ms,
                    completed_at_ms=req.created_at_ms + inference_ms,
                    status=status,
                    role=req.role,
                    tempo_bpm=req.tempo_bpm,
                    primer_tokens=req.primer_tokens,
                    generated_tokens=generated,
                    continuation_tokens=continuation,
                    cache_hit=cache_hit,
                    inference_ms=inference_ms,
                )
                self.result_queue.put(result)
            except Exception as exc:
                result = GenerationResult(
                    request_id=req.request_id,
                    created_at_ms=start_ms,
                    completed_at_ms=start_ms,
                    status="error",
                    role=req.role,
                    tempo_bpm=req.tempo_bpm,
                    primer_tokens=req.primer_tokens,
                    generated_tokens=[],
                    continuation_tokens=[],
                    cache_hit=False,
                    inference_ms=0.0,
                    error=str(exc),
                )
                self.result_queue.put(result)
