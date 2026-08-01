"""Optional Florence image captioning adapter."""

from __future__ import annotations

import torch

from ..config import CaptionModelConfig


class FlorenceCaptioner:
    """Thin, configurable wrapper around Florence's caption task."""

    def __init__(self, config: CaptionModelConfig, device: str = "auto") -> None:
        from transformers import AutoModelForCausalLM, AutoProcessor

        resolved_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
        if resolved_device == "auto":
            resolved_device = "cpu"
        self.device = resolved_device
        self.config = config
        kwargs = {
            "revision": config.revision,
            "local_files_only": config.local_files_only,
            "trust_remote_code": True,
        }
        self.processor = AutoProcessor.from_pretrained(config.name, **kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(config.name, **kwargs).to(self.device)
        self.model.eval()

    def caption(self, images: list, task: str = "<MORE_DETAILED_CAPTION>") -> list[str]:
        results: list[str] = []
        for start in range(0, len(images), self.config.batch_size):
            batch = images[start : start + self.config.batch_size]
            for image in batch:
                inputs = self.processor(text=task, images=image.convert("RGB"), return_tensors="pt")
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                with torch.inference_mode():
                    generated = self.model.generate(
                        **inputs,
                        max_new_tokens=self.config.max_new_tokens,
                        do_sample=False,
                    )
                text = self.processor.batch_decode(generated, skip_special_tokens=False)[0]
                parsed = self.processor.post_process_generation(
                    text, task=task, image_size=image.size
                )
                value = parsed.get(task, parsed) if isinstance(parsed, dict) else parsed
                results.append(str(value).strip())
        return results
