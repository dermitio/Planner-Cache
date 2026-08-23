"""Actual Pythia and llama.cpp runtime adapters for interactive Planner Cache."""

from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import tempfile
import time
from typing import Callable
from urllib.request import Request, urlopen

import torch
import torch.nn.functional as F

from pcm.planner.canonical import CanonicalPStore
from pcm.planner.interactive_session import (
    CanonicalQueryIntent,
    RELATION_NAMES,
    file_sha256,
)
from pcm.planner.compatibility import (
    LexicalTranslationLayer,
    TensorTranslationLayer,
    tokenizer_bundle_checksum,
)
from pcm.planner.personality import merge_active_personality_with_p_cache
from pcm.planner.pythia_split_translate import PythiaSplitTranslatedModel
from pcm.planner.split_translator import (
    ByteEntityEncoder,
    CanonicalPRouter,
    FactorizedCanonicalQuery,
)


Emit = Callable[..., object]


@dataclass(frozen=True)
class GenerationResult:
    text: str
    latency_seconds: float
    input_tokens: int | None
    output_tokens: int | None
    diagnostics: dict[str, object]


def merge_runtime_state(
    p_cache: CanonicalPStore,
    personality: CanonicalPStore | None,
) -> CanonicalPStore:
    if personality is None:
        return p_cache
    merged = merge_active_personality_with_p_cache(p_cache, personality)
    source_surfaces = getattr(p_cache, "_pcm_value_surfaces", {})
    if source_surfaces:
        merged_surfaces = {}
        for merged_index in merged.valid.nonzero(as_tuple=False).flatten().tolist():
            for source_index, surface in source_surfaces.items():
                if (
                    int(merged.entity_id[merged_index]) == int(p_cache.entity_id[source_index])
                    and int(merged.relation_id[merged_index]) == int(p_cache.relation_id[source_index])
                    and int(merged.value_id[merged_index]) == int(p_cache.value_id[source_index])
                ):
                    merged_surfaces[merged_index] = surface
                    break
        merged._pcm_value_surfaces = merged_surfaces
    return merged


def _entry(store: CanonicalPStore, index: int) -> dict[str, object]:
    return {
        "slot_id": index,
        "entity": store.cache.labels[index],
        "entity_id": int(store.entity_id[index]),
        "relation_id": int(store.relation_id[index]),
        "relation": (
            RELATION_NAMES[int(store.relation_id[index])]
            if 0 <= int(store.relation_id[index]) < len(RELATION_NAMES) else None
        ),
        "value_id": int(store.value_id[index]),
        "metadata_id": int(store.canonical_metadata_id[index]),
    }


def canonical_route(
    router: CanonicalPRouter,
    encoder: ByteEntityEncoder,
    store: CanonicalPStore,
    query: CanonicalQueryIntent,
) -> tuple[object | None, list[dict[str, object]]]:
    if not query.entity or query.relation_id is None or store.cache.occupied == 0:
        return None, []
    relation_logits = torch.full((1, router.config.relation_count), -12.0)
    relation_logits[0, query.relation_id] = 12.0
    metadata_logits = torch.full((1, router.config.metadata_count), -12.0)
    metadata_logits[0, 0] = 12.0
    factorized = FactorizedCanonicalQuery(
        entity=encoder([query.entity]),
        relation_logits=relation_logits,
        metadata_logits=metadata_logits,
    )
    index = router.build_index(store, encoder, device="cpu")
    scores, _features = router.all_scores(factorized, index)
    candidates = []
    for slot in torch.argsort(scores[0], descending=True).tolist():
        score = float(scores[0, slot].detach())
        if not math.isfinite(score):
            continue
        candidates.append({**_entry(store, slot), "score": score})
        if len(candidates) == 8:
            break
    return router.route(factorized, index, top_k=1), candidates


class PythiaInteractiveRuntime:
    name = "pythia"
    runtime = "transformers-gpt-neox"

    def __init__(
        self,
        *,
        model_path: Path,
        adapter_path: Path,
        router_path: Path,
        max_context_tokens: int = 1024,
        max_new_tokens: int = 96,
        temperature: float = 0.7,
        top_p: float = 0.9,
        seed: int = 1234,
        review_model_path: Path | None = None,
        review_llama_cpp_dir: Path | None = None,
        review_gpu_layers: int = 0,
        review_pid_file: Path | None = None,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.utils import logging as transformers_logging

        if not torch.cuda.is_available():
            raise RuntimeError("Pythia interactive runtime requires CUDA")
        transformers_logging.set_verbosity_error()
        transformers_logging.disable_progress_bar()
        self.model_path = model_path.resolve()
        self.adapter_path = adapter_path.resolve()
        self.router_path = router_path.resolve()
        self.max_context_tokens = max_context_tokens
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.base = AutoModelForCausalLM.from_pretrained(
            self.model_path, local_files_only=True, dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to("cuda").eval()
        self.package = TensorTranslationLayer.load(
            self.adapter_path, device="cuda", dtype=torch.float32,
        ).eval()
        self.router = CanonicalPRouter.load(self.router_path, device="cuda").eval()
        self.encoder = ByteEntityEncoder(128)
        self.wrapper = PythiaSplitTranslatedModel(
            self.base, self.package, self.router, self.encoder,
        ).to("cuda").eval()
        self.generator = torch.Generator(device="cuda")
        self.generator.manual_seed(seed)
        self.review_model_path = (
            None if review_model_path is None else review_model_path.resolve()
        )
        self.review_server = None
        if self.review_model_path is not None:
            assert review_llama_cpp_dir is not None
            self.review_server = LlamaServerProcess(
                binary=(
                    review_llama_cpp_dir.resolve() / "build/bin/llama-server"
                ),
                model=self.review_model_path,
                control_vector=self.review_model_path.with_suffix(".unused-review.gguf"),
                attachment_layer=0,
                context_tokens=2048,
                gpu_layers=review_gpu_layers,
                threads=8,
                pid_file=review_pid_file,
                startup_timeout=180,
            )

    @property
    def model_id(self) -> str:
        return self.package.config.model_id

    def metadata(self) -> dict[str, object]:
        import transformers

        return {
            "model": self.model_id,
            "model_path": str(self.model_path),
            "runtime": self.runtime,
            "runtime_version": transformers.__version__,
            "compatibility_layer": "ttl",
            "compatibility_support_level": "semantic/internal",
            "adapter_artifact": str(self.adapter_path),
            "adapter_checksum": file_sha256(self.adapter_path),
            "adapter_format": "planner-cache-ttl-v1",
            "adapter_attachment_layers": list(self.package.config.attachment_layers),
            "router_artifact": str(self.router_path),
            "router_checksum": file_sha256(self.router_path),
            "router_format": self.router.config.format,
            "recent_kv_context_tokens": self.max_context_tokens,
            "generation": {
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "seed": self.seed,
            },
            "memory_reviewer": {
                "mode": (
                    "same-frozen-pythia"
                    if self.review_model_path is None
                    else "separate-frozen-llama-json-reviewer"
                ),
                "model_path": (
                    None if self.review_model_path is None
                    else str(self.review_model_path)
                ),
                "ttl_or_ltl_active": False,
            },
        }

    def _prompt(self, history: list[tuple[str, str]], message: str) -> str:
        rows = []
        for user, assistant in history:
            rows.extend((f"User: {user}", f"Assistant: {assistant}"))
        rows.extend((f"User: {message}", "Assistant:"))
        text = "\n".join(rows)
        tokens = self.tokenizer(text, add_special_tokens=False).input_ids
        input_limit = max(16, self.max_context_tokens - self.max_new_tokens)
        if len(tokens) > input_limit:
            tokens = tokens[-input_limit:]
            text = self.tokenizer.decode(tokens, skip_special_tokens=True)
        return text

    def _sample(self, logits: torch.Tensor) -> torch.Tensor:
        if self.temperature <= 0:
            return logits.argmax(-1, keepdim=True)
        probabilities = F.softmax(logits.float() / self.temperature, dim=-1)
        sorted_probabilities, sorted_indices = probabilities.sort(descending=True)
        cumulative = sorted_probabilities.cumsum(-1)
        remove = cumulative - sorted_probabilities > self.top_p
        sorted_probabilities = sorted_probabilities.masked_fill(remove, 0)
        sorted_probabilities /= sorted_probabilities.sum(-1, keepdim=True)
        sampled = torch.multinomial(
            sorted_probabilities, 1, generator=self.generator,
        )
        return sorted_indices.gather(-1, sampled)

    def _emit_step(
        self,
        emit: Emit | None,
        store: CanonicalPStore,
        query: CanonicalQueryIntent,
        generation_step: int,
    ) -> dict[str, object]:
        diagnostics: dict[str, object] = {
            "attachment_layers": list(self.package.config.attachment_layers),
            "query_entity": query.entity,
        }
        if not self.wrapper.route_telemetry or not self.wrapper.query_telemetry:
            if emit:
                emit(
                    "ROUTER_QUERY", source="model_to_p", entity=query.entity,
                    requested_relation=query.relation, reason=query.reason,
                    generation_step=generation_step,
                )
                emit(
                    "ROUTER_CANDIDATES", source="p_cache", candidates=[],
                    generation_step=generation_step,
                )
                emit(
                    "ROUTER_REJECT", source="p_cache", reason="no active routable state",
                    entity=query.entity, relation=query.relation,
                    generation_step=generation_step,
                )
                emit(
                    "TTL_DISABLE", source="ttl", reason="router did not run",
                    generation_step=generation_step,
                )
            diagnostics.update({"router_accepted": False, "gate": 0.0})
            return diagnostics
        projected = self.wrapper.query_telemetry[-1]
        route = self.wrapper.route_telemetry[-1]
        relation_probability = F.softmax(projected.relation_logits[0, -1].float(), dim=-1)
        metadata_probability = F.softmax(projected.metadata_logits[0, -1].float(), dim=-1)
        index = int(route.indices[0, -1, 0])
        score = float(route.scores[0, -1, 0])
        accepted = bool(route.accepted[0, -1])
        selected = _entry(store, index) if route.has_valid else None
        gate = 0.0
        if self.wrapper.gate_telemetry:
            gate = float(self.wrapper.gate_telemetry[-1][0, -1])
        if emit:
            emit(
                "ROUTER_QUERY", source="model_to_p", entity=query.entity,
                requested_relation=query.relation,
                projected_relation_probabilities=relation_probability.tolist(),
                projected_metadata_probabilities=metadata_probability.tolist(),
                entity_anchor="tokenizer-independent-byte-anchor" if query.entity else "learned-hidden-query",
                generation_step=generation_step,
            )
            emit(
                "ROUTER_CANDIDATES", source="p_cache",
                candidates=[] if selected is None else [{**selected, "score": score}],
                generation_step=generation_step,
            )
            emit(
                "ROUTER_ACCEPT" if accepted else "ROUTER_REJECT",
                source="p_cache", score=score, selected_state=selected,
                generation_step=generation_step,
            )
            emit(
                "TTL_ENABLE" if accepted and gate > 0 else "TTL_DISABLE",
                source="ttl", gate=gate,
                attachment_layers=list(self.package.config.attachment_layers),
                selected_state=selected,
                generation_step=generation_step,
            )
            emit(
                "TTL_OUTPUT", source="ttl", gate=gate,
                active=accepted and gate > 0, selected_state=selected,
                generation_step=generation_step,
            )
        diagnostics.update({
            "router_accepted": accepted,
            "router_score": score,
            "selected_state": selected,
            "gate": gate,
            "projected_relation_probabilities": relation_probability.tolist(),
        })
        return diagnostics

    def generate(
        self,
        message: str,
        history: list[tuple[str, str]],
        p_cache: CanonicalPStore,
        personality: CanonicalPStore | None,
        query: CanonicalQueryIntent,
        *,
        emit: Emit | None = None,
        raw_messages: list[dict[str, object]] | None = None,
    ) -> GenerationResult:
        prompt = self._prompt(history, message)
        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = encoded.input_ids.to("cuda")
        attention_mask = encoded.attention_mask.to("cuda")
        input_count = int(input_ids.shape[1])
        active_store = merge_runtime_state(p_cache, personality)
        use_store = active_store if query.entity is not None else None
        generated: list[int] = []
        past = None
        started = time.perf_counter()
        diagnostics: dict[str, object] = {
            "generation_steps": [],
            "recent_context_turns": len(history),
            "recent_context_characters": len(prompt),
            "recent_context_input_tokens": input_count,
        }
        with torch.inference_mode():
            for step in range(self.max_new_tokens):
                current = input_ids if past is None else input_ids[:, -1:]
                output = self.wrapper(
                    input_ids=current,
                    attention_mask=attention_mask,
                    past_key_values=past,
                    p_store=use_store,
                    query_entity_surfaces=[query.entity] if query.entity else None,
                    collect_telemetry=emit is not None,
                    use_cache=True,
                )
                step_diagnostics = self._emit_step(
                    emit, active_store, query, generation_step=step,
                )
                diagnostics["generation_steps"].append(step_diagnostics)
                next_token = self._sample(output.logits[:, -1])
                token_id = int(next_token[0, 0])
                generated.append(token_id)
                past = output.past_key_values
                input_ids = torch.cat((input_ids, next_token), dim=1)
                attention_mask = torch.cat((attention_mask, torch.ones_like(next_token)), dim=1)
                if token_id == self.tokenizer.eos_token_id:
                    break
                decoded = self.tokenizer.decode(generated, skip_special_tokens=True)
                if "\nUser:" in decoded or "\nuser:" in decoded:
                    break
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        text = re_split_user(text).strip()
        return GenerationResult(
            text=text,
            latency_seconds=time.perf_counter() - started,
            input_tokens=input_count,
            output_tokens=len(generated),
            diagnostics=diagnostics,
        )

    def review_memory(
        self, prompt: str, schema: dict[str, object],
    ) -> str:
        """Generate hidden review JSON with the frozen base and no TTL path."""
        if self.review_server is not None:
            self.review_server.ensure(0.0, vector_key=None)
            body = json.dumps({
                "messages": [
                    {
                        "role": "system",
                        "content": "Return only valid JSON for the supplied schema.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 256,
                "temperature": 0.0,
                "top_p": 1.0,
                "seed": self.seed,
                "stream": False,
                "response_format": {"type": "json_schema", "schema": schema},
            }, ensure_ascii=False).encode("utf-8")
            request = Request(
                f"http://127.0.0.1:{self.review_server.port}/v1/chat/completions",
                data=body, headers={"Content-Type": "application/json"},
            )
            with urlopen(request, timeout=600) as response:
                result = json.load(response)
            return str(result["choices"][0]["message"]["content"])
        del schema  # Transformers generation has no native JSON grammar.
        review_text = (
            "Complete each memory-review task with JSON only.\n"
            "Example task: user says haha okay.\n"
            "JSON: {\"operations\":[{\"op\":\"IGNORE\",\"entity\":null,"
            "\"relation\":null,\"value\":null,\"confidence\":1.0,"
            "\"source\":\"explicit_user\"}]}\n"
            "Example task: user says I put the key in the drawer.\n"
            "JSON: {\"operations\":[{\"op\":\"CREATE\",\"entity\":\"key\","
            "\"relation\":\"location\",\"value\":\"drawer\","
            "\"confidence\":1.0,\"source\":\"rp_action\"}]}\n"
            f"Memory-review task:\n{prompt}\nJSON:"
        )
        encoded = self.tokenizer(
            review_text, return_tensors="pt", add_special_tokens=False,
        )
        input_limit = self.max_context_tokens - 160
        input_ids = encoded.input_ids
        if input_ids.shape[1] > input_limit:
            prefix_tokens = min(240, input_limit // 3)
            input_ids = torch.cat((
                input_ids[:, :prefix_tokens],
                input_ids[:, -(input_limit - prefix_tokens):],
            ), dim=1)
        input_ids = input_ids.to("cuda")
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            output = self.base.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=160,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(
            output[0, input_ids.shape[1]:], skip_special_tokens=True,
        )

    def close(self) -> None:
        if self.review_server is not None:
            self.review_server.stop()
        self.wrapper.close()


def re_split_user(text: str) -> str:
    for marker in ("\nUser:", "\nuser:"):
        if marker in text:
            return text.split(marker, 1)[0]
    return text


def gemma_chat_request_body(
    messages: list[dict[str, object]],
    *,
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    target_token_ids: list[int] | tuple[int, ...] = (),
) -> dict[str, object]:
    """Build a native request without altering the chat message structure."""
    body: dict[str, object] = {
        "messages": copy.deepcopy(messages),
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "stream": False,
    }
    if target_token_ids:
        body["logit_bias"] = {
            str(token_id): 100.0 for token_id in target_token_ids
        }
    return body


class LlamaServerProcess:
    """Managed llama-server that can switch inert and control-vector modes."""

    def __init__(
        self,
        *,
        binary: Path,
        model: Path,
        control_vector: Path,
        attachment_layer: int,
        context_tokens: int,
        gpu_layers: int,
        threads: int,
        startup_timeout: float = 90.0,
        pid_file: Path | None = None,
        popen_factory=subprocess.Popen,
    ) -> None:
        self.binary = binary
        self.model = model
        self.control_vector = control_vector
        self.attachment_layer = attachment_layer
        self.context_tokens = context_tokens
        self.gpu_layers = gpu_layers
        self.threads = threads
        self.startup_timeout = startup_timeout
        self.pid_file = pid_file
        self.popen_factory = popen_factory
        self.process = None
        self.port: int | None = None
        self.strength: float | None = None
        self.vector_key: str | None = None

    def command(self, port: int, strength: float) -> list[str]:
        command = [
            str(self.binary), "--model", str(self.model),
            "--host", "127.0.0.1", "--port", str(port),
            "--ctx-size", str(self.context_tokens), "--parallel", "1",
            "--threads", str(self.threads), "--threads-batch", str(self.threads),
            "--gpu-layers", str(self.gpu_layers), "--no-warmup",
            "--no-context-shift", "--log-disable", "--reasoning", "off",
        ]
        if strength != 0:
            command.extend((
                "--control-vector-scaled", f"{self.control_vector}:{strength}",
                "--control-vector-layer-range", str(self.attachment_layer),
                str(self.attachment_layer),
            ))
        return command

    def ensure(self, strength: float, *, vector_key: str | None = None) -> dict[str, object]:
        if (
            self.process is not None
            and self.process.poll() is None
            and self.strength == strength
            and self.vector_key == vector_key
        ):
            return {"restarted": False, "port": self.port, "strength": strength}
        self.stop()
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        command = self.command(port, strength)
        self.process = self.popen_factory(
            command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        self.port = port
        self.strength = strength
        self.vector_key = vector_key
        if self.pid_file is not None:
            self.pid_file.write_text(str(self.process.pid) + "\n")
        deadline = time.monotonic() + self.startup_timeout
        while True:
            if self.process.poll() is not None:
                self.process = None
                if self.pid_file is not None:
                    self.pid_file.unlink(missing_ok=True)
                raise RuntimeError("llama-server exited before becoming ready")
            try:
                with urlopen(f"http://127.0.0.1:{port}/health", timeout=1):
                    break
            except Exception:
                if time.monotonic() >= deadline:
                    self.stop()
                    raise TimeoutError("llama-server did not become ready")
                time.sleep(0.25)
        return {"restarted": True, "port": port, "strength": strength, "command": command}

    def stop(self) -> None:
        process = self.process
        self.process = None
        self.port = None
        self.strength = None
        self.vector_key = None
        if self.pid_file is not None:
            self.pid_file.unlink(missing_ok=True)
        if process is None or process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=15)


class GemmaInteractiveRuntime:
    name = "gemma"
    runtime = "llama.cpp-server-gguf"

    def __init__(
        self,
        *,
        model_path: Path,
        adapter_path: Path,
        router_path: Path,
        llama_cpp_dir: Path,
        tokenizer_bundle: Path,
        max_context_tokens: int = 4096,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        seed: int = 1234,
        gpu_layers: int = 12,
        threads: int = 8,
        pid_file: Path | None = None,
    ) -> None:
        self.model_path = model_path.resolve()
        self.adapter_path = adapter_path.resolve()
        self.router_path = router_path.resolve()
        self.llama_cpp_dir = llama_cpp_dir.resolve()
        self.tokenizer_bundle = tokenizer_bundle.resolve()
        self.server_binary = self.llama_cpp_dir / "build/bin/llama-server"
        self.cli_binary = self.llama_cpp_dir / "build/bin/llama-cli"
        self.supports_open_values = True
        self.package = LexicalTranslationLayer.load(self.adapter_path)
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_bundle, local_files_only=True,
        )
        self.tokenizer_bundle_sha256 = tokenizer_bundle_checksum(self.tokenizer_bundle)
        self.router = CanonicalPRouter.load(self.router_path)
        self.encoder = ByteEntityEncoder(128)
        self.max_context_tokens = max_context_tokens
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self.model_sha256 = file_sha256(self.model_path)
        version = subprocess.run(
            [str(self.cli_binary), "--version"], check=True, text=True,
            capture_output=True,
        )
        self.llama_version = "\n".join(
            part.strip() for part in (version.stdout, version.stderr) if part.strip()
        )
        self.package.validate_compatibility(
            model_id=self.package.config.model_id,
            model_architecture=self.package.config.model_architecture,
            model_sha256=self.model_sha256,
            runtime="llama.cpp",
            runtime_version=self.llama_version,
            tokenizer_bundle_sha256=self.tokenizer_bundle_sha256,
        )
        self._temporary = tempfile.TemporaryDirectory(prefix="planner-cache-gemma-chat-")
        self.control_vector = Path(self._temporary.name) / "unused-ltl-control-vector.gguf"
        self.server = LlamaServerProcess(
            binary=self.server_binary, model=self.model_path,
            control_vector=self.control_vector,
            attachment_layer=0,
            context_tokens=max_context_tokens, gpu_layers=gpu_layers, threads=threads,
            pid_file=pid_file,
        )

    @property
    def model_id(self) -> str:
        return self.package.config.model_id

    def metadata(self) -> dict[str, object]:
        return {
            "model": self.model_id,
            "model_path": str(self.model_path),
            "model_gguf_sha256": self.model_sha256,
            "runtime": self.runtime,
            "runtime_version": self.llama_version,
            "llama_cpp_dir": str(self.llama_cpp_dir),
            "compatibility_layer": "ltl",
            "compatibility_support_level": "lexical/output",
            "adapter_artifact": str(self.adapter_path),
            "adapter_checksum": file_sha256(self.adapter_path),
            "adapter_format": self.package.config.format,
            "adapter_attachment_layers": [],
            "lexical_control": self.package.config.control,
            "tokenizer_bundle": str(self.tokenizer_bundle),
            "tokenizer_bundle_sha256": self.tokenizer_bundle_sha256,
            "adapter_parameter_count": self.package.config.parameter_count,
            "router_artifact": str(self.router_path),
            "router_checksum": file_sha256(self.router_path),
            "router_format": self.router.config.format,
            "recent_kv_context_tokens": self.max_context_tokens,
            "generation": {
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "seed": self.seed,
            },
            "memory_reviewer": {
                "mode": "same-frozen-gemma",
                "model_path": str(self.model_path),
                "ttl_or_ltl_active": False,
            },
        }

    def _route_and_translate(
        self,
        store: CanonicalPStore,
        query: CanonicalQueryIntent,
        emit: Emit | None,
    ) -> tuple[str | None, dict[str, object]]:
        if emit:
            emit(
                "ROUTER_QUERY", source="canonical_query", entity=query.entity,
                relation=query.relation, reason=query.reason,
            )
        route, candidates = canonical_route(self.router, self.encoder, store, query)
        if emit:
            emit("ROUTER_CANDIDATES", source="p_cache", candidates=candidates)
        if route is None or not route.has_valid:
            if emit:
                emit("ROUTER_REJECT", source="p_cache", reason="incomplete query or empty state")
                emit("LTL_DISABLE", source="ltl", gate=0.0, runtime_inert=True)
            return None, {"router_accepted": False, "gate": 0.0, "runtime_inert": True}
        index = int(route.indices[0, 0])
        score = float(route.scores[0, 0].detach())
        accepted = bool(route.accepted[0])
        selected = _entry(store, index)
        if emit:
            emit(
                "ROUTER_ACCEPT" if accepted else "ROUTER_REJECT",
                source="p_cache", score=score, selected_state=selected,
            )
        value_surface = getattr(store, "_pcm_value_surfaces", {}).get(index)
        target = self.package.target(str(value_surface or ""), route_accepted=accepted)
        active = target is not None
        gate = 1.0 if active else 0.0
        if emit:
            emit(
                "LTL_ENABLE" if active else "LTL_DISABLE",
                source="ltl", gate=gate,
                selected_state=selected, value_surface=value_surface,
                lexical_control=self.package.config.control, runtime_inert=not active,
            )
        token_targets = self.package.token_targets(
            str(value_surface or ""), self.tokenizer, route_accepted=accepted,
        )
        if emit and token_targets:
            emit(
                "LTL_TOKEN_TARGET", source="ltl", target_text=target,
                target_token_ids=list(token_targets),
                strategy=self.package.config.control,
                model_gguf_sha256=self.model_sha256,
                llama_cpp_version=self.llama_version,
            )
        return target, {
            "router_accepted": accepted,
            "router_score": score,
            "selected_state": selected,
            "gate": gate,
            "runtime_inert": not active,
            "value_surface": value_surface,
            "compatibility_layer": "ltl",
            "lexical_target": target,
            "lexical_target_token_ids": list(token_targets),
        }

    def generate(
        self,
        message: str,
        history: list[tuple[str, str]],
        p_cache: CanonicalPStore,
        personality: CanonicalPStore | None,
        query: CanonicalQueryIntent,
        *,
        emit: Emit | None = None,
        raw_messages: list[dict[str, object]] | None = None,
    ) -> GenerationResult:
        active_store = merge_runtime_state(p_cache, personality)
        target, diagnostics = self._route_and_translate(active_store, query, emit)
        server_started = time.perf_counter()
        server_state = self.server.ensure(0.0, vector_key=None)
        diagnostics["server_restart_latency_seconds"] = time.perf_counter() - server_started
        diagnostics["server_restarted"] = server_state["restarted"]
        if raw_messages is not None:
            # Browser messages remain an opaque native llama.cpp input. Planner
            # Cache observes the newest user text separately and never rewrites
            # this structure for memory, routing, or compatibility handling.
            messages = copy.deepcopy(raw_messages)
            diagnostics["message_path"] = "native-structured-passthrough"
            diagnostics["recent_context_turns"] = sum(
                1 for row in messages if row.get("role") == "user"
            )
            diagnostics["native_message_sha256"] = hashlib.sha256(
                json.dumps(
                    messages, ensure_ascii=False, sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
        else:
            # The legacy terminal front end has no structured message array.
            current = {"role": "user", "content": message}
            character_budget = max(
                512, (self.max_context_tokens - self.max_new_tokens) * 3
            )
            retained: list[tuple[str, str]] = []
            used = len(message)
            for user, assistant in reversed(history):
                cost = len(user) + len(assistant) + 32
                if used + cost > character_budget:
                    break
                retained.append((user, assistant))
                used += cost
            messages = []
            for user, assistant in reversed(retained):
                messages.extend((
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": assistant},
                ))
            messages.append(current)
            diagnostics["message_path"] = "terminal-history"
            diagnostics["recent_context_turns"] = len(retained)
            diagnostics["recent_context_characters"] = used
        target_ids = diagnostics.get("lexical_target_token_ids", [])
        request_body = gemma_chat_request_body(
            messages,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            seed=self.seed,
            target_token_ids=target_ids,
        )
        body = json.dumps(request_body).encode()
        started = time.perf_counter()
        request = Request(
            f"http://127.0.0.1:{self.server.port}/v1/chat/completions",
            data=body, headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=300) as response:
            result = json.load(response)
        usage = result.get("usage", {})
        # Preserve llama.cpp's assistant content exactly. Trimming here would
        # alter the assistant message that the Web UI returns on the next turn.
        text = str(result["choices"][0]["message"]["content"])
        if emit and target is not None:
            observed = target.casefold() in text.casefold()
            if observed:
                emit("LTL_COMPLETE", source="ltl", target_text=target)
            else:
                emit(
                    "LTL_DISABLE", source="ltl", target_text=target,
                    reason="target sequence was not completed",
                )
        return GenerationResult(
            text=text,
            latency_seconds=time.perf_counter() - started,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            diagnostics=diagnostics,
        )

    def review_memory(
        self, prompt: str, schema: dict[str, object],
    ) -> str:
        """Run a separate grammar-constrained review without LTL controls."""
        self.server.ensure(0.0, vector_key=None)
        body = json.dumps({
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Return only valid JSON for the supplied schema. "
                        "This is a hidden memory review, not a user-visible reply."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 256,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": self.seed,
            "stream": False,
            "response_format": {
                "type": "json_schema",
                "schema": schema,
            },
        }, ensure_ascii=False).encode("utf-8")
        request = Request(
            f"http://127.0.0.1:{self.server.port}/v1/chat/completions",
            data=body, headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=300) as response:
            result = json.load(response)
        return str(result["choices"][0]["message"]["content"])

    def close(self) -> None:
        self.server.stop()
        self._temporary.cleanup()
