"""Shared Planner Cache chat session and terminal entry point."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import traceback

import torch

from pcm.planner.canonical import CANONICAL_P_PROTOCOL, CanonicalPStore
from pcm.planner.compatibility import CompatibilityKind, resolve_compatibility
from pcm.planner.interactive_runtimes import (
    GenerationResult,
    GemmaInteractiveRuntime,
    PythiaInteractiveRuntime,
    canonical_route,
)
from pcm.planner.interactive_session import (
    CanonicalStateManager,
    PersonalityManager,
    SessionRecorder,
    git_commit,
    utc_now,
)
from pcm.planner.memory_review import PostTurnMemoryReviewer
from pcm.planner.representation import train_and_probe_representation


HELP = """Chat normally by typing any message.

Commands:
  /help         show this help
  /state        show active canonical P-cache entries
  /personality  show promoted personality entries and the last retrieval
  /events       show the most recent Planner Cache events
  /save         checkpoint P-cache and P-package state
  /quit         save and exit

Memory extraction is automatic. These explicit forms are also recognized:
  The silver key belongs to Alice
  Alice owns the silver key
  The silver key is currently in Paris
  The current status of the silver key is garden
  remember: silver key.owner=Alice
  invalidate: silver key.owner

Pythia consumes accepted state through its TTL. Gemma uses an LTL for accepted
lexical values. Rejected routes remain inert and conversation continues normally.
"""


def parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[3]
    result = argparse.ArgumentParser(description="Planner Cache interactive terminal")
    result.add_argument("runtime", choices=("pythia", "gemma"))
    result.add_argument("--repo-root", type=Path, default=root)
    result.add_argument("--model", type=Path, required=True)
    result.add_argument("--adapter", type=Path, required=True)
    result.add_argument("--router", type=Path, required=True)
    result.add_argument("--llama-cpp-dir", type=Path)
    result.add_argument("--tokenizer-bundle", type=Path)
    result.add_argument("--ppkg", type=Path)
    result.add_argument("--session-root", type=Path, required=True)
    result.add_argument("--p-cache", type=Path)
    result.add_argument("--slots", type=int, default=128)
    result.add_argument("--context-tokens", type=int)
    result.add_argument("--max-new-tokens", type=int, default=96)
    result.add_argument("--temperature", type=float, default=0.7)
    result.add_argument("--top-p", type=float, default=0.9)
    result.add_argument("--seed", type=int, default=1234)
    result.add_argument("--gpu-layers", type=int, default=12)
    result.add_argument("--threads", type=int, default=8)
    result.add_argument("--llama-pid-file", type=Path)
    result.add_argument("--review-model", type=Path)
    result.add_argument("--review-llama-cpp-dir", type=Path)
    result.add_argument("--review-gpu-layers", type=int, default=0)
    result.add_argument("--review-pid-file", type=Path)
    result.add_argument("--no-logging", action="store_true")
    return result


def require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def validate_args(args: argparse.Namespace) -> None:
    if args.runtime == "pythia":
        if not args.model.is_dir():
            raise FileNotFoundError(f"pythia model not found: {args.model}")
        review_model = getattr(args, "review_model", None)
        review_llama_cpp_dir = getattr(args, "review_llama_cpp_dir", None)
        if review_model is not None:
            require_file(review_model, "structured review model")
            if review_llama_cpp_dir is None:
                raise ValueError(
                    "--review-llama-cpp-dir is required with --review-model"
                )
            require_file(
                review_llama_cpp_dir / "build/bin/llama-server",
                "review llama-server",
            )
    else:
        require_file(args.model, "gemma model")
    expected = ".ttl" if args.runtime == "pythia" else ".ltl"
    require_file(args.adapter, f"{expected} compatibility artifact")
    if args.adapter.suffix != expected:
        raise ValueError(f"{args.runtime} requires a {expected} compatibility artifact")
    resolution = resolve_compatibility(args.adapter)
    expected_kind = (
        CompatibilityKind.TTL if args.runtime == "pythia" else CompatibilityKind.LTL
    )
    if resolution.kind is not expected_kind:
        raise ValueError(
            f"{args.runtime} cannot use {resolution.kind.value} compatibility"
        )
    require_file(args.router, ".router artifact")
    if args.p_cache is not None:
        require_file(args.p_cache, "P-cache snapshot")
    if args.runtime == "gemma":
        if args.llama_cpp_dir is None:
            raise ValueError("--llama-cpp-dir is required for Gemma")
        if args.tokenizer_bundle is None or not args.tokenizer_bundle.is_dir():
            raise FileNotFoundError("--tokenizer-bundle is required for Gemma")
        require_file(args.llama_cpp_dir / "build/bin/llama-server", "llama-server")
        require_file(args.llama_cpp_dir / "build/bin/llama-cli", "llama-cli")


def build_runtime(args: argparse.Namespace):
    common = {
        "model_path": args.model,
        "adapter_path": args.adapter,
        "router_path": args.router,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
    }
    if args.runtime == "pythia":
        return PythiaInteractiveRuntime(
            **common,
            max_context_tokens=args.context_tokens or 1024,
            review_model_path=getattr(args, "review_model", None),
            review_llama_cpp_dir=getattr(args, "review_llama_cpp_dir", None),
            review_gpu_layers=getattr(args, "review_gpu_layers", 0),
            review_pid_file=getattr(args, "review_pid_file", None),
        )
    return GemmaInteractiveRuntime(
        **common,
        llama_cpp_dir=args.llama_cpp_dir,
        tokenizer_bundle=args.tokenizer_bundle,
        max_context_tokens=args.context_tokens or 4096,
        gpu_layers=args.gpu_layers,
        threads=args.threads,
        pid_file=args.llama_pid_file,
    )


def display_json(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))


def save_session_state(
    recorder: SessionRecorder,
    state: CanonicalStateManager,
    personality: PersonalityManager | None,
    *,
    reason: str,
) -> None:
    snapshot = recorder.directory / "p-cache.safetensors"
    if recorder.enabled:
        state.store.save(snapshot)
        state.save_runtime_metadata(recorder.directory / "p-cache-runtime.json")
    checksum = personality.checkpoint() if personality is not None else None
    recorder.save_event(
        reason,
        p_cache_snapshot=str(snapshot) if recorder.enabled else None,
        ppkg_checksum=checksum,
    )


def final_payload(
    state: CanonicalStateManager,
    personality: PersonalityManager | None,
) -> dict[str, object]:
    personality_page = (
        {
            "entries": [], "total_active": 0, "returned": 0,
            "limit": 100, "offset": 0, "truncated": False,
        }
        if personality is None
        else personality.visible_entry_page(limit=100)
    )
    return {
        "canonical_p_protocol": CANONICAL_P_PROTOCOL,
        "active_p_cache_entries": state.snapshot(),
        "ppkg_path": None if personality is None else str(personality.path),
        "ppkg_mutations": [] if personality is None else personality.mutations,
        "promoted_personality_entries": personality_page.pop("entries"),
        "promoted_personality_page": personality_page,
    }


class PlannerChatSession:
    """One active Planner Cache session shared by terminal and web front ends."""

    def __init__(self, args: argparse.Namespace) -> None:
        validate_args(args)
        self.args = args
        representation, _config, probe = train_and_probe_representation()
        store = (
            CanonicalPStore.load(args.p_cache, dtype=torch.float32)
            if args.p_cache is not None else None
        )
        self.state = CanonicalStateManager(representation, slots=args.slots, store=store)
        if args.p_cache is not None:
            self.state.load_runtime_metadata(
                args.p_cache.with_name("p-cache-runtime.json")
            )
        self.personality = (
            PersonalityManager(args.ppkg, representation)
            if args.ppkg is not None else None
        )
        try:
            self.runtime = build_runtime(args)
        except Exception:
            if self.personality is not None:
                self.personality.close()
            raise
        self.reviewer = PostTurnMemoryReviewer(self.runtime.review_memory)
        metadata = {
            **self.runtime.metadata(),
            "canonical_p_protocol_version": CANONICAL_P_PROTOCOL,
            "ppkg_path": (
                None if self.personality is None
                else str(self.personality.path.resolve())
            ),
            "git_commit": git_commit(args.repo_root),
            "launch_arguments": vars(args) | {
                key: None if value is None else str(value)
                for key, value in vars(args).items() if isinstance(value, Path)
            },
            "canonical_representation": {
                "recipe": "train_and_probe_representation",
                "seed": 97,
                "training_steps": 600,
                "held_out_p_only_state_recovery": probe["p_only_state_recovery"],
            },
        }
        self.recorder = SessionRecorder(
            args.session_root, args.runtime, metadata,
            enabled=not args.no_logging,
        )
        self.history: list[tuple[str, str]] = []
        self._pending_review: tuple[str, str, list[tuple[str, str]]] | None = None
        self.closed = False

    @property
    def title(self) -> str:
        return (
            "Planner Cache — Pythia-1.4B"
            if self.args.runtime == "pythia"
            else "Planner Cache — Gemma4 E4B"
        )

    def command(
        self,
        command: str,
        *,
        source: str = "terminal",
        personality_limit: int = 100,
        personality_offset: int = 0,
    ) -> object:
        command = command.strip().casefold()
        self.recorder.event("SESSION_COMMAND", source=source, command=command)
        if command == "/help":
            return HELP
        if command == "/state":
            return self.state.snapshot()
        if command == "/personality":
            page = (
                {
                    "entries": [], "total_active": 0, "returned": 0,
                    "limit": personality_limit, "offset": personality_offset,
                    "truncated": False,
                }
                if self.personality is None
                else self.personality.visible_entry_page(
                    limit=personality_limit, offset=personality_offset,
                )
            )
            return {
                "promoted": page.pop("entries"),
                "page": page,
                "last_retrieval": (
                    None
                    if self.personality is None
                    or self.personality.last_selection is None
                    else {
                        "route": asdict(self.personality.last_selection.route),
                        "entries": [
                            asdict(entry)
                            for entry in self.personality.last_selection.entries
                        ],
                    }
                ),
            }
        if command == "/events":
            return list(self.recorder.recent_events)
        if command == "/save":
            save_session_state(
                self.recorder, self.state, self.personality, reason="explicit"
            )
            return "Session state saved."
        if command == "/quit":
            return "quit"
        return "Unknown command. Type /help."

    def chat(
        self,
        message: str,
        *,
        raw_messages: list[dict[str, object]] | None = None,
    ) -> GenerationResult:
        if self._pending_review is not None:
            raise RuntimeError("previous post-turn memory review is incomplete")
        self.recorder.turn += 1
        self.recorder.transcript(
            role="user", text=message, model=self.runtime.model_id,
            runtime=self.runtime.runtime,
        )
        self.recorder.event(
            "P_STATE_BEFORE", source="p_cache", entries=self.state.snapshot()
        )
        mutations = self.state.extract_manual_mutations(message)
        if mutations:
            self.state.apply(mutations, self.recorder)
        else:
            self.recorder.event(
                "P_IGNORE", source="p_cache",
                reason="no deterministic manual override before generation",
            )
        if self.personality is not None:
            evidence = self.personality.extract_evidence(
                message, turn=self.recorder.turn, timestamp=utc_now(),
            )
            if evidence is not None:
                self.personality.ingest(evidence, self.recorder)
            personality_store = self.personality.query(message, self.recorder)
        else:
            personality_store = None
        query = self.state.infer_query(message)
        if (
            query.entity is None
            and personality_store is not None
            and personality_store.cache.occupied
        ):
            query = type(query)(
                entity="user", relation_id=0, relation="response_style",
                reason="accepted context-specific P-package state",
            )
        open_values = bool(getattr(self.runtime, "supports_open_values", False))
        translation_store = self.state.translation_store(
            include_open_values=open_values
        )
        if query.entity is not None and query.relation_id is not None:
            full_route, full_candidates = canonical_route(
                self.runtime.router, self.runtime.encoder, self.state.store, query,
            )
            if full_route is not None and full_route.has_valid:
                selected_slot = int(full_route.indices[0, 0])
                accepted = bool(full_route.accepted[0])
                if (
                    accepted
                    and not open_values
                    and not self.state.translator_compatible.get(selected_slot, True)
                ):
                    self.recorder.event(
                        "ROUTER_QUERY", source="canonical_compatibility_precheck",
                        entity=query.entity, relation=query.relation,
                    )
                    self.recorder.event(
                        "ROUTER_CANDIDATES", source="p_cache",
                        candidates=full_candidates,
                    )
                    self.recorder.event(
                        "ROUTER_ACCEPT", source="p_cache",
                        score=float(full_route.scores[0, 0].detach()),
                        selected_state=self.state.entry(selected_slot),
                    )
                    self.recorder.event(
                        "TTL_DISABLE", source="ttl",
                        reason=(
                            "canonical value is outside this TTL's supported vocabulary"
                        ),
                        selected_state=self.state.entry(selected_slot),
                    )
        self.recorder.event(
            "MODEL_GENERATION_START", source="model", model=self.runtime.model_id,
            runtime=self.runtime.runtime, query=asdict(query),
        )
        started = utc_now()
        try:
            result = self.runtime.generate(
                message, self.history, translation_store, personality_store, query,
                emit=self.recorder.event if self.recorder.enabled else None,
                raw_messages=raw_messages,
            )
        except Exception as error:
            self.recorder.event(
                "ERROR", source="model", error_type=type(error).__name__,
                message=str(error), traceback=traceback.format_exc(),
            )
            raise
        self.recorder.event(
            "MODEL_GENERATION_END", source="model", started_at=started,
            latency_seconds=result.latency_seconds,
            input_tokens=result.input_tokens, output_tokens=result.output_tokens,
            diagnostics=result.diagnostics,
        )
        self.recorder.transcript(
            role="assistant", text=result.text, model=self.runtime.model_id,
            runtime=self.runtime.runtime, latency_seconds=result.latency_seconds,
            input_tokens=result.input_tokens, output_tokens=result.output_tokens,
        )
        model_candidates = self.state.extract_mutations(result.text)
        for candidate in model_candidates:
            self.recorder.event(
                "P_IGNORE", source="model_output", candidate=asdict(candidate),
                reason="model-generated state is not authoritative without user evidence",
            )
        self._pending_review = (message, result.text, list(self.history))
        return result

    def complete_turn_review(self) -> None:
        """Finish the hidden review before another visible turn may start."""
        if self._pending_review is None:
            return
        message, response, prior_history = self._pending_review
        self._pending_review = None
        self.reviewer.run(
            user_message=message,
            assistant_response=response,
            recent_context=prior_history,
            state=self.state,
            recorder=self.recorder,
        )
        self.history.append((message, response))
        self.recorder.event(
            "P_STATE_AFTER", source="p_cache", entries=self.state.snapshot()
        )

    def close(self, *, reason: str = "normal") -> None:
        if self.closed:
            return
        self.closed = True
        if self._pending_review is not None:
            self.complete_turn_review()
        try:
            self.runtime.close()
        except Exception as error:
            self.recorder.event(
                "ERROR", source="runtime_cleanup",
                error_type=type(error).__name__, message=str(error),
            )
        try:
            save_session_state(
                self.recorder, self.state, self.personality, reason="final"
            )
        except Exception as error:
            self.recorder.event(
                "ERROR", source="session_save",
                error_type=type(error).__name__, message=str(error),
            )
        try:
            payload = final_payload(self.state, self.personality)
        except Exception as error:
            self.recorder.event(
                "ERROR", source="final_state",
                error_type=type(error).__name__, message=str(error),
            )
            payload = {
                "canonical_p_protocol": CANONICAL_P_PROTOCOL,
                "active_p_cache_entries": self.state.snapshot(),
                "ppkg_mutations": [],
                "final_state_error": str(error),
            }
        self.recorder.finalize(payload, reason=reason)
        if self.personality is not None:
            try:
                self.personality.close()
            except Exception:
                pass


def run(args: argparse.Namespace) -> int:
    session = None
    reason = "normal"
    try:
        session = PlannerChatSession(args)
        title = (
            "Planner Cache — Pythia-1.4B"
            if args.runtime == "pythia"
            else "Planner Cache — Gemma4 E4B"
        )
        print(f"\n{title}\n")
        while True:
            try:
                message = input("You: ")
            except EOFError:
                reason = "eof"
                break
            stripped = message.strip()
            if not stripped:
                continue
            if stripped.startswith("/"):
                command = stripped.casefold()
                if command == "/quit":
                    session.command(command)
                    reason = "quit"
                    break
                value = session.command(command)
                if isinstance(value, str):
                    print(value)
                else:
                    display_json(value)
                continue
            try:
                result = session.chat(message)
            except KeyboardInterrupt:
                reason = "ctrl-c"
                raise
            except Exception as error:
                print(f"Generation failed: {error}", file=sys.stderr)
                continue
            print(f"Assistant: {result.text}")
            session.complete_turn_review()
    except KeyboardInterrupt:
        reason = "ctrl-c"
        print("\nStopping.")
    finally:
        if session is not None:
            session.close(reason=reason)
    return 0


def main() -> None:
    try:
        raise SystemExit(run(parser().parse_args()))
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"Planner Cache startup failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
