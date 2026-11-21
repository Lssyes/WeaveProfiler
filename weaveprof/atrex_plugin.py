"""Atrex v0.4.0 thin plugin adapter (`aka-plugin-autoprof`) for WeaveProfiler.

Zero hard dependency on Atrex: follows the `apply(ctx, config)` + `ctx.provide(...)`
capability contract so optimization agents can invoke sealed, verified intra-kernel
profiling and SW+HW causal bottleneck attribution with a single plugin registration.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from .api import build_agent_report, doctor, identify, locate_or_remap_0token, profile_for_agent, verify
from .causal import analyze_sw_hw_causality, compare_optimization_rounds, plan_context_and_search_space
from .harness import (
    HarnessPolicy,
    gpu_critical_region,
    register_lock_provider,
    run_compiler_feedback_selection,
    verify_seal,
)


PLUGIN_NAME = "aka-plugin-autoprof"
PLUGIN_VERSION = "1.0.0"
CAPABILITY_KEY = "intra_kernel_verifier_profiler"

# AKA Core plugin framework exports (`docs/plugin-framework.md`)
name = "weaveprofiler"
provide = (CAPABILITY_KEY,)
Defaults = {
    "default_limit": 0.10,
    "model": "qwen4-max",
    "card": "auto",
    "max_rounds": 3,
    "enforce_trace_processor": False,
    "require_abba_order": False,
}
Config = {
    "type": "object",
    "properties": {
        "default_limit": {"type": "number", "minimum": 0, "maximum": 0.5},
        "model": {"type": "string", "minLength": 1},
        "card": {"type": "string", "minLength": 1},
        "max_rounds": {"type": "integer", "minimum": 1, "maximum": 10},
        "lock_provider": {"type": "string", "minLength": 1},
        "enforce_trace_processor": {"type": "boolean"},
        "require_abba_order": {"type": "boolean"},
    },
    "additionalProperties": True,
}


@dataclass
class AutoProfService:
    """Capability object provided to Atrex under `intra_kernel_verifier_profiler`."""

    default_limit: float = 0.10
    default_model: str = "qwen4-max"
    default_card: str = "auto"
    default_max_rounds: int = 3
    lock_provider: Optional[str] = None
    enforce_trace_processor: bool = False
    require_abba_order: bool = False

    def policy(self, **overrides: Any) -> HarnessPolicy:
        params = dict(
            max_overhead_limit=self.default_limit,
            enforce_trace_processor=self.enforce_trace_processor,
            require_abba_order=self.require_abba_order,
        )
        params.update(overrides)
        return HarnessPolicy(**params)

    def profile_for_agent(self, source, *, bench, **kwargs: Any) -> Dict[str, Any]:
        kwargs.setdefault("limit", self.default_limit)
        kwargs.setdefault("model", self.default_model)
        kwargs.setdefault("card", self.default_card)
        kwargs.setdefault("max_rounds", self.default_max_rounds)
        kwargs.setdefault("policy", self.policy())
        return profile_for_agent(source, bench=bench, **kwargs)

    def identify(self, source, **kwargs: Any) -> Dict[str, Any]:
        kwargs.setdefault("model", self.default_model)
        kwargs.setdefault("card", self.default_card)
        return identify(source, **kwargs)

    def verify(self, probe, *, bench, **kwargs: Any) -> Dict[str, Any]:
        kwargs.setdefault("limit", self.default_limit)
        kwargs.setdefault("policy", self.policy())
        return verify(probe, bench=bench, **kwargs)

    def verify_seal(self, seal_path, **kwargs: Any) -> Dict[str, Any]:
        return verify_seal(seal_path, **kwargs)

    def analyze_causality(self, data: Mapping[str, Any], **kwargs: Any) -> Dict[str, Any]:
        return analyze_sw_hw_causality(data, **kwargs)

    def plan_context(self, source_text: str, **kwargs: Any) -> Dict[str, Any]:
        return plan_context_and_search_space(source_text, **kwargs)

    def select_compiler_arm(self, compile_fn: Any, source_text: str, **kwargs: Any) -> Dict[str, Any]:
        """D2: compile WP_CTX_1R first, fall back to FI_SMEM_ZREG / WP_CTX_1R_NOINLINE on spill/STL regression."""
        return run_compiler_feedback_selection(compile_fn, source_text, **kwargs)

    def compare_rounds(self, prev_report: Mapping[str, Any], curr_report: Mapping[str, Any], **kwargs: Any) -> Dict[str, Any]:
        """D3: round-to-round causal diff + next-round directive for autonomous optimizers."""
        return compare_optimization_rounds(prev_report, curr_report, **kwargs)

    def build_agent_report(self, result: Mapping[str, Any], **kwargs: Any) -> Dict[str, Any]:
        return build_agent_report(result, **kwargs)

    def remap_probe_0token(self, prior_spec: Mapping[str, Any], old_source_text: str, new_source_text: str, **kwargs: Any):
        return locate_or_remap_0token(prior_spec, old_source_text, new_source_text, **kwargs)

    def critical_region(self, gpu_id: str = "auto", **kwargs: Any):
        if self.lock_provider and "lock_provider" not in kwargs:
            kwargs["lock_provider"] = self.lock_provider
        return gpu_critical_region(gpu_id=gpu_id, **kwargs)

    def doctor(self, *, check_gpu: bool = True) -> Dict[str, Any]:
        return doctor(check_gpu=check_gpu)


def apply(ctx: Any, config: Optional[Mapping[str, Any]] = None) -> AutoProfService:
    """Atrex v0.4.0 plugin entrypoint: register `intra_kernel_verifier_profiler`."""
    cfg = dict(config or {})
    lock_provider = cfg.get("lock_provider")
    if callable(cfg.get("lock_provider_factory")) and lock_provider:
        register_lock_provider(str(lock_provider), cfg["lock_provider_factory"])

    service = AutoProfService(
        default_limit=float(cfg.get("default_limit", 0.10)),
        default_model=str(cfg.get("model", "qwen4-max")),
        default_card=str(cfg.get("card", "auto")),
        default_max_rounds=int(cfg.get("max_rounds", 3)),
        lock_provider=str(lock_provider) if lock_provider else None,
        enforce_trace_processor=bool(cfg.get("enforce_trace_processor", False)),
        require_abba_order=bool(cfg.get("require_abba_order", False)),
    )

    if hasattr(ctx, "provide") and callable(ctx.provide):
        ctx.provide(CAPABILITY_KEY, service)
    elif isinstance(ctx, dict):
        ctx[CAPABILITY_KEY] = service

    if hasattr(ctx, "register_tool") and callable(ctx.register_tool):
        ctx.register_tool(
            name="weaveprof_profile_for_agent",
            description=(
                "Run deterministic, HMAC-sealed intra-kernel GPU profiling and return a compact "
                "SW+HW causal bottleneck report (overhead %, fidelity gates, ranked optimization bounds)."
            ),
            handler=service.profile_for_agent,
        )
    return service
