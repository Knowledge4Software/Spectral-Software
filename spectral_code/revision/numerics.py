"""Bounded effective-batch replay for AMP overflow, without skipping updates.

The finite-gradient path preserves the original accumulation, clipping and
optimizer sequence. Failed attempts restore RNG and model buffers and never
advance the training cursor. A persistent FP32 failure is still fatal.
"""
from __future__ import annotations

from pathlib import Path

import torch

POLICY = "amp_batch_replay_fp32_fallback_v1"
MAX_AMP_ATTEMPTS = 3
LEGACY_IMPLEMENTATIONS = frozenset({"6da6f8fffdc37744ce177a3f573260a6e52f5f8909d23124bcb5aae61e6e4f7a"})
# Filled after the numerical-only training patch is frozen. This explicitly
# prevents later architecture/data/training edits inheriting upgrade permission.
APPROVED_UPGRADE_TARGET = "3b8b72701001931d3c3eb090521567b7902b05deac92b7e97f5523dd8b79d477"
VALIDATION_RECOVERY_SOURCE = "8ad938f286142084e2c89ef8439bf9310db61e30019f3d61d02b90426df52e26"
APPROVED_VALIDATION_TARGET = "8a3a065956f00155399f64574847ceb638db612693bfd9c3da9a1a57abad13ba"


def upgrade_target_hash():
    from .protocol import fingerprint
    from .versioning import TRAINING_SOURCES
    root = Path(__file__).parent
    return fingerprint({name: (root / name).read_text(encoding="utf-8")
                        for name in TRAINING_SOURCES if name not in {"numerics.py", "versioning.py"}})


def can_upgrade(implementation_hash):
    target = upgrade_target_hash()
    return ((implementation_hash in LEGACY_IMPLEMENTATIONS and target == APPROVED_UPGRADE_TARGET)
            or (implementation_hash == VALIDATION_RECOVERY_SOURCE and target == APPROVED_VALIDATION_TARGET))


def _finite_tree(value):
    if torch.is_tensor(value):
        return not value.is_floating_point() or bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(_finite_tree(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return all(_finite_tree(v) for v in value)
    return True


def validate_checkpoint_numerics(state):
    if not _finite_tree({key: state.get(key) for key in ("model", "best_model", "optimizer")}):
        raise ValueError("Checkpoint contains non-finite weights/optimizer state; do not resume it")


class NumericalStepFailure(FloatingPointError):
    def __init__(self, diagnostic):
        self.diagnostic = diagnostic
        super().__init__("Non-finite loss/gradients persisted in FP32; no optimizer update was applied")


class _AttemptFailure(Exception):
    def __init__(self, reason, names=()):
        self.reason, self.names = reason, list(names)


def optimizer_group(model, optimizer, scaler, group, *, config, device, losses,
                    loss_function, positive_class_weight, clock=None, on_event=None):
    """Apply exactly one optimizer update, retrying the SAME effective batch.

    Input batches remain on CPU until their forward pass. This adds at most one
    effective batch of CPU storage, not its combined activations on the GPU.
    """
    from .training import rng_state, restore_rng

    initial_rng = rng_state()
    buffers = {name: buffer.detach().clone() for name, buffer in model.named_buffers()}
    examples = sum(len(labels) for _, _, labels in group)
    events = []
    use_scaler = scaler.is_enabled()
    attempts = MAX_AMP_ATTEMPTS + 1 if use_scaler else 1

    def restore():
        optimizer.zero_grad(set_to_none=True)
        restore_rng(initial_rng)
        with torch.no_grad():
            for name, buffer in model.named_buffers():
                buffer.copy_(buffers[name])

    for attempt in range(attempts):
        fp32_fallback = use_scaler and attempt == MAX_AMP_ATTEMPTS
        scaled_attempt = use_scaler and not fp32_fallback
        autocast_enabled = config.amp and device.type == "cuda" and not fp32_fallback
        if attempt:
            restore()
        contributions = []
        try:
            if clock:
                clock.check()
            for left, right, labels in group:
                labels = labels.to(device)
                with torch.autocast(device_type=device.type, enabled=autocast_enabled):
                    logits, auxiliary = model(left.to(device), right.to(device), None)
                    loss, _ = loss_function(logits, auxiliary, labels,
                                             positive_class_weight=positive_class_weight, **losses)
                weighted = loss * len(labels) / examples
                # Initialize the public scaler before checking a non-finite
                # loss, so an explicit backoff can reset its attempt state.
                backward_loss = scaler.scale(weighted) if scaled_attempt else weighted
                if not bool(torch.isfinite(loss)):
                    raise _AttemptFailure("nonfinite_loss")
                backward_loss.backward()
                contributions.append((float(loss.detach().cpu()) * len(labels), len(labels)))
            if scaled_attempt:
                scaler.unscale_(optimizer)
            try:
                # error_if_nonfinite checks BEFORE clipping can turn inf*0
                # into NaN. Do not mask invalid gradients with nan_to_num.
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            except RuntimeError as error:
                if "non-finite" not in str(error).lower():
                    raise
                names = [name for name, parameter in model.named_parameters()
                         if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())]
                raise _AttemptFailure("nonfinite_gradient_norm", names[:12]) from error
        except _AttemptFailure as failure:
            # A non-finite forward loss has not run backward, so its graph can
            # still hold GPU activations. Release it before the replay/fallback.
            del logits, auxiliary, loss, weighted, backward_loss
            event = {"attempt": attempt + 1, "precision": "fp32_fallback" if fp32_fallback else
                     "amp" if scaled_attempt else "fp32", "reason": failure.reason,
                     "nonfinite_parameters": failure.names,
                     "loss_scale": float(scaler.get_scale()) if scaled_attempt else None}
            if scaled_attempt:
                # No step occurs for this failed attempt. Public explicit-scale
                # update clears the unscale state for the replayed group.
                new_scale = max(float(scaler.get_scale()) * scaler.get_backoff_factor(),
                                torch.finfo(torch.float32).tiny)
                scaler.update(new_scale=new_scale)
                event["next_loss_scale"] = new_scale
                event["action"] = "retry_same_group" if attempt + 1 < MAX_AMP_ATTEMPTS else "retry_same_group_fp32"
            else:
                event["action"] = "stop_with_recoverable_checkpoint"
            events.append(event)
            if on_event:
                on_event(event)
            if not scaled_attempt:
                restore()
                raise NumericalStepFailure({"policy": POLICY, "events": events}) from failure
            continue
        except BaseException:
            # In particular, a session deadline must save the pre-group RNG,
            # buffers and cursor, not a half-computed accumulation group.
            restore()
            raise

        if scaled_attempt:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if fp32_fallback:
            event = {"attempt": attempt + 1, "precision": "fp32_fallback", "action": "group_recovered"}
            events.append(event)
            if on_event:
                on_event(event)
        return contributions, events

    raise AssertionError("All AMP attempts must lead to a finite update or a recorded FP32 failure")
