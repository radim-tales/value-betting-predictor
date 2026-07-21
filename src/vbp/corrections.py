from __future__ import annotations


def apply_correction(anchor_p: dict, delta: dict,
                     zero_sum_tol: float = 0.08,
                     clip_lo: float = 0.01, clip_hi: float = 0.98) -> tuple[dict, bool]:
    """Apply LLM delta to anchor probs. Returns (probs, skipped).
    Skip (return anchor unchanged) if deltas are not ~zero-sum (a broken/incoherent
    correction) - per spec §10 we do NOT silently normalize large errors. Default
    tolerance is locked at 0.08 per spec §10."""
    d = {o: float(delta.get(f"d{o}", 0.0)) for o in ("H", "D", "A")}
    if abs(sum(d.values())) > zero_sum_tol:
        return dict(anchor_p), True
    raw = {o: anchor_p[o] + d[o] for o in ("H", "D", "A")}
    return _clip_and_renormalize(raw, clip_lo, clip_hi), False


def _clip_and_renormalize(raw: dict, clip_lo: float, clip_hi: float) -> dict:
    """Project raw values onto the simplex (sum=1) with each coordinate bounded
    to [clip_lo, clip_hi]. A naive "clip then divide by total" pass can push a
    just-clipped value back outside the bound once the total is renormalized
    (e.g. sum=1.06 after clipping means clip_lo/1.06 < clip_lo). Instead, fix
    any out-of-bounds coordinate exactly at its bound and redistribute the
    remaining mass proportionally among the rest, repeating until stable.
    """
    fixed: dict = {}
    free = set(raw.keys())
    while True:
        remaining_mass = 1.0 - sum(fixed.values())
        free_sum = sum(raw[o] for o in free)
        candidates = {
            o: (remaining_mass * raw[o] / free_sum) if free_sum else (remaining_mass / len(free))
            for o in free
        }
        violators = {o: v for o, v in candidates.items() if v < clip_lo or v > clip_hi}
        if not violators:
            fixed.update(candidates)
            # Degenerate case: all coordinates got pinned to a clip bound in a
            # single pass (possible with custom clip_lo/clip_hi that don't
            # satisfy 2*clip_lo + clip_hi == 1), so `free` emptied out without
            # ever redistributing remaining mass. Guarantee sum-to-one via a
            # final proportional renormalization, even if that means slightly
            # violating a clip bound in this pathological case.
            total = sum(fixed.values())
            if not free and abs(total - 1.0) > 1e-9 and total:
                fixed = {o: v / total for o, v in fixed.items()}
            return fixed
        for o, v in violators.items():
            fixed[o] = clip_lo if v < clip_lo else clip_hi
            free.discard(o)
