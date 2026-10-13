"""Reference-only scales for diagonal residual-stress fitting (units: MPa).

The default preserves the historical per-component objective. Shared RMS is an
explicit opt-in: every normal stress error uses the same denominator. Scales
are frozen from AM data or fixed initial material yield stress, never from the
optimizer's predicted field.
"""
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class StressNormalization:
    mode: str
    scales_mpa: tuple
    legacy_component_scales_mpa: tuple
    reference_component_rms_mpa: tuple
    floor_mpa: float
    active_roi_cells: int
    initial_yield_stress_mpa: float | None = None

    def report(self):
        return {
            'mode': self.mode, 'units': 'MPa', 'floor_mpa': self.floor_mpa,
            'scales_mpa': dict(zip(('xx', 'yy', 'zz'), self.scales_mpa)),
            'legacy_component_scales_mpa': dict(zip(('xx', 'yy', 'zz'), self.legacy_component_scales_mpa)),
            'reference_component_rms_mpa': dict(zip(('xx', 'yy', 'zz'), self.reference_component_rms_mpa)),
            'active_roi_cells': self.active_roi_cells,
            'shared_formula': 'sqrt(mean_active_cells((sxx^2+syy^2+szz^2)/3))',
            'vm_normalization_changed': False,
            'initial_yield_stress_mpa': self.initial_yield_stress_mpa,
            'scale_source': ('material.initial_yield_stress_pa / 1e6 (fixed)'
                             if self.mode == 'material_yield' else 'AM reference RMS'),
        }


def initial_yield_scale_mpa(material_config):
    if not isinstance(material_config, dict) or 'initial_yield_stress_pa' not in material_config:
        raise ValueError('material_yield requires material.initial_yield_stress_pa')
    try:
        value = float(material_config['initial_yield_stress_pa']) / 1e6
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError('Initial yield stress must be finite and positive (Pa)') from error
    if not math.isfinite(value) or value <= 0:
        raise ValueError('Initial yield stress must be finite and positive (Pa)')
    return value


def build_stress_normalization(reference_diagonal_mpa, config=None, active_mask=None, material_config=None):
    config = {} if config is None else config
    if not isinstance(config, dict):
        raise ValueError('optimization.stress_normalization must be a mapping')
    mode = config.get('mode', 'component_rms')
    if mode not in ('component_rms', 'shared_rms', 'material_yield'):
        raise ValueError('stress_normalization.mode must be component_rms, shared_rms or material_yield')
    floor = float(config.get('floor_mpa', 1.0))
    if not math.isfinite(floor) or floor <= 0:
        raise ValueError('stress_normalization.floor_mpa must be finite and positive')
    ref = np.asarray(reference_diagonal_mpa, dtype=float)
    if ref.ndim != 2 or ref.shape[1] != 3 or ref.shape[0] == 0 or not np.all(np.isfinite(ref)):
        raise ValueError('Stress reference must be a finite nonempty (ROI cells, 3) array')
    mask = np.ones(len(ref), dtype=bool) if active_mask is None else np.asarray(active_mask)
    if mask.shape != (len(ref),) or not np.all(np.isin(mask, [0, 1])) or not np.any(mask):
        raise ValueError('Stress ROI active_mask must be binary, match ROI, and select cells')
    mask = mask.astype(bool)
    # Keep legacy scales over the full ROI, exactly as the previous code did.
    legacy_rms = np.sqrt(np.mean(ref**2, axis=0))
    legacy = tuple(float(x) for x in np.maximum(legacy_rms, 1.0))
    active_rms = np.sqrt(np.mean(ref[mask]**2, axis=0))
    yield_mpa = None
    if mode == 'component_rms':
        scales = tuple(float(x) for x in np.maximum(legacy_rms, floor))
    elif mode == 'material_yield':
        yield_mpa = initial_yield_scale_mpa(material_config)
        # Do not apply an AM RMS floor or a hardening-dependent yield value.
        scales = (yield_mpa, yield_mpa, yield_mpa)
    else:
        shared = max(float(np.sqrt(np.mean(ref[mask]**2))), floor)
        scales = (shared, shared, shared)
    if not np.all(np.isfinite(scales)):
        raise ValueError('Nonfinite stress normalization scale')
    return StressNormalization(mode, scales, legacy, tuple(map(float, active_rms)), floor, int(mask.sum()), yield_mpa)
