from __future__ import annotations

from typing import Any

import pandas as pd

from ...results.performance import PerformanceTrainResult
from ..domain import DomainCodec

EXCLUDED_FIELDS = [
    'address', 'main_dimension', 'main_stat', 'measure',
    'period', 'frequency', 'total', 'lineages',
]


class PerformanceCodec(DomainCodec):
    """Encodes/decodes performance analysis structures and results."""

    def get_train_frame_names(self, config: Any) -> list[str]:
        return ['perf']

    def decode_train_results(
        self, frames: dict[str, pd.DataFrame],
        keys: dict[str, Any],
        config: Any,
        model_id: str,
        encoding: bool,
        **kwargs: Any,
    ) -> PerformanceTrainResult:
        raw = kwargs.get('raw_analysis', {})
        analysis: dict[str, Any] = raw if isinstance(raw, dict) else {}
        if encoding and analysis:
            fref = keys.get('fref', {})
            xref = keys.get('xref', {})
            analysis = {
                'drivers': _decode_driver(analysis.get('drivers', {}), fref),
                'dimensions': _decode_dimension(analysis.get('dimensions', {}), fref, xref),
                'synopsis': {},
            }
        return PerformanceTrainResult(model_id=model_id, analysis=analysis)

    @staticmethod
    def encode_edges(edges: list[tuple[str, str]], fref: dict[str, Any]) -> list[tuple[str, str]]:
        """Encode primary graph edges using field name references."""
        return [(fref['forward'][e[0]], fref['forward'][e[1]]) for e in edges]

    @staticmethod
    def encode_hierarchies(
        hierarchies: list[dict[str, Any]], fref: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Encode dimensional hierarchies using field name references."""
        return [_encode_hierarchy_item(item, fref) for item in hierarchies]

    @staticmethod
    def encode_udf(udf: dict[str, str], fref: dict[str, Any]) -> dict[str, str]:
        """Encode user-defined functions by replacing field names in formulas."""
        udf2: dict[str, str] = {}
        field_names = sorted(list(fref['forward'].keys()), key=len, reverse=True)
        for key in udf:
            key2 = fref['forward'][key]
            formula2 = udf[key]
            for name in field_names:
                formula2 = formula2.replace(name, fref['forward'][name])
            udf2[key2] = formula2
        return udf2

    @staticmethod
    def encode_coefs(coef: dict[str, float], fref: dict[str, Any]) -> dict[str, float]:
        """Encode coefficient overrides using field name references."""
        return {fref['forward'][key]: coef[key] for key in coef}

    @staticmethod
    def encode_address(
        address: list[str | None], dimensions: list[str],
        fref: dict[str, Any], xref: dict[str, Any],
    ) -> tuple[str | None, ...]:
        """Encode dimensional address for run operations."""
        address2: list[str | None] = []
        for idx, dim in enumerate(dimensions):
            member = xref[dim]['forward'][address[idx]] if address[idx] is not None else None
            address2.append(member)
        return tuple(address2)


def _encode_hierarchy_item(item: dict[str, Any], fref: dict[str, Any]) -> dict[str, Any]:
    """Recursively encode a hierarchy item and its children."""
    item2: dict[str, Any] = {
        'dimension': fref['forward'][item['dimension']],
        'name': fref['forward'][item['dimension']],
    }
    if item['child'] is not None:
        item2['child'] = _encode_hierarchy_item(item['child'], fref)
    else:
        item2['child'] = None
    return item2


def _decode_driver(obj: dict[str, Any], fref: dict[str, Any]) -> dict[str, Any]:
    """Recursively decode driver analysis results."""
    if not obj:
        return {}
    obj2: dict[str, Any] = {
        'node_id': obj['node_id'],
        'period': obj['period'],
        'measure': fref['reverse'][obj['measure']],
        'stats': obj['stats'],
        'anomaly_detection': obj['anomaly_detection'],
        'children': [],
        'main_driver': obj['main_driver'],
    }
    if obj2['main_driver']['measure'] is not None:
        obj2['main_driver']['measure'] = fref['reverse'][obj2['main_driver']['measure']]
    for child in obj['children']:
        obj2['children'].append(_decode_driver(child, fref))
    return obj2


def _decode_dimension(
    obj: dict[str, Any], fref: dict[str, Any], xref: dict[str, Any],
) -> dict[str, Any]:
    """Decode dimensional analysis results."""
    if not obj:
        return {}
    obj2: dict[str, Any] = {
        'measure': fref['reverse'][obj['measure']],
        'address': obj['address'],
        'period': obj['period'],
        'total': obj['total'],
        'main_dimension': obj['main_dimension'],
    }
    obj2['main_dimension']['dimension'] = fref['reverse'][obj['main_dimension']['dimension']]
    obj2['main_dimension']['member'] = xref[obj2['main_dimension']['dimension']]['reverse'][obj['main_dimension']['member']]
    for dim in obj:
        if dim not in EXCLUDED_FIELDS and dim in fref['reverse']:
            dim2 = fref['reverse'][dim]
            obj2[dim2] = {}
            for member in obj[dim]:
                member2 = xref[dim2]['reverse'][member] if member != 'residual' else member
                obj2[dim2][member2] = obj[dim][member]
    return obj2
