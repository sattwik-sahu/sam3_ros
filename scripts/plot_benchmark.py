#!/usr/bin/env python3
"""Render the SAM3-ROS benchmark JSON as dependency-free SVG charts.

Usage: python scripts/plot_benchmark.py
Reads docs/assets/benchmark.json and writes benchmark_latency.svg and
benchmark_vram.svg next to it.
"""

from __future__ import annotations

import json
from pathlib import Path

BG = '#f8fafc'
GRID = '#e2e8f0'
TEXT = '#0f172a'
MUTED = '#64748b'
GREEN = '#16a34a'
BLUE = '#2563eb'
ORANGE = '#ea580c'
W, H = 860, 380
ML, MR, MT, MB = 70, 24, 58, 58


def scale(value, lo, hi, a, b):
    if hi == lo:
        return (a + b) / 2
    return a + (value - lo) / (hi - lo) * (b - a)


def nice_step(span: float) -> float:
    raw = span / 5.0
    magnitude = 10 ** int(f'{raw:e}'.split('e')[1])
    for mult in (1, 2, 2.5, 5, 10):
        if raw <= mult * magnitude:
            return mult * magnitude
    return 10 * magnitude


def chart(rows, series, ylabel, title, subtitle, out, ymin=None, ymax=None):
    xs = [r['megapixels'] for r in rows]
    xmin, xmax = 0, max(xs) * 1.08
    data_max = max(v for _, key, _ in series for v in (r[key] for r in rows))
    if ymax is None:
        ymax = data_max * 1.1
    if ymin is None:
        ymin = 0
    step = nice_step(ymax - ymin)
    ymax = ymin + step * 5
    plot_w, plot_h = W - ML - MR, H - MT - MB
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        f'<rect width="{W}" height="{H}" fill="{BG}"/>',
        f'<text x="{ML}" y="26" font-family="DejaVu Sans, sans-serif" font-size="18" font-weight="700" fill="{TEXT}">{title}</text>',
        f'<text x="{ML}" y="45" font-family="DejaVu Sans, sans-serif" font-size="12" fill="{MUTED}">{subtitle}</text>',
    ]
    for i in range(6):
        y = MT + plot_h * i / 5
        val = ymax - (ymax - ymin) * i / 5
        svg.append(f'<line x1="{ML}" y1="{y:.1f}" x2="{W-MR}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        svg.append(f'<text x="{ML-10}" y="{y+4:.1f}" text-anchor="end" font-family="DejaVu Sans, sans-serif" font-size="11" fill="{MUTED}">{val:.0f}</text>')
    # x ticks are categorical to keep every resolution label readable
    def x_at(i):
        return ML + (plot_w * i / max(len(rows) - 1, 1))

    for i, r in enumerate(rows):
        x = x_at(i)
        svg.append(f'<line x1="{x:.1f}" y1="{MT}" x2="{x:.1f}" y2="{MT+plot_h}" stroke="{GRID}" stroke-width="1"/>')
        anchor = 'start' if i == 0 else ('end' if i == len(rows) - 1 else 'middle')
        svg.append(f'<text x="{x:.1f}" y="{MT+plot_h+18}" text-anchor="{anchor}" font-family="DejaVu Sans, sans-serif" font-size="9.5" fill="{TEXT}">{r["width"]}×{r["height"]}</text>')
        svg.append(f'<text x="{x:.1f}" y="{MT+plot_h+31}" text-anchor="{anchor}" font-family="DejaVu Sans, sans-serif" font-size="9" fill="{MUTED}">{r["megapixels"]:.2f} MP</text>')
    svg.append(f'<text x="{ML+plot_w/2:.1f}" y="{H-16}" text-anchor="middle" font-family="DejaVu Sans, sans-serif" font-size="12" fill="{MUTED}">input resolution (ordered by pixel count)</text>')
    svg.append(f'<text x="18" y="{MT+plot_h/2:.1f}" transform="rotate(-90 18 {MT+plot_h/2:.1f})" text-anchor="middle" font-family="DejaVu Sans, sans-serif" font-size="12" fill="{MUTED}">{ylabel}</text>')
    # series
    for name, key, color in series:
        pts = [(x_at(i), scale(r[key], ymin, ymax, MT + plot_h, MT)) for i, r in enumerate(rows)]
        path = ' '.join(f'{x:.1f},{y:.1f}' for x, y in pts)
        svg.append(f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="2.5" stroke-linejoin="round"/>')
        for (x, y), r in zip(pts, rows):
            svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{BG}" stroke="{color}" stroke-width="2"/>')
        # legend
        lx = W - MR - 190
        ly = MT + 10 + 20 * list(series).index((name, key, color))
        svg.append(f'<line x1="{lx}" y1="{ly}" x2="{lx+24}" y2="{ly}" stroke="{color}" stroke-width="3"/>')
        svg.append(f'<text x="{lx+30}" y="{ly+4}" font-family="DejaVu Sans, sans-serif" font-size="12" fill="{TEXT}">{name}</text>')
    svg.append('</svg>')
    Path(out).write_text('\n'.join(svg))
    print('wrote', out)


def main():
    assets = Path(__file__).resolve().parent.parent / 'docs' / 'assets'
    payload = json.loads((assets / 'benchmark.json').read_text())
    rows = payload['rows']
    meta = payload['meta']
    chart(
        rows,
        [('p50', 'p50_ms', GREEN), ('p95', 'p95_ms', ORANGE)],
        'latency (ms)',
        'End-to-end segmentation latency',
        f"{meta['gpu']} · torch {meta['torch']} · CUDA {meta['cuda']} · {meta['precision']} · 1008² encoder",
        assets / 'benchmark_latency.svg',
        ymin=140,
        ymax=210,
    )
    chart(
        rows,
        [('peak allocated', 'peak_allocated_mib', BLUE), ('peak reserved', 'peak_reserved_mib', ORANGE), ('device used', 'device_used_mib', GREEN)],
        'GPU memory (MiB)',
        'GPU memory vs input resolution',
        f"baseline after load: {meta['baseline_allocated_mib']:.0f} MiB allocated, {meta['baseline_reserved_mib']:.0f} MiB reserved · axis starts at 3000 MiB",
        assets / 'benchmark_vram.svg',
        ymin=3000,
        ymax=7000,
    )


if __name__ == '__main__':
    main()
