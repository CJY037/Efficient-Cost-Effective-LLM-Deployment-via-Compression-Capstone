#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import gc
import html
import math
import os
import queue
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import gradio as gr
import pandas as pd
import torch
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

import llama3_8b_wikitext2 as base

APP_TITLE = "Efficient and Cost-Effective Deployment of Large Language Models via Compression Techniques"
APP_TITLE_DISPLAY = (
    "<span>Efficient and Cost-Effective Deployment of</span>"
    "<span>Large Language Models via Compression Techniques</span>"
)
APP_SUBTITLE = "Evaluation dashboard for model compression trade-offs across quality, latency, throughput, and GPU memory."
PWA_DIR = Path(__file__).resolve().parent / "pwa_assets"
GB = 1024 ** 3
DEFAULT_MAX_NEW_TOKENS = 756
QUALITY_MAX_NEW_TOKENS = 512
QUALITY_MIN_CAP_NEW_TOKENS = 128
QUALITY_MIN_OUTPUT_TOKENS = 40
BENCHMARK_MAX_NEW_TOKENS = 256
DEFAULT_BENCHMARK_NEW_TOKENS = 256
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 0.9
QUALITY_TEMPERATURE = 0.0
QUALITY_TOP_P = 1.0
DEFAULT_REPETITION_PENALTY = 1.10
DEFAULT_NO_REPEAT_NGRAM_SIZE = 4
DEFAULT_SEED = 42
DEFAULT_MAX_INPUT_CHARS = 60000
FINANCE_SYSTEM_PROMPT = "You are a helpful, harmless, and honest assistant."
DEFAULT_SYSTEM_PROMPT = (
    f"{FINANCE_SYSTEM_PROMPT} Answer directly and concisely. "
    "Do not invent names, dates, companies, sources, forecasts, or statistics. "
    "If the prompt does not provide a fact, avoid pretending it is known."
)
FINANCE_SENTIMENT_MAX_NEW_TOKENS = 1
FINANCE_GOOD_LABEL_CANDIDATES = (" Good", "Good")
FINANCE_BAD_LABEL_CANDIDATES = (" Bad", "Bad")
FINANCE_CALIBRATION_ARTICLE = "No company-specific financial article is provided."
DEFAULT_PROMPT = (
    "Explain fintech in simple terms using online banking as the example. Mention why people use it."
)
METRIC_DISPLAY_SPECS = [
    {
        "key": "prompt_tokens",
        "label": "Prompt tokens",
        "precision": 0,
        "suffix": "",
        "description": "Input context length used for the benchmark.",
        "direction": "equal",
    },
    {
        "key": "output_tokens",
        "label": "Output tokens",
        "precision": 0,
        "suffix": "",
        "description": "Generated or scored output-token count.",
        "direction": "equal",
    },
    {
        "key": "prefill_latency_ms",
        "label": "Prefill latency",
        "precision": 1,
        "suffix": " ms",
        "description": "Prompt-processing latency before decode begins.",
        "direction": "lower",
    },
    {
        "key": "prefill_tok_s",
        "label": "Prefill tok/s",
        "precision": 2,
        "suffix": "",
        "description": "Prompt-processing throughput.",
        "direction": "higher",
    },
    {
        "key": "decode_ms_per_token",
        "label": "Decode latency/token",
        "precision": 1,
        "suffix": " ms",
        "description": "Average per-token decode latency after prefill.",
        "direction": "lower",
    },
    {
        "key": "decode_tok_s",
        "label": "Decode tok/s",
        "precision": 2,
        "suffix": "",
        "description": "Autoregressive decode throughput after prefill.",
        "direction": "higher",
    },
    {
        "key": "total_latency_ms",
        "label": "End-to-end latency",
        "precision": 1,
        "suffix": " ms",
        "description": "Total request latency including prefill and decode.",
        "direction": "lower",
    },
    {
        "key": "e2e_tok_s",
        "label": "End-to-end tok/s",
        "precision": 2,
        "suffix": "",
        "description": "Overall output throughput across the full request.",
        "direction": "higher",
    },
    {
        "key": "peak_vram_gb",
        "label": "Peak VRAM",
        "precision": 2,
        "suffix": " GB",
        "description": "Highest GPU memory allocated during the run.",
        "direction": "lower",
    },
]
APP_THEME = gr.themes.Glass(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.cyan,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Manrope"), "Segoe UI Variable", "Helvetica Neue", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "Consolas", "monospace"],
).set(
    body_background_fill="#0b1f36",
    block_background_fill="rgba(0, 0, 0, 0)",
    block_border_width="1px",
    block_border_color="rgba(255, 255, 255, 0.12)",
    block_radius="24px",
    button_primary_background_fill="#2563eb",
    button_primary_background_fill_hover="#2563eb",
    button_primary_text_color="#ffffff",
    button_secondary_background_fill="#132640",
    button_secondary_background_fill_hover="#17304f",
    button_secondary_border_color="rgba(255, 255, 255, 0.12)",
    button_secondary_text_color="#e8f1ff",
)

CSS = """
:root {
  --text: #f5f9ff;
  --muted: #afc4dd;
  --muted-strong: #d7e5f6;
  --border: rgba(255, 255, 255, 0.10);
  --surface: linear-gradient(180deg, rgba(12, 24, 40, 0.90), rgba(7, 15, 28, 0.96));
  --surface-soft: rgba(255, 255, 255, 0.05);
  --surface-deep: rgba(4, 12, 24, 0.48);
  --dense: #82b7ff;
  --sparse: #f2c66d;
  --win: #5ee18b;
  --win-soft: rgba(94, 225, 139, 0.12);
  --win-border: rgba(94, 225, 139, 0.44);
  --warn: #f6d68c;
  --accent: #6aa6f8;
  --accent-strong: #2563eb;
  --hero-a: #07111f;
  --hero-b: #0d2540;
  --hero-c: #16436e;
  --hero-d: #21608f;
}

html,
body,
.gradio-container {
  min-height: 100%;
  background:
    radial-gradient(circle at 14% 8%, rgba(59, 130, 246, 0.30), transparent 30%),
    radial-gradient(circle at 88% 18%, rgba(34, 211, 238, 0.18), transparent 28%),
    linear-gradient(145deg, #0b1f36 0%, #0d2d4e 44%, #123f67 100%);
  color: var(--text);
}

html {
  -webkit-text-size-adjust: 100%;
}

.gradio-container {
  max-width: none !important;
  width: 100% !important;
  padding: 28px !important;
}

.gradio-container .contain,
.gradio-container .main {
  max-width: none !important;
}

.gradio-container .form,
.gradio-container .gr-box,
.gradio-container .block,
.gradio-container .gr-block,
.gradio-container .gr-panel,
.gradio-container .gr-group,
.gradio-container [data-testid="block"],
.gradio-container [data-testid="column"] {
  background: transparent !important;
  box-shadow: none !important;
  border: none !important;
}

.gradio-container .gap {
  gap: 18px !important;
}

.hero-shell {
  position: relative;
  overflow: hidden;
  margin-bottom: 22px;
  padding: 38px 44px 32px;
  border-radius: 28px;
  background:
    linear-gradient(128deg, rgba(255, 255, 255, 0.08), rgba(255, 255, 255, 0.02) 28%, transparent 52%),
    linear-gradient(140deg, var(--hero-a) 0%, var(--hero-b) 34%, var(--hero-c) 68%, var(--hero-d) 100%);
  border: 1px solid rgba(255, 255, 255, 0.12);
  box-shadow:
    0 24px 60px rgba(3, 10, 22, 0.36),
    inset 0 1px 0 rgba(255, 255, 255, 0.10);
}

.hero-copy {
  min-width: 0;
  max-width: 100%;
}

.hero-topbar {
  display: block;
}

.hero-eyebrow,
.pwa-title,
.score-title,
.metrics-title,
.panel-title,
.log-title,
.note-title,
.section-kicker {
  margin: 0;
  color: rgba(231, 240, 255, 0.72);
  font-size: 0.78rem;
  font-weight: 800;
  letter-spacing: 0;
  text-transform: uppercase;
}

.hero-title {
  margin: 0;
  color: #fbfdff;
  font-size: clamp(2.05rem, 3.35vw, 3.65rem);
  line-height: 1.01;
  letter-spacing: 0;
  font-weight: 800;
  text-wrap: balance;
}

.hero-title span {
  display: block;
  white-space: normal;
}

.hero-subtitle {
  margin: 14px 0 22px;
  max-width: none;
  color: rgba(232, 241, 255, 0.88);
  font-size: 1rem;
  line-height: 1.5;
  white-space: normal;
}

.hero-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

.hero-tag {
  padding: 10px 16px;
  border-radius: 14px;
  border: 1px solid rgba(255, 255, 255, 0.12);
  background: rgba(255, 255, 255, 0.08);
  color: #f8fbff;
  font-size: 0.82rem;
  font-weight: 700;
}

.pwa-shell {
  position: absolute;
  right: 44px;
  bottom: 12px;
  width: 190px;
  padding: 10px 10px 10px;
  border-radius: 18px;
  background: linear-gradient(180deg, rgba(8, 20, 38, 0.54), rgba(6, 16, 31, 0.72));
  border: 1px solid rgba(255, 255, 255, 0.12);
  box-shadow: 0 20px 42px rgba(3, 10, 22, 0.28);
  backdrop-filter: blur(18px);
}

.pwa-title {
  white-space: nowrap;
  font-size: 0.74rem;
  text-align: center;
}

.pwa-btn {
  width: 100%;
  margin-top: 10px;
  border: none;
  border-radius: 14px;
  background: linear-gradient(135deg, #3479e8, #199db0);
  color: #ffffff;
  font-weight: 800;
  padding: 10px 14px;
  cursor: pointer;
  box-shadow: 0 14px 30px rgba(46, 134, 255, 0.28);
}

.pwa-status {
  min-height: 0;
  margin-top: 8px;
  color: rgba(232, 241, 255, 0.70);
  font-size: 0.8rem;
  line-height: 1.5;
}

.pwa-status:empty {
  display: none;
}

.control-shell,
.panel-card,
.metric-shell,
.score-shell,
.log-shell,
.viz-shell,
.table-shell,
.note-shell {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 24px;
  box-shadow:
    0 18px 48px rgba(2, 9, 20, 0.26),
    inset 0 1px 0 rgba(255, 255, 255, 0.05);
  backdrop-filter: blur(18px);
  padding: 24px;
}

.control-shell > div,
.table-shell > div,
.viz-shell > div,
.model-shell > div {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}

.model-shell {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.section-head {
  margin-bottom: 16px;
}

.section-head h2 {
  margin: 10px 0 8px;
  color: #f8fbff;
  font-size: 1.56rem;
  line-height: 1.12;
  letter-spacing: 0;
}

.section-head p {
  margin: 0;
  color: rgba(232, 241, 255, 0.78);
  line-height: 1.68;
}

.section-head.compact {
  margin-bottom: 10px;
}

.section-kicker.dense {
  color: var(--dense);
}

.section-kicker.sparse {
  color: var(--sparse);
}

.action-row {
  margin-top: 14px;
  gap: 14px;
}

.dual-action-row {
  align-items: end;
}

.action-panel {
  display: flex;
  flex-direction: column;
  gap: 10px;
  min-width: 0;
}

.action-panel button {
  width: 100% !important;
}

#benchmark-task-radio {
  margin: 0 !important;
}

#benchmark-task-radio > label,
#benchmark-task-radio .label-wrap {
  color: rgba(232, 241, 255, 0.86) !important;
  font-weight: 800 !important;
}

#benchmark-task-radio .wrap {
  display: flex !important;
  flex-wrap: wrap !important;
  gap: 8px !important;
}

#benchmark-task-radio label {
  white-space: nowrap;
}

#benchmark-task-radio .wrap label {
  min-height: 42px;
  display: inline-flex !important;
  align-items: center;
  gap: 9px;
  padding: 9px 14px !important;
  border-radius: 14px !important;
  border: 1px solid rgba(255, 255, 255, 0.14) !important;
  background: linear-gradient(180deg, rgba(20, 35, 58, 0.88), rgba(12, 26, 46, 0.92)) !important;
  color: #eaf2ff !important;
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.08),
    0 10px 22px rgba(2, 9, 20, 0.16);
}

#benchmark-task-radio .wrap label:has(input[type="radio"]:checked) {
  border-color: rgba(99, 239, 213, 0.62) !important;
  background: linear-gradient(135deg, rgba(68, 142, 255, 0.98), rgba(31, 196, 207, 0.94)) !important;
  color: #ffffff !important;
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.16),
    0 16px 30px rgba(46, 134, 255, 0.26);
}

#benchmark-task-radio input[type="radio"] {
  accent-color: #1fc4cf;
}

.panel-title {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 12px;
  margin-bottom: 14px;
}

.panel-name {
  color: #f8fbff;
  font-size: 1.08rem;
  font-weight: 800;
}

.panel-name.model-dense,
.metrics-title.model-dense,
.comparison-grid th.model-dense {
  color: var(--dense);
}

.panel-name.model-sparse,
.metrics-title.model-sparse,
.comparison-grid th.model-sparse {
  color: var(--sparse);
}

.panel-mode {
  color: #f6d68c;
  font-size: 0.76rem;
  font-weight: 700;
  letter-spacing: 0;
  text-transform: uppercase;
}

.panel-line,
.log-entry,
.note-copy {
  color: var(--muted);
  line-height: 1.65;
}

.note-list {
  margin: 0;
  padding-left: 1.15rem;
  color: var(--muted);
  line-height: 1.7;
}

.note-list li + li {
  margin-top: 10px;
}

.note-list strong {
  display: block;
  color: #f4f8ff;
  font-weight: 700;
  margin-bottom: 2px;
}

.protocol-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
  margin-top: 14px;
}

.protocol-item {
  min-height: 112px;
  padding: 15px 16px;
  border-radius: 16px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  background: rgba(255, 255, 255, 0.035);
}

.protocol-title {
  margin: 0 0 7px;
  color: #f4f8ff;
  font-weight: 800;
}

.protocol-copy {
  margin: 0;
  color: var(--muted);
  line-height: 1.55;
}

.panel-line {
  margin: 0 0 6px;
}

.metric-grid,
.score-grid {
  display: grid;
  gap: 12px;
}

.metric-grid {
  grid-template-columns: repeat(2, minmax(0, 1fr));
}

.score-grid {
  grid-template-columns: repeat(3, minmax(0, 1fr));
}

.metric-tile,
.score-tile {
  border-radius: 18px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  background:
    linear-gradient(180deg, rgba(255, 255, 255, 0.055), rgba(255, 255, 255, 0.025));
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
  transition: border-color 160ms ease, background 160ms ease, transform 160ms ease, box-shadow 160ms ease;
}

.metric-tile {
  padding: 15px;
  min-height: 94px;
}

.score-tile {
  padding: 18px;
}

.metric-tile:hover,
.score-tile:hover {
  transform: translateY(-1px);
  border-color: rgba(255, 255, 255, 0.18);
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.07),
    0 14px 28px rgba(2, 9, 20, 0.16);
}

.metric-tile.is-winner {
  border-color: var(--win-border);
  background:
    linear-gradient(180deg, rgba(94, 225, 139, 0.08), rgba(94, 225, 139, 0.025)),
    rgba(255, 255, 255, 0.035);
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.08),
    0 12px 26px rgba(16, 185, 129, 0.08);
}

.metric-label,
.score-label {
  color: var(--muted);
  font-size: 0.76rem;
  text-transform: uppercase;
  letter-spacing: 0;
  font-weight: 700;
}

.metric-value {
  margin-top: 8px;
  color: #f8fbff;
  font-size: 1.34rem;
  font-weight: 800;
}

.metric-value.is-winner {
  color: var(--win);
  text-shadow: 0 0 22px rgba(97, 242, 154, 0.22);
}

.metric-foot,
.score-foot {
  margin-top: 6px;
  color: var(--muted);
  font-size: 0.88rem;
  line-height: 1.55;
}

.score-value {
  margin-top: 10px;
  font-size: 1.7rem;
  font-weight: 800;
}

.score-tile.is-sparse-winner {
  border-color: var(--win-border);
  background:
    linear-gradient(180deg, rgba(97, 242, 154, 0.15), rgba(97, 242, 154, 0.045)),
    rgba(255, 255, 255, 0.035);
}

.score-tile.is-dense-winner {
  border-color: rgba(117, 184, 255, 0.42);
  background:
    linear-gradient(180deg, rgba(117, 184, 255, 0.13), rgba(117, 184, 255, 0.035)),
    rgba(255, 255, 255, 0.035);
}

.winner-dense {
  color: var(--dense);
}

.winner-sparse {
  color: var(--win);
  text-shadow: 0 0 24px rgba(97, 242, 154, 0.22);
}

.winner-neutral {
  color: #f8fbff;
}

.log-entry {
  margin: 0 0 10px;
}

.log-entry strong {
  color: var(--muted-strong);
}

#prompt-box textarea,
#dense-box textarea,
#sparse-box textarea {
  border-radius: 18px !important;
  border: 1px solid rgba(255, 255, 255, 0.10) !important;
  background: linear-gradient(180deg, rgba(4, 12, 24, 0.58), rgba(9, 19, 35, 0.92)) !important;
  color: #f8fbff !important;
  font-size: 1rem !important;
  line-height: 1.72 !important;
  padding: 18px 20px !important;
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.04),
    0 8px 26px rgba(2, 9, 20, 0.16) !important;
}

#prompt-box textarea {
  min-height: 220px !important;
}

.prompt-meta {
  display: flex;
  justify-content: flex-end;
  margin: 8px 4px 0;
  color: rgba(232, 241, 255, 0.66);
  font-size: 0.84rem;
  line-height: 1.4;
}

.prompt-counter.is-warning {
  color: #f6d68c;
}

.prompt-counter.is-over {
  color: #ff8f8f;
  font-weight: 800;
}

#dense-box textarea,
#sparse-box textarea {
  min-height: 430px !important;
}

.gradio-container button,
.gr-button-primary,
.gr-button-secondary {
  min-height: 52px !important;
  border-radius: 14px !important;
  font-weight: 800 !important;
}

.gr-button-primary {
  background: linear-gradient(135deg, #357fe8, #2563eb) !important;
  box-shadow: 0 18px 36px rgba(46, 134, 255, 0.24);
}

.gr-button-secondary {
  color: #eaf2ff !important;
  background: linear-gradient(180deg, rgba(20, 35, 58, 0.92), rgba(14, 28, 47, 0.94)) !important;
  border: 1px solid rgba(255, 255, 255, 0.12) !important;
}

.gradio-container .plot-container,
.gradio-container .table-wrap {
  border-radius: 18px !important;
  border: 1px solid rgba(255, 255, 255, 0.10) !important;
  background: linear-gradient(180deg, rgba(7, 17, 31, 0.72), rgba(9, 19, 35, 0.90)) !important;
  box-shadow: none !important;
  backdrop-filter: blur(18px);
}

.table-shell .table-wrap {
  overflow-x: auto !important;
  overflow-y: hidden !important;
  -webkit-overflow-scrolling: touch;
  touch-action: pan-x pan-y;
}

.comparison-scroll {
  overflow-x: auto;
  overflow-y: hidden;
  -webkit-overflow-scrolling: touch;
  touch-action: pan-x pan-y;
  border-radius: 18px;
  background: linear-gradient(180deg, rgba(18, 34, 57, 0.68), rgba(10, 20, 37, 0.74));
  box-shadow:
    inset 0 0 0 1px rgba(255, 255, 255, 0.05),
    inset 0 1px 0 rgba(255, 255, 255, 0.03),
    0 14px 34px rgba(2, 9, 20, 0.12);
  backdrop-filter: blur(18px);
  scrollbar-width: thin;
  scrollbar-color: rgba(105, 167, 255, 0.72) rgba(7, 17, 31, 0.72);
}

.comparison-scroll::-webkit-scrollbar {
  height: 12px;
}

.comparison-scroll::-webkit-scrollbar-track {
  background: rgba(7, 17, 31, 0.72);
  border-radius: 999px;
}

.comparison-scroll::-webkit-scrollbar-thumb {
  background: linear-gradient(135deg, rgba(85, 150, 255, 0.95), rgba(46, 196, 214, 0.88));
  border-radius: 999px;
  border: 2px solid rgba(7, 17, 31, 0.72);
}

.comparison-grid {
  width: 100%;
  border-collapse: collapse;
  color: #edf4ff;
  font-family: "Manrope", "Segoe UI Variable", "Helvetica Neue", sans-serif;
  background: transparent;
}

.comparison-grid th,
.comparison-grid td {
  padding: 16px 14px;
  text-align: left;
  white-space: nowrap;
  border: none;
}

.comparison-grid th {
  background: rgba(255, 255, 255, 0.035);
  color: #f4f8ff;
  font-size: 0.94rem;
  font-weight: 700;
  letter-spacing: 0;
  text-transform: none;
  border-bottom: 1px solid rgba(255, 255, 255, 0.07);
}

.comparison-grid td {
  background: transparent;
  color: #dce8f8;
  border-top: 1px solid rgba(255, 255, 255, 0.05);
}

.comparison-grid td:first-child {
  color: #f4f8ff;
  font-weight: 700;
}

.comparison-grid td.comparison-definition {
  min-width: 260px;
  max-width: 360px;
  white-space: normal;
  color: rgba(220, 232, 248, 0.78);
  line-height: 1.45;
}

.comparison-value {
  font-weight: 800;
}

.comparison-value.is-winner {
  color: var(--win) !important;
  text-shadow: 0 0 18px rgba(97, 242, 154, 0.18);
}

.comparison-badge {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 96px;
  padding: 7px 10px;
  border-radius: 12px;
  border: 1px solid rgba(255, 255, 255, 0.11);
  color: #eaf2ff;
  background: rgba(255, 255, 255, 0.055);
  font-size: 0.78rem;
  font-weight: 900;
}

.comparison-badge.is-sparse {
  color: #211602;
  border-color: rgba(242, 198, 109, 0.70);
  background: linear-gradient(135deg, #f8d991, #f0b95e);
  box-shadow: 0 10px 24px rgba(242, 198, 109, 0.14);
}

.comparison-badge.is-dense {
  color: #06172b;
  border-color: rgba(130, 183, 255, 0.64);
  background: linear-gradient(135deg, #b7d9ff, #82b7ff);
}

.comparison-badge.is-neutral {
  color: rgba(234, 242, 255, 0.80);
}

.comparison-change {
  color: rgba(232, 241, 255, 0.72) !important;
  font-weight: 800;
}

.comparison-change.is-sparse {
  color: var(--win) !important;
}

.comparison-empty {
  color: var(--muted);
  text-align: center;
  padding: 26px 18px !important;
}

.gradio-container table {
  overflow: hidden !important;
  border-radius: 24px !important;
  color: #edf4ff !important;
  font-family: "Manrope", "Segoe UI Variable", "Helvetica Neue", sans-serif !important;
}

.table-shell,
.table-shell * {
  font-family: "Manrope", "Segoe UI Variable", "Helvetica Neue", sans-serif !important;
}

.gradio-container th {
  background: rgba(18, 33, 56, 0.96) !important;
  color: #edf4ff !important;
  font-family: "Manrope", "Segoe UI Variable", "Helvetica Neue", sans-serif !important;
}

.gradio-container td {
  background: rgba(10, 21, 39, 0.88) !important;
  color: #dce8f8 !important;
  font-family: "Manrope", "Segoe UI Variable", "Helvetica Neue", sans-serif !important;
}

.table-shell .comparison-grid th {
  background: rgba(255, 255, 255, 0.035) !important;
  color: #f4f8ff !important;
  font-size: 0.94rem !important;
  font-weight: 700 !important;
  letter-spacing: 0 !important;
  text-transform: none !important;
  border: none !important;
  border-bottom: 1px solid rgba(255, 255, 255, 0.07) !important;
}

.table-shell .comparison-grid th + th {
  box-shadow: inset 1px 0 0 rgba(255, 255, 255, 0.05);
}

.table-shell .comparison-grid th.model-dense {
  color: var(--dense) !important;
}

.table-shell .comparison-grid th.model-sparse {
  color: var(--sparse) !important;
}

.table-shell .comparison-grid td {
  background: transparent !important;
  color: #dce8f8 !important;
  border: none !important;
  border-top: 1px solid rgba(255, 255, 255, 0.05) !important;
}

.table-shell .comparison-grid tbody tr:nth-child(even) td {
  background: rgba(255, 255, 255, 0.02) !important;
}

.table-shell .comparison-grid tbody tr:hover td {
  background: rgba(89, 153, 255, 0.07) !important;
}

.table-shell .comparison-grid tbody tr:first-child td {
  border-top: 1px solid rgba(255, 255, 255, 0.05) !important;
}

.table-shell .comparison-grid td:first-child {
  color: #f4f8ff !important;
  font-weight: 700 !important;
}

.table-shell .comparison-grid td.comparison-definition {
  color: rgba(220, 232, 248, 0.78) !important;
}

.table-shell .comparison-grid .comparison-value {
  font-weight: 800 !important;
}

.table-shell .comparison-grid .comparison-value.is-winner,
.table-shell .comparison-grid .comparison-change.is-sparse {
  color: var(--win) !important;
  text-shadow: 0 0 18px rgba(97, 242, 154, 0.18);
}

.table-shell .comparison-grid .comparison-badge.is-sparse {
  color: #211602 !important;
}

.table-shell .comparison-grid .comparison-badge.is-dense {
  color: #06172b !important;
}

.table-shell .comparison-grid .comparison-badge.is-neutral {
  color: rgba(234, 242, 255, 0.80) !important;
}

.table-shell .comparison-empty {
  background: transparent !important;
  color: var(--muted) !important;
}

.gradio-container label,
.gradio-container .prose,
.gradio-container .wrap,
.gradio-container .message,
.gradio-container .table-wrap *,
.gradio-container .plot-container * {
  color: #edf4ff !important;
}

.gradio-container footer {
  display: none !important;
}

.toast-wrap .toast-body.warning {
  display: none !important;
}

@media (max-width: 1180px) {
  .hero-shell {
    min-height: auto;
  }

  .metric-grid,
  .score-grid {
    grid-template-columns: 1fr;
  }

  .hero-title span {
    white-space: normal;
  }

  .hero-subtitle {
    white-space: normal;
  }

  .pwa-shell {
    position: static;
    margin-top: 0;
    width: min(220px, 100%);
  }

  .protocol-grid {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 900px) {
  .gradio-container {
    padding: 14px !important;
  }

  .hero-shell,
  .control-shell,
  .panel-card,
  .metric-shell,
  .score-shell,
  .log-shell,
  .viz-shell,
  .table-shell {
    padding: 20px;
    border-radius: 28px;
  }

  .hero-shell {
    padding: 28px 22px;
  }

  .hero-title {
    font-size: 2.15rem;
    line-height: 1.04;
  }

  .hero-subtitle {
    font-size: 0.96rem;
    line-height: 1.55;
    margin-bottom: 16px;
  }

  .hero-tags {
    gap: 8px;
  }

  .hero-tag {
    padding: 9px 12px;
    font-size: 0.78rem;
  }

  .pwa-shell {
    width: 100%;
    max-width: none;
  }

  .table-shell table {
    min-width: 1040px !important;
    width: max-content !important;
  }

  .comparison-grid {
    min-width: 1040px;
    width: max-content;
  }

  .action-row {
    flex-direction: column !important;
  }

  .action-row > * {
    width: 100% !important;
  }

  .action-panel {
    width: 100% !important;
  }

  #benchmark-task-radio .wrap {
    width: 100%;
  }

  #benchmark-task-radio .wrap label {
    flex: 1 1 100%;
    justify-content: center;
    white-space: normal;
  }

  #prompt-box textarea {
    min-height: 180px !important;
  }

  .action-row {
    gap: 10px;
  }

  #dense-box textarea,
  #sparse-box textarea {
    min-height: 280px !important;
  }

  .table-shell {
    padding: 16px;
  }

  .table-shell .table-wrap {
    margin: 0 -2px;
  }
}
"""

HEAD = """
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover" />
<link rel="manifest" href="/manifest.webmanifest" />
<meta name="theme-color" content="#07111f" />
<meta name="apple-mobile-web-app-capable" content="yes" />
<meta name="mobile-web-app-capable" content="yes" />
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent" />
<script>
(() => {
  let deferredPrompt = null;

  function updatePwaStatus(message) {
    const node = document.getElementById("pwa-install-status");
    if (node) {
      node.textContent = message;
    }
  }

  window.capstoneInstallPwa = async () => {
    if (!deferredPrompt) {
      const secureContext = window.isSecureContext || ["localhost", "127.0.0.1"].includes(window.location.hostname);
      updatePwaStatus(
        secureContext
          ? "Install prompt not ready yet. Open this in Chrome or Edge, then refresh once."
          : "Install requires HTTPS or localhost. A plain remote HTTP URL will not show the install prompt."
      );
      return;
    }
    deferredPrompt.prompt();
    const choice = await deferredPrompt.userChoice;
    updatePwaStatus(choice.outcome === "accepted" ? "App installed." : "Install dismissed.");
    deferredPrompt = null;
  };

  window.addEventListener("beforeinstallprompt", (event) => {
    event.preventDefault();
    deferredPrompt = event;
    updatePwaStatus("Install is available on this device.");
  });

  window.addEventListener("appinstalled", () => {
    updatePwaStatus("App installed.");
    deferredPrompt = null;
  });

  if ("serviceWorker" in navigator) {
    window.addEventListener("load", () => {
      navigator.serviceWorker.register("/service-worker.js").catch(() => {
        updatePwaStatus("Service worker registration failed.");
      });
    });
  }

})();
</script>
"""


def add_bool_argument(parser: argparse.ArgumentParser, name: str, default: bool) -> None:
    dest = name.replace("-", "_")
    option = f"--{name}"
    if hasattr(argparse, "BooleanOptionalAction"):
        parser.add_argument(option, action=argparse.BooleanOptionalAction, default=default)
        return
    group = parser.add_mutually_exclusive_group()
    group.add_argument(option, dest=dest, action="store_true")
    group.add_argument(f"--no-{name}", dest=dest, action="store_false")
    parser.set_defaults(**{dest: default})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PWA Gradio app for Llama 3.2 summarization model comparison.")
    parser.add_argument("--dense-model", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--sparse-model", default="./pruned-llama32-3b-instruct")
    parser.add_argument("--dense-label", default="Original Llama 3.2 3B Instruct")
    parser.add_argument("--sparse-label", default="Pruned Llama 3.2 3B Instruct")
    parser.add_argument("--dense-device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sparse-device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--dense-dtype", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--sparse-dtype", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--hf-cache-dir", default="./.hf_cache")
    parser.add_argument(
        "--tokenizer-model",
        default=None,
        help="Shared tokenizer/chat-template source. Defaults to --dense-model so both models receive identical prompt token IDs.",
    )
    parser.add_argument("--server-name", default="0.0.0.0")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--max-new-tokens-cap", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--max-input-chars", type=int, default=DEFAULT_MAX_INPUT_CHARS)
    add_bool_argument(parser, "reload", default=False)
    add_bool_argument(parser, "share", default=True)
    return parser.parse_args()


def resolve_model_source(model_name: str, hf_cache_dir: str) -> str:
    model_path = Path(model_name).expanduser()
    if model_path.exists():
        return str(model_path)

    repo_cache_dir = Path(hf_cache_dir) / f"models--{model_name.replace('/', '--')}"
    ref_path = repo_cache_dir / "refs" / "main"
    if not ref_path.is_file():
        return model_name

    snapshot_name = ref_path.read_text().strip()
    if not snapshot_name:
        return model_name

    snapshot_dir = repo_cache_dir / "snapshots" / snapshot_name
    has_config = (snapshot_dir / "config.json").is_file()
    has_weights = any(
        (snapshot_dir / filename).is_file()
        for filename in (
            "model.safetensors.index.json",
            "model.safetensors",
            "pytorch_model.bin.index.json",
            "pytorch_model.bin",
        )
    )
    if has_config and has_weights:
        return str(snapshot_dir)
    return model_name


def resolve_dtype(name: str) -> torch.dtype:
    if name == "auto":
        return base.get_torch_dtype()
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def resolve_sparse_dtype_name(args: argparse.Namespace) -> str:
    if args.sparse_dtype:
        return args.sparse_dtype
    if args.dtype:
        return args.dtype
    return "auto"


def parse_device(device: str) -> Tuple[str, Any]:
    normalized = device.strip().lower()
    if normalized == "cuda":
        normalized = "cuda:0"
    if normalized.startswith("cuda:"):
        index = int(normalized.split(":", 1)[1])
        return normalized, index
    if normalized == "cpu":
        return normalized, "cpu"
    raise ValueError(f"Unsupported device string: {device!r}. Use cpu or cuda:N.")


def maybe_cache_dir(model_name: str, hf_cache_dir: str) -> Dict[str, Any]:
    if Path(model_name).expanduser().exists():
        return {}
    return {"cache_dir": hf_cache_dir}


def load_chat_template_if_needed(tokenizer: AutoTokenizer, model_source: str) -> None:
    if getattr(tokenizer, "chat_template", None):
        return
    template_path = Path(model_source).expanduser() / "chat_template.jinja"
    if template_path.is_file():
        tokenizer.chat_template = template_path.read_text(encoding="utf-8")


def force_module_weight(module: Optional[torch.nn.Module], weight: torch.nn.Parameter) -> None:
    if module is None:
        return
    if hasattr(module, "_parameters"):
        module._parameters["weight"] = weight
    try:
        if getattr(module, "weight", None) is not weight:
            module.weight = weight
    except Exception:
        pass


def ensure_tied_lm_head(model: torch.nn.Module) -> None:
    """Tie lm_head.weight to the loaded input embedding when checkpoints omit it."""
    try:
        input_embeddings = model.get_input_embeddings()
        output_embeddings = model.get_output_embeddings()
    except Exception:
        return
    if input_embeddings is None:
        return

    input_weight = getattr(input_embeddings, "weight", None)
    if input_weight is None or getattr(input_weight, "is_meta", False):
        return

    output_weight = getattr(output_embeddings, "weight", None) if output_embeddings is not None else None
    config_ties_weights = bool(getattr(getattr(model, "config", None), "tie_word_embeddings", True))
    output_is_missing = output_weight is None or getattr(output_weight, "is_meta", False)
    if not config_ties_weights and not output_is_missing:
        return

    force_module_weight(output_embeddings, input_weight)
    force_module_weight(getattr(model, "lm_head", None), input_weight)

    try:
        model.tie_weights()
    except Exception:
        pass

    # Some Transformers/Accelerate paths can recreate lm_head.weight after
    # tie_weights(); enforce the shared non-meta parameter as the final step.
    input_embeddings = model.get_input_embeddings()
    input_weight = getattr(input_embeddings, "weight", None) if input_embeddings is not None else None
    if input_weight is None or getattr(input_weight, "is_meta", False):
        return
    force_module_weight(model.get_output_embeddings(), input_weight)
    force_module_weight(getattr(model, "lm_head", None), input_weight)


def is_tied_lm_head_meta(model: torch.nn.Module, name: str, param: torch.nn.Parameter) -> bool:
    if name != "lm_head.weight" or not getattr(param, "is_meta", False):
        return False
    try:
        input_embeddings = model.get_input_embeddings()
        output_embeddings = model.get_output_embeddings()
    except Exception:
        return False
    input_weight = getattr(input_embeddings, "weight", None) if input_embeddings is not None else None
    output_weight = getattr(output_embeddings, "weight", None) if output_embeddings is not None else None
    lm_head = getattr(model, "lm_head", None)
    lm_head_weight = getattr(lm_head, "weight", None) if lm_head is not None else None
    return (
        input_weight is not None
        and not getattr(input_weight, "is_meta", False)
        and output_weight is input_weight
        and lm_head_weight is input_weight
    )


def find_meta_tensors(model: torch.nn.Module) -> List[str]:
    ensure_tied_lm_head(model)
    meta_tensors = [
        name
        for name, param in model.named_parameters()
        if getattr(param, "is_meta", False) and not is_tied_lm_head_meta(model, name, param)
    ]
    meta_tensors.extend(
        name for name, buffer in model.named_buffers() if getattr(buffer, "is_meta", False)
    )
    return meta_tensors


def has_hf_device_map(model: torch.nn.Module) -> bool:
    return getattr(model, "hf_device_map", None) is not None


def format_number(value: Optional[float], precision: int = 2, suffix: str = "") -> str:
    if value is None:
        return "--"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "--"
    return f"{value:.{precision}f}{suffix}"


def finite_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def compare_metric_values(value: Any, other_value: Any, direction: str) -> Optional[int]:
    current = finite_float(value)
    other = finite_float(other_value)
    if current is None or other is None:
        return None
    if math.isclose(current, other, rel_tol=1e-9, abs_tol=1e-9):
        return 0
    if direction == "higher":
        return 1 if current > other else -1
    if direction == "lower":
        return 1 if current < other else -1
    return None


def metric_value_class(value: Any, other_value: Any, direction: str) -> str:
    outcome = compare_metric_values(value, other_value, direction)
    return "metric-value is-winner" if outcome == 1 else "metric-value"


def metric_tile_class(value: Any, other_value: Any, direction: str) -> str:
    outcome = compare_metric_values(value, other_value, direction)
    return "metric-tile is-winner" if outcome == 1 else "metric-tile"


def render_prompt_counter(prompt_text: Optional[str], max_chars: int) -> str:
    text = prompt_text or ""
    words = len(text.strip().split()) if text.strip() else 0
    chars = len(text)
    css_classes = ["prompt-counter"]
    if max_chars > 0 and chars > max_chars:
        css_classes.append("is-over")
    elif max_chars > 0 and chars >= max_chars * 0.9:
        css_classes.append("is-warning")
    return f"""
    <div class="prompt-meta">
      <span id="prompt-counter" class="{' '.join(css_classes)}">
        {words:,} words · {chars:,} / {max_chars:,} characters
      </span>
    </div>
    """


def empty_timeseries_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["elapsed_s", "value", "model"])


def empty_metrics_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["Metric", "Definition", "Dense", "Sparse", "Winner", "Delta"])


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    model_name: str
    device: str
    dtype_name: str
    mode: str
    prune_masks_path: Optional[str] = None

    def cache_key(self) -> Tuple[str, str, str, str, Optional[str]]:
        return (self.model_name, self.device, self.dtype_name, self.mode, self.prune_masks_path)


@dataclass
class LoadedBundle:
    spec: ModelSpec
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    load_info: Dict[str, Any]


@dataclass
class RunState:
    spec: ModelSpec
    status: str = "Waiting"
    response_text: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    timeline: List[Dict[str, Any]] = field(default_factory=list)
    load_info: Dict[str, Any] = field(default_factory=dict)
    completed: bool = False
    errored: bool = False
    error_text: Optional[str] = None


class ModelManager:
    def __init__(self, hf_cache_dir: str, tokenizer_model: Optional[str] = None):
        self.hf_cache_dir = hf_cache_dir
        self.tokenizer_model = tokenizer_model
        self._lock = threading.Lock()
        self._bundles: Dict[Tuple[str, str, str, str, Optional[str]], LoadedBundle] = {}
        self._inflight: Dict[Tuple[str, str, str, str, Optional[str]], threading.Event] = {}
        self._failures: Dict[Tuple[str, str, str, str, Optional[str]], Exception] = {}

    def get_or_load(self, spec: ModelSpec) -> LoadedBundle:
        key = spec.cache_key()
        while True:
            with self._lock:
                bundle = self._bundles.get(key)
                if bundle is not None:
                    return bundle

                failure = self._failures.get(key)
                if failure is not None:
                    raise failure

                wait_event = self._inflight.get(key)
                if wait_event is None:
                    wait_event = threading.Event()
                    self._inflight[key] = wait_event
                    should_load = True
                else:
                    should_load = False

            if should_load:
                try:
                    bundle = self._load_bundle(spec)
                except Exception as exc:
                    with self._lock:
                        self._failures[key] = exc
                        event = self._inflight.pop(key, None)
                    if event is not None:
                        event.set()
                    raise

                with self._lock:
                    self._bundles[key] = bundle
                    self._failures.pop(key, None)
                    event = self._inflight.pop(key, None)
                if event is not None:
                    event.set()
                return bundle

            wait_event.wait()

    def _load_bundle(self, spec: ModelSpec) -> LoadedBundle:
        requested_dtype: Optional[torch.dtype]
        dense_dtype_auto = spec.mode == "dense" and spec.dtype_name == "auto"
        if spec.dtype_name == "auto":
            requested_dtype = None if dense_dtype_auto else base.get_torch_dtype()
        else:
            requested_dtype = resolve_dtype(spec.dtype_name)
        model_source = resolve_model_source(spec.model_name, self.hf_cache_dir)
        tokenizer_name = self.tokenizer_model or spec.model_name
        tokenizer_source = resolve_model_source(tokenizer_name, self.hf_cache_dir)
        tokenizer_kwargs = {"use_fast": True, "fix_mistral_regex": True}
        tokenizer_kwargs.update(maybe_cache_dir(tokenizer_source, self.hf_cache_dir))
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, **tokenizer_kwargs)
        load_chat_template_if_needed(tokenizer, tokenizer_source)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        normalized_device, device_target = parse_device(spec.device)
        use_device_map = normalized_device != "cpu"
        load_kwargs: Dict[str, Any] = {
            "low_cpu_mem_usage": use_device_map,
            "dtype": "auto" if spec.dtype_name == "auto" else requested_dtype,
        }
        if use_device_map:
            load_kwargs["device_map"] = {"": device_target}
        load_kwargs.update(maybe_cache_dir(model_source, self.hf_cache_dir))

        def clear_partial_model(partial_model: Optional[torch.nn.Module]) -> None:
            if partial_model is not None:
                del partial_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        def load_with_to_empty(reason: str) -> torch.nn.Module:
            checkpoint_dir = Path(model_source)
            if not checkpoint_dir.exists():
                resolved = resolve_model_source(spec.model_name, self.hf_cache_dir)
                checkpoint_dir = Path(resolved)
            if not checkpoint_dir.exists():
                raise RuntimeError(
                    "Cannot recover from meta tensor loading because no local checkpoint directory "
                    f"was found for {spec.model_name!r}. Last failure: {reason}"
                )

            from accelerate import init_empty_weights
            from transformers.modeling_utils import load_sharded_checkpoint

            config_kwargs = maybe_cache_dir(model_source, self.hf_cache_dir)
            config = AutoConfig.from_pretrained(str(checkpoint_dir), **config_kwargs)
            dtype_for_init = requested_dtype
            if dtype_for_init is None and spec.dtype_name == "auto":
                dtype_for_init = getattr(config, "torch_dtype", None)
                if isinstance(dtype_for_init, str):
                    dtype_for_init = getattr(torch, dtype_for_init.split(".")[-1], None)
            from_config_kwargs: Dict[str, Any] = {}
            if dtype_for_init is not None:
                from_config_kwargs["torch_dtype"] = dtype_for_init

            with init_empty_weights():
                empty_model = AutoModelForCausalLM.from_config(config, **from_config_kwargs)
            target_device = torch.device(normalized_device if normalized_device != "cpu" else "cpu")
            empty_model.to_empty(device=target_device)
            if (checkpoint_dir / "model.safetensors.index.json").is_file() or (
                checkpoint_dir / "pytorch_model.bin.index.json"
            ).is_file():
                load_sharded_checkpoint(empty_model, str(checkpoint_dir), strict=False, prefer_safe=True)
            elif (checkpoint_dir / "model.safetensors").is_file():
                from safetensors.torch import load_file as safe_load_file

                state_dict = safe_load_file(str(checkpoint_dir / "model.safetensors"), device="cpu")
                empty_model.load_state_dict(state_dict, strict=False)
                del state_dict
            elif (checkpoint_dir / "pytorch_model.bin").is_file():
                state_dict = torch.load(checkpoint_dir / "pytorch_model.bin", map_location="cpu")
                empty_model.load_state_dict(state_dict, strict=False)
                del state_dict
            else:
                raise RuntimeError(f"No supported checkpoint weights found in {checkpoint_dir}")
            
            # Tie weights after the checkpoint tensors have been materialized.
            ensure_tied_lm_head(empty_model)

            # Final pass to ensure lm_head.weight is not on meta.
            ensure_tied_lm_head(empty_model)

            if normalized_device == "cpu" and requested_dtype is not None:
                empty_model = empty_model.to(dtype=requested_dtype)
                ensure_tied_lm_head(empty_model)
            return empty_model

        start_time = time.perf_counter()
        model: Optional[torch.nn.Module]
        try:
            model = AutoModelForCausalLM.from_pretrained(model_source, **load_kwargs)
        except NotImplementedError as exc:
            if "Cannot copy out of meta tensor" not in str(exc):
                raise
            model = load_with_to_empty("normal full checkpoint load hit a meta tensor copy")
        
        # Verify and fix meta tensors before any normal .to() call.
        meta_tensors = find_meta_tensors(model)
        if meta_tensors:
            partial_model = model
            model = None
            clear_partial_model(partial_model)
            model = load_with_to_empty(f"{len(meta_tensors)} tensors stayed on meta")
            meta_tensors = find_meta_tensors(model)

        if meta_tensors:
            sample = ", ".join(meta_tensors[:5])
            raise RuntimeError(
                "Model checkpoint did not fully materialize after to_empty retry. "
                f"{len(meta_tensors)} tensor(s) are still on the meta device: {sample}"
            )
        
        model.tie_weights()
        ensure_tied_lm_head(model)
        
        if normalized_device == "cpu" and requested_dtype is not None:
            model = model.to(dtype=requested_dtype)
        elif not has_hf_device_map(model):
            model = model.to(device=torch.device(normalized_device))
        
        model.tie_weights()
        ensure_tied_lm_head(model)
        model.eval()
        model.config.use_cache = True
        if getattr(model.config, "pad_token_id", None) is None:
            model.config.pad_token_id = tokenizer.pad_token_id

        conversion_info: Dict[str, Any] = {}

        runtime_dtype = "unknown"
        try:
            runtime_dtype = str(next(model.parameters()).dtype)
        except StopIteration:
            if requested_dtype is not None:
                runtime_dtype = str(requested_dtype)

        load_seconds = time.perf_counter() - start_time
        load_info = {
            "model_source": model_source,
            "tokenizer_source": tokenizer_source,
            "device": normalized_device,
            "dtype": runtime_dtype,
            "model_load_time_s": load_seconds,
            "mode": spec.mode,
            "memory_footprint_gb": base.get_model_memory_footprint_bytes(model) / GB,
            "conversion_info": conversion_info,
        }
        return LoadedBundle(spec=spec, model=model, tokenizer=tokenizer, load_info=load_info)

    def clear(self) -> None:
        with self._lock:
            bundles = list(self._bundles.values())
            self._bundles.clear()
            self._inflight.clear()
            self._failures.clear()
        for bundle in bundles:
            del bundle.model
            del bundle.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class MetricsStreamer(TextIteratorStreamer):
    def __init__(self, tokenizer: AutoTokenizer, start_time: float, timeout: float = 0.2):
        super().__init__(tokenizer, skip_prompt=True, timeout=timeout, skip_special_tokens=True)
        self.start_time = start_time
        self.first_token_at: Optional[float] = None
        self.generated_tokens = 0
        self.timeline: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def put(self, value):
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("MetricsStreamer only supports batch size 1.")
        if len(value.shape) > 1:
            value = value[0]

        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        now = time.perf_counter()
        new_tokens = int(value.numel())
        with self._lock:
            self.generated_tokens += new_tokens
            if self.first_token_at is None and new_tokens > 0:
                self.first_token_at = now
            decode_tok_s = None
            if self.first_token_at is not None and self.generated_tokens > 1:
                decode_elapsed = max(now - self.first_token_at, 1e-9)
                decode_tok_s = (self.generated_tokens - 1) / decode_elapsed
            self.timeline.append(
                {
                    "elapsed_s": now - self.start_time,
                    "tokens": self.generated_tokens,
                    "decode_tok_s": decode_tok_s,
                }
            )

        self.token_cache.extend(value.tolist())
        text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)
        if text.endswith("\n"):
            printable_text = text[self.print_len :]
            self.token_cache = []
            self.print_len = 0
        elif len(text) > 0 and self._is_chinese_char(ord(text[-1])):
            printable_text = text[self.print_len :]
            self.print_len += len(printable_text)
        else:
            printable_text = text[self.print_len : text.rfind(" ") + 1]
            self.print_len += len(printable_text)
        self.on_finalized_text(printable_text)

    def snapshot(self) -> Dict[str, Any]:
        now = time.perf_counter()
        with self._lock:
            first_token_at = self.first_token_at
            generated_tokens = self.generated_tokens
            timeline = list(self.timeline)

        ttft_ms = None
        decode_tok_s = None
        avg_itl_ms = None
        if first_token_at is not None:
            ttft_ms = (first_token_at - self.start_time) * 1000.0
            if generated_tokens > 1:
                decode_elapsed = max(now - first_token_at, 1e-9)
                decode_tok_s = (generated_tokens - 1) / decode_elapsed
                avg_itl_ms = (decode_elapsed / (generated_tokens - 1)) * 1000.0

        return {
            "elapsed_s": now - self.start_time,
            "generated_tokens": generated_tokens,
            "ttft_ms": ttft_ms,
            "decode_tok_s": decode_tok_s,
            "avg_itl_ms": avg_itl_ms,
            "timeline": timeline,
        }


def build_specs(
    dense_model: str,
    sparse_model: str,
    dense_label: str,
    sparse_label: str,
    dense_device: str,
    sparse_device: str,
    dense_dtype_name: str,
    sparse_dtype_name: str,
) -> List[ModelSpec]:
    return [
        ModelSpec(
            key="dense",
            label=dense_label.strip() or "Dense",
            model_name=dense_model.strip(),
            device=dense_device.strip(),
            dtype_name=dense_dtype_name,
            mode="dense",
        ),
        ModelSpec(
            key="sparse",
            label=sparse_label.strip() or "Sparse",
            model_name=sparse_model.strip(),
            device=sparse_device.strip(),
            dtype_name=sparse_dtype_name,
            mode="pruned",
        ),
    ]


def safe_peak_vram_gb(device: str) -> Optional[float]:
    if not torch.cuda.is_available():
        return None
    normalized, target = parse_device(device)
    if normalized.startswith("cuda:"):
        return torch.cuda.max_memory_allocated(int(target)) / GB
    return None


def reset_peak_vram(device: str) -> None:
    if not torch.cuda.is_available():
        return
    normalized, target = parse_device(device)
    if normalized.startswith("cuda:"):
        torch.cuda.reset_peak_memory_stats(int(target))


def sync_torch_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def inputs_on_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def get_model_runtime_dtype(model: torch.nn.Module) -> Optional[torch.dtype]:
    try:
        return next(model.parameters()).dtype
    except StopIteration:
        for _, buffer in model.named_buffers():
            return buffer.dtype
    return None


def render_hero() -> str:
    return f"""
    <section class="hero-shell">
      <div class="hero-copy">
        <div class="hero-topbar">
          <p class="hero-eyebrow">Model Compression Evaluation</p>
        </div>
        <h1 class="hero-title">{APP_TITLE_DISPLAY}</h1>
        <p class="hero-subtitle">{html.escape(APP_SUBTITLE)}</p>
        <div class="hero-tags">
          <span class="hero-tag">Original baseline</span>
          <span class="hero-tag">Pruned checkpoint</span>
          <span class="hero-tag">Latency, throughput, VRAM</span>
        </div>
      </div>
      <div class="pwa-shell">
        <p class="pwa-title">Progressive Web App</p>
        <button class="pwa-btn" onclick="window.capstoneInstallPwa()">Install PWA</button>
        <div class="pwa-status" id="pwa-install-status"></div>
      </div>
    </section>
    """


def model_identity_class(state: RunState) -> str:
    if state.spec.key == "dense":
        return "model-dense"
    if state.spec.key == "sparse":
        return "model-sparse"
    return ""


def render_model_overview(state: RunState) -> str:
    status = html.escape(state.status)
    source = html.escape(state.spec.model_name)
    device = html.escape(state.spec.device)
    dtype_name = html.escape(str(state.load_info.get("dtype", state.spec.dtype_name)))
    model_class = model_identity_class(state)
    return f"""
    <section class="panel-card">
      <div class="panel-title">
        <span class="panel-name {model_class}">{html.escape(state.spec.label)}</span>
      </div>
      <p class="panel-line"><strong>Status:</strong> {status}</p>
      <p class="panel-line"><strong>Source:</strong> {source}</p>
      <p class="panel-line"><strong>Device:</strong> {device}</p>
      <p class="panel-line"><strong>Runtime dtype:</strong> {dtype_name}</p>
    </section>
    """


def render_metrics_card(state: RunState, other_state: Optional[RunState] = None) -> str:
    metrics = state.metrics
    load_info = state.load_info
    model_class = model_identity_class(state)
    tiles = []
    for spec in METRIC_DISPLAY_SPECS:
        value = metrics.get(spec["key"])
        other_value = other_state.metrics.get(spec["key"]) if other_state else None
        direction = spec.get("direction", "equal")
        tiles.append(
            (
                spec["label"],
                format_number(value, spec["precision"], spec["suffix"]),
                metric_tile_class(value, other_value, direction),
                metric_value_class(value, other_value, direction),
            )
        )
    load_line = ""
    if load_info:
        load_line = (
            f"<p class=\"metric-foot\">Model loaded in {format_number(load_info.get('model_load_time_s'), 2, ' s')}. "
            f"Loaded model footprint {format_number(load_info.get('memory_footprint_gb'), 2, ' GB')}.</p>"
        )
    stop_reason = metrics.get("stop_reason")
    stop_line = ""
    if stop_reason == "length":
        stop_line = "<p class=\"metric-foot\">Stopped because the output-token cap was reached.</p>"
    elif stop_reason == "eos":
        stop_line = "<p class=\"metric-foot\">Stopped naturally at the EOS token.</p>"
    elif stop_reason == "fixed":
        stop_line = "<p class=\"metric-foot\">Fixed one-token classification benchmark.</p>"
    tile_html = "".join(
        f"""
        <div class="{tile_class}">
          <div class="metric-label">{html.escape(label)}</div>
          <div class="{value_class}">{html.escape(value)}</div>
        </div>
        """
        for label, value, tile_class, value_class in tiles
    )
    return f"""
    <section class="metric-shell">
      <p class="metrics-title {model_class}">{html.escape(state.spec.label)} metrics</p>
      <div class="metric-grid">{tile_html}</div>
      {load_line}
      {stop_line}
    </section>
    """


def comparison_rows(dense: RunState, sparse: RunState) -> List[Tuple[str, Any, Any, str, str]]:
    rows = [
        (
            f"{spec['label']} ({spec['suffix'].strip()})" if spec["suffix"].strip() else spec["label"],
            dense.metrics.get(spec["key"]),
            sparse.metrics.get(spec["key"]),
            spec["direction"],
        )
        for spec in METRIC_DISPLAY_SPECS
    ]
    normalized = []
    for metric, dense_value, sparse_value, direction in rows:
        if dense_value is None or sparse_value is None:
            winner = "Pending"
            winner_key = "pending"
        elif direction == "higher":
            outcome = compare_metric_values(dense_value, sparse_value, direction)
            if outcome == 1:
                winner = dense.spec.label
                winner_key = "dense"
            elif outcome == -1:
                winner = sparse.spec.label
                winner_key = "sparse"
            else:
                winner = "Tie"
                winner_key = "tie"
        elif direction == "lower":
            outcome = compare_metric_values(dense_value, sparse_value, direction)
            if outcome == 1:
                winner = dense.spec.label
                winner_key = "dense"
            elif outcome == -1:
                winner = sparse.spec.label
                winner_key = "sparse"
            else:
                winner = "Tie"
                winner_key = "tie"
        else:
            winner = "Matched" if dense_value == sparse_value else "Varies"
            winner_key = "matched" if dense_value == sparse_value else "varies"
        normalized.append((metric, dense_value, sparse_value, winner, winner_key))
    return normalized


def format_pruned_comparison(dense_value: Any, sparse_value: Any, direction: str, winner_key: str) -> str:
    if winner_key in ("tie", "matched"):
        return "No change"
    if winner_key in ("pending", "varies"):
        return "--"
    dense_number = finite_float(dense_value)
    sparse_number = finite_float(sparse_value)
    if dense_number is None or sparse_number is None or direction not in ("higher", "lower"):
        return "--"
    if direction == "higher":
        ratio = sparse_number / max(dense_number, 1e-9)
        percent_delta = (ratio - 1.0) * 100
        if math.isclose(percent_delta, 0.0, rel_tol=1e-9, abs_tol=1e-9):
            return "No change"
        sign = "+" if percent_delta > 0 else "-"
        return f"{sign}{abs(percent_delta):.0f}% ({ratio:.2f}x)"
    percent_delta = ((sparse_number - dense_number) / max(dense_number, 1e-9)) * 100
    if math.isclose(percent_delta, 0.0, rel_tol=1e-9, abs_tol=1e-9):
        return "No change"
    sign = "+" if percent_delta > 0 else "-"
    return f"{sign}{abs(percent_delta):.0f}%"


def comparison_winner_display(winner_key: str) -> str:
    return {
        "sparse": "Pruned model",
        "dense": "Original model",
        "tie": "Same value",
        "matched": "Matched",
        "varies": "Varies",
        "pending": "Pending",
    }.get(winner_key, "Pending")


def metric_definition(spec: Dict[str, Any]) -> str:
    description = str(spec.get("description", "")).strip()
    direction = spec.get("direction", "equal")
    if direction == "higher":
        return f"{description} Higher is better."
    if direction == "lower":
        return f"{description} Lower is better."
    return description


def build_comparison_df(dense: RunState, sparse: RunState) -> pd.DataFrame:
    rows = []
    spec_by_metric = {
        (f"{spec['label']} ({spec['suffix'].strip()})" if spec["suffix"].strip() else spec["label"]): spec
        for spec in METRIC_DISPLAY_SPECS
    }
    for metric, dense_value, sparse_value, winner, winner_key in comparison_rows(dense, sparse):
        spec = spec_by_metric.get(metric, {})
        direction = spec.get("direction", "equal")
        precision = int(spec.get("precision", 2))
        suffix = str(spec.get("suffix", ""))
        rows.append(
            {
                "Metric": metric,
                "Definition": metric_definition(spec),
                "Dense": format_number(dense_value, precision, suffix),
                "Sparse": format_number(sparse_value, precision, suffix),
                "Winner": comparison_winner_display(winner_key),
                "WinnerKey": winner_key,
                "WinnerLabel": winner,
                "Delta": format_pruned_comparison(dense_value, sparse_value, direction, winner_key),
            }
        )
    return pd.DataFrame(rows)


def render_scoreboard(dense: RunState, sparse: RunState, concurrent: bool, benchmark_mode: bool) -> str:
    dense_prefill = dense.metrics.get("prefill_tok_s")
    sparse_prefill = sparse.metrics.get("prefill_tok_s")
    dense_decode = dense.metrics.get("decode_tok_s")
    sparse_decode = sparse.metrics.get("decode_tok_s")
    dense_e2e = dense.metrics.get("e2e_tok_s")
    sparse_e2e = sparse.metrics.get("e2e_tok_s")
    dense_total = dense.metrics.get("total_latency_ms")
    sparse_total = sparse.metrics.get("total_latency_ms")

    prefill_speedup = (sparse_prefill / dense_prefill) if dense_prefill and sparse_prefill else None
    decode_speedup = (sparse_decode / dense_decode) if dense_decode and sparse_decode else None
    end_to_end_speedup = (sparse_e2e / dense_e2e) if dense_e2e and sparse_e2e else None
    if end_to_end_speedup is None and dense_total and sparse_total:
        end_to_end_speedup = dense_total / sparse_total

    same_device = dense.spec.device == sparse.spec.device
    if benchmark_mode:
        if same_device:
            note = "Benchmark is running sequentially because both models are configured on the same GPU."
        elif concurrent:
            note = "Benchmark is running concurrently on separate GPUs with a fixed output-token budget."
        else:
            note = "Benchmark uses separate GPUs with a fixed output-token budget."
    else:
        note = "Concurrent comparison on a shared GPU can distort absolute speedup numbers." if concurrent else "Live comparison is intended for qualitative inspection."
        if concurrent and same_device:
            note = "Both models are sharing the same GPU. Use Run Performance Benchmark for defendable numbers."

    def speedup_class(value: Optional[float]) -> str:
        if value is None:
            return "winner-neutral"
        if math.isclose(value, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            return "winner-neutral"
        return "winner-sparse" if value > 1.0 else "winner-dense"

    def speedup_tile_class(value: Optional[float]) -> str:
        value_class = speedup_class(value)
        if value_class == "winner-sparse":
            return "score-tile is-sparse-winner"
        if value_class == "winner-dense":
            return "score-tile is-dense-winner"
        return "score-tile"

    tiles = [
        (
            "Prefill speedup",
            format_number(prefill_speedup, 2, "x"),
            "Pruned prefill tok/s divided by original prefill tok/s.",
            speedup_class(prefill_speedup),
            speedup_tile_class(prefill_speedup),
        ),
        (
            "Decode speedup",
            format_number(decode_speedup, 2, "x"),
            "Pruned decode tok/s divided by original decode tok/s.",
            speedup_class(decode_speedup),
            speedup_tile_class(decode_speedup),
        ),
        (
            "End-to-end speedup",
            format_number(end_to_end_speedup, 2, "x"),
            note,
            speedup_class(end_to_end_speedup),
            speedup_tile_class(end_to_end_speedup),
        ),
    ]
    tile_html = "".join(
        f"""
        <div class="{tile_class}">
          <div class="score-label">{html.escape(label)}</div>
          <div class="score-value {css_class}">{html.escape(value)}</div>
          <div class="score-foot">{html.escape(copy)}</div>
        </div>
        """
        for label, value, copy, css_class, tile_class in tiles
    )
    return f"""
    <section class="score-shell">
      <p class="score-title">Performance Summary</p>
      <div class="score-grid">{tile_html}</div>
    </section>
    """


def render_log(messages: List[str]) -> str:
    entries = "".join(
        f"<p class=\"log-entry\"><strong>Run log</strong> {html.escape(message)}</p>"
        for message in messages[-8:]
    )
    return f"""
    <section class="log-shell">
      <p class="log-title">Run Status</p>
      {entries or '<p class="log-entry">No events yet.</p>'}
    </section>
    """


def render_quality_log(messages: List[str]) -> str:
    cleaned_messages = []
    for message in messages:
        if "completed " in message and " output tokens" in message:
            cleaned_messages.append(message.split(": completed ", 1)[0] + ": completed.")
        else:
            cleaned_messages.append(message)
    return render_log(cleaned_messages)


def render_notes(quality_cap: int, benchmark_cap: int) -> str:
    return f"""
    <section class="note-shell">
      <p class="note-title">Evaluation Method</p>
      <div class="protocol-grid">
        <div class="protocol-item">
          <p class="protocol-title">Response Quality</p>
          <p class="protocol-copy">Sends the same prompt to both models and shows their answers side by side. This mode is for reviewing output quality, with a {quality_cap}-token cap to keep responses within a reasonable length.</p>
        </div>
        <div class="protocol-item">
          <p class="protocol-title">Performance Benchmark</p>
          <p class="protocol-copy">Forces both models to run the same workload so speedup is measured fairly. Summarization generates exactly {benchmark_cap} output tokens, and financial sentiment evaluates one Good/Bad output token.</p>
        </div>
        <div class="protocol-item">
          <p class="protocol-title">Timing Measurements</p>
          <p class="protocol-copy">Records latency on the inference server where the models run. Browser rendering, UI updates, and network transfer time are excluded from the reported latency.</p>
        </div>
        <div class="protocol-item">
          <p class="protocol-title">Memory Measurements</p>
          <p class="protocol-copy">Reports loaded model footprint and peak VRAM separately. Footprint is the GPU memory used just to keep the model loaded, while peak VRAM is the highest GPU memory reached during inference.</p>
        </div>
      </div>
    </section>
    """


def render_comparison_table(df: pd.DataFrame) -> str:
    headers = [
        ("Metric", "Metric"),
        ("Definition", "Definition"),
        ("Dense", "Original"),
        ("Sparse", "Pruned"),
        ("Winner", "Better Model"),
        ("Delta", "Relative Difference"),
    ]
    if df.empty:
        rows_html = (
            f'<tr><td colspan="{len(headers)}" class="comparison-empty">Run a benchmark or live comparison to populate this table.</td></tr>'
        )
    else:
        row_chunks = []
        for _, row in df.iterrows():
            winner_key = row.get("WinnerKey", "pending")
            winner_key = winner_key if isinstance(winner_key, str) else "pending"
            row_class = f"comparison-row-{winner_key}" if winner_key in {"dense", "sparse"} else "comparison-row-neutral"
            dense_class = "comparison-value is-winner" if winner_key == "dense" else "comparison-value"
            sparse_class = "comparison-value is-winner" if winner_key == "sparse" else "comparison-value"
            badge_class = {
                "sparse": "comparison-badge is-sparse",
                "dense": "comparison-badge is-dense",
            }.get(winner_key, "comparison-badge is-neutral")
            change_class = "comparison-change is-sparse" if winner_key == "sparse" else "comparison-change"
            row_chunks.append(
                f"""
                <tr class="{row_class}">
                  <td class="comparison-metric">{html.escape(str(row.get("Metric", "--")))}</td>
                  <td class="comparison-definition">{html.escape(str(row.get("Definition", "--")))}</td>
                  <td class="{dense_class}">{html.escape(str(row.get("Dense", "--")))}</td>
                  <td class="{sparse_class}">{html.escape(str(row.get("Sparse", "--")))}</td>
                  <td><span class="{badge_class}">{html.escape(str(row.get("Winner", "--")))}</span></td>
                  <td class="{change_class}">{html.escape(str(row.get("Delta", "--")))}</td>
                </tr>
                """
            )
        rows_html = "".join(row_chunks)

    def header_class(key: str) -> str:
        if key == "Dense":
            return "model-dense"
        if key == "Sparse":
            return "model-sparse"
        return ""

    head_html = "".join(
        f"<th class=\"{header_class(key)}\">{html.escape(label)}</th>"
        for key, label in headers
    )
    return f"""
    <div class="comparison-scroll">
      <table class="comparison-grid">
        <thead>
          <tr>{head_html}</tr>
        </thead>
        <tbody>
          {rows_html}
        </tbody>
      </table>
    </div>
    """


def build_token_df(states: Iterable[RunState]) -> pd.DataFrame:
    rows = []
    for state in states:
        for point in state.timeline:
            rows.append(
                {
                    "elapsed_s": point["elapsed_s"],
                    "value": point["tokens"],
                    "model": state.spec.label,
                }
            )
    if not rows:
        return empty_timeseries_df()
    return pd.DataFrame(rows)


def build_throughput_df(states: Iterable[RunState]) -> pd.DataFrame:
    rows = []
    for state in states:
        for point in state.timeline:
            value = point.get("decode_tok_s")
            if value is None:
                continue
            rows.append(
                {
                    "elapsed_s": point["elapsed_s"],
                    "value": value,
                    "model": state.spec.label,
                }
            )
    if not rows:
        return empty_timeseries_df()
    return pd.DataFrame(rows)


def serialize_error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def classify_prompt_task(prompt: str) -> str:
    lower = prompt.lower()
    finance_terms = (
        "finance",
        "financial",
        "stock",
        "stocks",
        "market",
        "investor",
        "investors",
        "earnings",
        "shares",
        "revenue",
        "profit",
    )
    sentiment_terms = (
        "sentiment",
        "good or bad",
        "positive or negative",
        "bullish",
        "bearish",
    )
    if any(term in lower for term in finance_terms) and any(term in lower for term in sentiment_terms):
        return "finance_sentiment"
    if any(term in lower for term in ("summarize", "summarise", "summary", "summarization", "summarisation")):
        return "summarization"
    return "other"


def benchmark_task_key(label: str) -> str:
    normalized = (label or "").strip().lower()
    if "financial" in normalized or "finance" in normalized or "sentiment" in normalized:
        return "finance_sentiment"
    return "summarization"


def ensure_finance_answer_cue(prompt: str) -> str:
    cleaned = prompt.strip()
    if not cleaned:
        return cleaned
    lines = [line.rstrip() for line in cleaned.splitlines()]
    while lines and not lines[-1].strip():
        lines.pop()
    if lines and lines[-1].strip().lower() in {"answer:", "answer"}:
        lines[-1] = "Answer:"
        return "\n".join(lines)
    return cleaned + "\nAnswer:"


def extract_finance_text(prompt: str) -> str:
    cleaned = prompt.strip()
    lower = cleaned.lower()
    for marker in ("article:\n", "text:\n"):
        start_marker = lower.find(marker)
        if start_marker == -1:
            continue
        start = start_marker + len(marker)
        question_start = lower.find("\n\nquestion:", start)
        if question_start == -1:
            question_start = lower.find("\nquestion:", start)
        if question_start != -1:
            return cleaned[start:question_start].strip()
        return cleaned[start:].strip()
    return cleaned


def prompt_has_summarization_instruction(prompt: str) -> bool:
    lower = prompt.lower()
    return any(
        term in lower
        for term in ("summarize", "summarise", "summary", "summarization", "summarisation")
    )


def build_summarization_prompt(article_text: str) -> str:
    return (
        "Write a factual summary of this article. "
        "Cover the main points and important supporting details. "
        "Use only facts stated in the article. "
        "Do not add unsupported dates, numbers, names, or claims. "
        "Avoid repeating the same point.\n\n"
        f"Article:\n{article_text.strip()}"
    )


def prepare_prompt_for_task(prompt: str, task: str) -> str:
    if task == "finance_sentiment":
        return build_finance_sentiment_prompt(extract_finance_text(prompt))
    if task == "summarization" and not prompt_has_summarization_instruction(prompt):
        return build_summarization_prompt(prompt)
    return prompt


def build_finance_sentiment_prompt(finance_text: str) -> str:
    return (
        "Read the financial news article below.\n\n"
        f"Article:\n{finance_text}\n\n"
        "Classify the likely investor sentiment. Answer with exactly one word: Good or Bad."
    )


def extract_finance_sentiment_label(text: str) -> Optional[str]:
    cleaned = text.strip()
    for raw_token in cleaned.split():
        token = raw_token.strip(" \t\r\n:;,.!?\"'`()[]{}").lower()
        if token in {"good", "positive", "bullish", "favorable", "favourable"}:
            return "Good"
        if token in {"bad", "negative", "bearish", "unfavorable", "unfavourable"}:
            return "Bad"
    return None


def normalize_finance_sentiment_response(text: str) -> str:
    return extract_finance_sentiment_label(text) or text.strip()


def clean_benchmark_summary_text(text: str) -> str:
    cleaned = text.strip()
    for _ in range(3):
        cleaned = re.sub(r"\b([A-Za-z][A-Za-z0-9]*)\s+\1\b", r"\1", cleaned, flags=re.IGNORECASE)
    replacements = {
        "tobe": "to be",
        "desksduring": "desks during",
        "come form": "come from",
        "f rom": "from",
        "neeed": "need",
        "othe rse": "other",
        "internhips": "internships",
    }
    for bad, good in replacements.items():
        cleaned = re.sub(re.escape(bad), good, cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bQ{2,}[A-Za-z0-9-]*\b", "quarter", cleaned)
    cleaned = re.sub(r"\bqQ[A-Za-z0-9-]*\b", "quarter", cleaned)
    cleaned = re.sub(r"\bq/Q\b", "quarter", cleaned)
    cleaned = re.sub(r"\b(FY){2,}\b", "FY", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = cleaned.strip()
    lines = [
        line.strip(" \t-*\u2022")
        for line in cleaned.splitlines()
        if line.strip(" \t-*\u2022")
    ]
    if len(lines) < 3:
        lines = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", cleaned)
            if sentence.strip()
        ]
    def is_summary_preamble(line: str) -> bool:
        lower = line.strip(" \t-*\u2022:").lower()
        return (
            lower.startswith("here are")
            or lower.startswith("sure")
            or lower.startswith("summary:")
            or (
                "bullet point" in lower
                and any(term in lower for term in ("summar", "article"))
            )
        )

    lines = [line for line in lines if not is_summary_preamble(line)]
    if lines:
        points = []
        for line in lines[:3]:
            normalized = line.rstrip()
            if normalized and normalized[-1] not in ".!?":
                normalized += "."
            points.append(f"- {normalized}")
        return "\n".join(points)
    return cleaned


def normalize_token_id_list(value: Any) -> List[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, (list, tuple, set)):
        token_ids: List[int] = []
        for item in value:
            try:
                token_ids.append(int(item))
            except (TypeError, ValueError):
                continue
        return token_ids
    try:
        return [int(value)]
    except (TypeError, ValueError):
        return []


def resolve_eos_token_ids(bundle: LoadedBundle) -> List[int]:
    token_ids: List[int] = []
    generation_config = getattr(bundle.model, "generation_config", None)
    for source in (
        getattr(generation_config, "eos_token_id", None) if generation_config is not None else None,
        getattr(bundle.model.config, "eos_token_id", None),
        bundle.tokenizer.eos_token_id,
    ):
        token_ids.extend(normalize_token_id_list(source))
    return list(dict.fromkeys(token_ids))


def build_model_prompt(bundle: LoadedBundle, prompt: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    stripped = prompt.strip()
    chat_template = getattr(bundle.tokenizer, "chat_template", None)
    if chat_template:
        try:
            return bundle.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": stripped},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    special_tokens = ("<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>")
    unk_token_id = getattr(bundle.tokenizer, "unk_token_id", None)
    special_token_ids = [bundle.tokenizer.convert_tokens_to_ids(token) for token in special_tokens]
    if all(isinstance(token_id, int) and token_id >= 0 and token_id != unk_token_id for token_id in special_token_ids):
        return (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{stripped}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    return f"System: {system_prompt}\n\nQuestion: {stripped}\nAnswer:"


def encode_model_prompt(
    bundle: LoadedBundle,
    prompt: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> Dict[str, torch.Tensor]:
    model_prompt = build_model_prompt(bundle, prompt, system_prompt=system_prompt)
    return bundle.tokenizer(model_prompt, return_tensors="pt", padding=False, truncation=False)


@torch.inference_mode()
def score_finance_sentiment_label(
    bundle: LoadedBundle,
    prompt: str,
    use_chat_template: bool = True,
) -> Tuple[str, int, float, Dict[str, Any]]:
    device = base.get_model_input_device(bundle.model)
    prompt_encoded = (
        encode_model_prompt(bundle, prompt, system_prompt=FINANCE_SYSTEM_PROMPT)
        if use_chat_template
        else bundle.tokenizer(prompt.strip(), return_tensors="pt", padding=False, truncation=False)
    )
    prompt_tokens = int(prompt_encoded["input_ids"].shape[1])
    batch = inputs_on_device(prompt_encoded, device)
    model_dtype = get_model_runtime_dtype(bundle.model)

    autocast_ctx = contextlib.nullcontext()
    if device.type == "cuda" and model_dtype in {torch.float16, torch.bfloat16}:
        autocast_ctx = torch.autocast(device_type="cuda", dtype=model_dtype)

    sync_torch_device(device)
    prefill_start = time.perf_counter()
    with autocast_ctx:
        outputs = bundle.model(**batch, use_cache=True)
    sync_torch_device(device)
    prefill_latency_s = time.perf_counter() - prefill_start

    def score_label_candidates(
        raw_log_probs: torch.Tensor,
        prior_log_probs: torch.Tensor,
        candidates: Tuple[str, ...],
    ) -> Tuple[float, float, float, Optional[int], str]:
        best_calibrated = -math.inf
        best_raw = -math.inf
        best_prior = -math.inf
        best_token_id: Optional[int] = None
        best_candidate = candidates[0] if candidates else ""
        for candidate in candidates:
            label_ids = bundle.tokenizer(candidate, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
            if int(label_ids.shape[1]) != 1:
                continue
            token_id = int(label_ids[0, 0].item())
            raw_score = float(raw_log_probs[0, token_id].item())
            prior_score = float(prior_log_probs[0, token_id].item())
            calibrated_score = raw_score - prior_score
            if calibrated_score > best_calibrated:
                best_calibrated = calibrated_score
                best_raw = raw_score
                best_prior = prior_score
                best_token_id = token_id
                best_candidate = candidate
        return best_calibrated, best_raw, best_prior, best_token_id, best_candidate

    sync_torch_device(device)
    scoring_start = time.perf_counter()
    calibration_prompt = build_finance_sentiment_prompt(FINANCE_CALIBRATION_ARTICLE)
    calibration_encoded = (
        encode_model_prompt(bundle, calibration_prompt, system_prompt=FINANCE_SYSTEM_PROMPT)
        if use_chat_template
        else bundle.tokenizer(calibration_prompt, return_tensors="pt", padding=False, truncation=False)
    )
    calibration_batch = inputs_on_device(calibration_encoded, device)
    with autocast_ctx:
        calibration_outputs = bundle.model(**calibration_batch, use_cache=False)
    raw_log_probs = torch.log_softmax(outputs.logits[:, -1, :].float(), dim=-1)
    prior_log_probs = torch.log_softmax(calibration_outputs.logits[:, -1, :].float(), dim=-1)
    good_score, good_raw_score, good_prior_score, good_token_id, good_candidate = score_label_candidates(
        raw_log_probs,
        prior_log_probs,
        FINANCE_GOOD_LABEL_CANDIDATES,
    )
    bad_score, bad_raw_score, bad_prior_score, bad_token_id, bad_candidate = score_label_candidates(
        raw_log_probs,
        prior_log_probs,
        FINANCE_BAD_LABEL_CANDIDATES,
    )
    sync_torch_device(device)
    scoring_latency_s = time.perf_counter() - scoring_start
    label = "Good" if good_score >= bad_score else "Bad"
    decode_token_id = good_token_id if label == "Good" else bad_token_id

    decode_latency_s: Optional[float] = None
    if decode_token_id is not None and getattr(outputs, "past_key_values", None) is not None:
        try:
            decode_input_ids = torch.tensor([[decode_token_id]], device=device, dtype=batch["input_ids"].dtype)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                decode_attention_mask = torch.ones(
                    (attention_mask.shape[0], attention_mask.shape[1] + 1),
                    device=device,
                    dtype=attention_mask.dtype,
                )
            else:
                decode_attention_mask = None
            decode_kwargs: Dict[str, Any] = {
                "input_ids": decode_input_ids,
                "past_key_values": outputs.past_key_values,
                "use_cache": True,
            }
            if decode_attention_mask is not None:
                decode_kwargs["attention_mask"] = decode_attention_mask
            sync_torch_device(device)
            decode_start = time.perf_counter()
            with autocast_ctx:
                bundle.model(**decode_kwargs)
            sync_torch_device(device)
            decode_latency_s = time.perf_counter() - decode_start
        except Exception:
            decode_latency_s = None

    total_latency_s = prefill_latency_s + scoring_latency_s + (decode_latency_s or 0.0)
    phase_metrics = {
        "prefill_latency_ms": prefill_latency_s * 1000.0,
        "prefill_tok_s": prompt_tokens / max(prefill_latency_s, 1e-9),
        "decode_ms_per_token": (decode_latency_s * 1000.0) if decode_latency_s is not None else None,
        "decode_tok_s": (1.0 / max(decode_latency_s, 1e-9)) if decode_latency_s is not None else None,
        "phase_total_latency_ms": total_latency_s * 1000.0,
        "label_scoring_latency_ms": scoring_latency_s * 1000.0,
        "good_logprob": good_raw_score,
        "bad_logprob": bad_raw_score,
        "good_prior_logprob": good_prior_score,
        "bad_prior_logprob": bad_prior_score,
        "good_calibrated_logprob": good_score,
        "bad_calibrated_logprob": bad_score,
        "sentiment_margin_logprob": abs(good_score - bad_score),
        "good_scoring_token": good_candidate,
        "bad_scoring_token": bad_candidate,
        "sentiment_scoring": "calibrated_next_token",
    }
    return label, prompt_tokens, total_latency_s, phase_metrics


def generate_kwargs_for_prompt(
    prompt: str,
    bundle: LoadedBundle,
    max_new_tokens: int,
    min_new_tokens: Optional[int],
    ignore_eos: bool,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    seed: int,
) -> Tuple[Dict[str, Any], int]:
    encoded = encode_model_prompt(bundle, prompt)
    prompt_tokens = int(encoded["input_ids"].shape[1])
    device = base.get_model_input_device(bundle.model)
    batch = inputs_on_device(encoded, device)
    do_sample = temperature > 0.0
    if do_sample:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    kwargs: Dict[str, Any] = {
        **batch,
        "max_new_tokens": max_new_tokens,
        "min_new_tokens": min_new_tokens,
        "pad_token_id": bundle.tokenizer.pad_token_id,
        "use_cache": True,
        "do_sample": do_sample,
        "temperature": temperature if do_sample else None,
        "top_p": top_p if do_sample else None,
        "repetition_penalty": repetition_penalty,
        "no_repeat_ngram_size": no_repeat_ngram_size if no_repeat_ngram_size > 0 else None,
    }
    if not ignore_eos:
        eos_token_ids = resolve_eos_token_ids(bundle)
        if eos_token_ids:
            kwargs["eos_token_id"] = eos_token_ids[0] if len(eos_token_ids) == 1 else eos_token_ids
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    return kwargs, prompt_tokens


def finalize_metrics(
    bundle: LoadedBundle,
    streamer: MetricsStreamer,
    prompt_tokens: int,
    output_tokens: int,
    total_latency_s: float,
) -> Dict[str, Any]:
    snapshot = streamer.snapshot()
    ttft_ms = snapshot.get("ttft_ms")
    total_latency_ms = total_latency_s * 1000.0
    decode_tok_s = None
    avg_itl_ms = None
    if snapshot.get("generated_tokens", 0) > 1 and streamer.first_token_at is not None:
        decode_elapsed = max(time.perf_counter() - streamer.first_token_at, 1e-9)
        decode_tok_s = (output_tokens - 1) / decode_elapsed if output_tokens > 1 else None
        avg_itl_ms = (decode_elapsed / max(output_tokens - 1, 1)) * 1000.0 if output_tokens > 1 else None

    return {
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "ttft_ms": ttft_ms,
        "prefill_latency_ms": ttft_ms,
        "prefill_tok_s": (prompt_tokens / max(ttft_ms / 1000.0, 1e-9)) if ttft_ms else None,
        "decode_tok_s": decode_tok_s,
        "avg_itl_ms": avg_itl_ms,
        "decode_ms_per_token": avg_itl_ms,
        "total_latency_ms": total_latency_ms,
        "e2e_tok_s": output_tokens / max(total_latency_s, 1e-9) if output_tokens > 0 else 0.0,
        "peak_vram_gb": safe_peak_vram_gb(bundle.spec.device),
    }


def finalize_fixed_token_metrics(
    bundle: LoadedBundle,
    prompt_tokens: int,
    output_tokens: int,
    total_latency_s: float,
    phase_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    total_latency_ms = total_latency_s * 1000.0
    tokens_per_second = output_tokens / max(total_latency_s, 1e-9) if output_tokens > 0 else 0.0
    phase_metrics = phase_metrics or {}
    prefill_latency_ms = phase_metrics.get("prefill_latency_ms", total_latency_ms)
    prefill_tok_s = phase_metrics.get("prefill_tok_s", prompt_tokens / max(total_latency_s, 1e-9))
    decode_tok_s = phase_metrics.get("decode_tok_s")
    decode_ms_per_token = phase_metrics.get("decode_ms_per_token")
    return {
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "ttft_ms": prefill_latency_ms,
        "prefill_latency_ms": prefill_latency_ms,
        "prefill_tok_s": prefill_tok_s,
        "decode_tok_s": decode_tok_s,
        "avg_itl_ms": decode_ms_per_token,
        "decode_ms_per_token": decode_ms_per_token,
        "total_latency_ms": total_latency_ms,
        "e2e_tok_s": tokens_per_second,
        "peak_vram_gb": safe_peak_vram_gb(bundle.spec.device),
    }


def worker_run(
    manager: ModelManager,
    spec: ModelSpec,
    prompt: str,
    task: str,
    max_new_tokens: int,
    min_new_tokens: Optional[int],
    ignore_eos: bool,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    seed: int,
    event_queue: "queue.Queue[Dict[str, Any]]",
    bundle: Optional[LoadedBundle] = None,
) -> None:
    try:
        if bundle is None:
            event_queue.put({"type": "status", "key": spec.key, "message": f"{spec.label}: loading model."})
            bundle = manager.get_or_load(spec)
            event_queue.put(
                {
                    "type": "loaded",
                    "key": spec.key,
                    "message": f"{spec.label}: model ready on {spec.device}.",
                    "load_info": bundle.load_info,
                }
            )

        if task == "finance_sentiment":
            event_queue.put({"type": "status", "key": spec.key, "message": f"{spec.label}: scoring sentiment."})
            reset_peak_vram(spec.device)
            final_text, prompt_tokens, total_latency_s, phase_metrics = score_finance_sentiment_label(bundle, prompt)
            output_tokens = FINANCE_SENTIMENT_MAX_NEW_TOKENS
            final_metrics = finalize_fixed_token_metrics(
                bundle,
                prompt_tokens,
                output_tokens,
                total_latency_s,
                phase_metrics=phase_metrics,
            )
            final_metrics["stop_reason"] = "fixed"
            timeline = [
                {
                    "elapsed_s": total_latency_s,
                    "tokens": output_tokens,
                    "decode_tok_s": final_metrics["decode_tok_s"],
                }
            ]
            event_queue.put(
                {
                    "type": "done",
                    "key": spec.key,
                    "message": f"{spec.label}: completed {output_tokens} output token.",
                    "response_text": final_text,
                    "metrics": final_metrics,
                    "timeline": timeline,
                    "load_info": bundle.load_info,
                    "stop_reason": "fixed",
                }
            )
            return

        kwargs, prompt_tokens = generate_kwargs_for_prompt(
            prompt=prompt,
            bundle=bundle,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            ignore_eos=ignore_eos,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            seed=seed,
        )
        start_time = time.perf_counter()
        reset_peak_vram(spec.device)
        streamer = MetricsStreamer(bundle.tokenizer, start_time=start_time, timeout=0.2)
        kwargs["streamer"] = streamer
        result_holder: Dict[str, Any] = {}

        def _generate() -> None:
            generation_config = getattr(bundle.model, "generation_config", None)
            old_generation_eos = getattr(generation_config, "eos_token_id", None) if generation_config else None
            old_model_eos = getattr(bundle.model.config, "eos_token_id", None)
            try:
                with torch.inference_mode():
                    model_device = base.get_model_input_device(bundle.model)
                    model_dtype = get_model_runtime_dtype(bundle.model)
                    autocast_ctx = contextlib.nullcontext()
                    if model_device.type == "cuda" and model_dtype in {torch.float16, torch.bfloat16}:
                        autocast_ctx = torch.autocast(device_type="cuda", dtype=model_dtype)
                    with autocast_ctx:
                        if ignore_eos:
                            if generation_config is not None:
                                generation_config.eos_token_id = None
                            bundle.model.config.eos_token_id = None
                        result_holder["result"] = bundle.model.generate(**kwargs)
            except Exception as exc:  # pragma: no cover - surfaced to UI
                result_holder["error"] = exc
            finally:
                if generation_config is not None:
                    generation_config.eos_token_id = old_generation_eos
                bundle.model.config.eos_token_id = old_model_eos

        event_queue.put({"type": "status", "key": spec.key, "message": f"{spec.label}: generating tokens."})
        thread = threading.Thread(target=_generate, daemon=True)
        thread.start()

        response_text = ""
        while thread.is_alive():
            try:
                piece = next(streamer)
                response_text += piece
                snapshot = streamer.snapshot()
                snapshot["prefill_tok_s"] = (
                    prompt_tokens / max(snapshot["ttft_ms"] / 1000.0, 1e-9)
                    if snapshot.get("ttft_ms")
                    else None
                )
                event_queue.put(
                    {
                        "type": "chunk",
                        "key": spec.key,
                        "response_text": (
                            ""
                            if task == "finance_sentiment"
                            else response_text
                        ),
                        "metrics": snapshot,
                        "timeline": snapshot["timeline"],
                    }
                )
            except queue.Empty:
                continue
            except StopIteration:
                break

        thread.join()
        if "error" in result_holder:
            raise result_holder["error"]

        result_tokens = result_holder["result"]
        while True:
            try:
                piece = next(streamer)
                response_text += piece
            except queue.Empty:
                break
            except StopIteration:
                break

        output_tokens = max(int(result_tokens.shape[1]) - prompt_tokens, 0)
        eos_token_ids = resolve_eos_token_ids(bundle)
        last_token_id = int(result_tokens[0, -1].item()) if result_tokens.shape[1] > 0 else None
        stop_reason = "eos" if last_token_id in set(eos_token_ids or []) else "length"
        final_text = bundle.tokenizer.decode(
            result_tokens[0, prompt_tokens:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()
        final_text = final_text or response_text.strip()
        total_latency_s = time.perf_counter() - start_time
        final_metrics = finalize_metrics(bundle, streamer, prompt_tokens, output_tokens, total_latency_s)
        final_metrics["stop_reason"] = stop_reason
        event_queue.put(
            {
                "type": "done",
                "key": spec.key,
                "message": f"{spec.label}: completed {output_tokens} output tokens.",
                "response_text": final_text,
                "metrics": final_metrics,
                "timeline": list(streamer.timeline),
                "load_info": bundle.load_info,
                "stop_reason": stop_reason,
            }
        )
    except Exception as exc:  # pragma: no cover - surfaced to UI
        event_queue.put(
            {
                "type": "error",
                "key": spec.key,
                "message": f"{spec.label}: {serialize_error(exc)}",
                "error_text": serialize_error(exc),
            }
        )


def initial_states(specs: List[ModelSpec]) -> Dict[str, RunState]:
    return {spec.key: RunState(spec=spec) for spec in specs}


def apply_event(states: Dict[str, RunState], event: Dict[str, Any]) -> List[str]:
    state = states[event["key"]]
    messages: List[str] = []
    event_type = event["type"]
    if event_type == "status":
        state.status = event["message"]
        messages.append(event["message"])
    elif event_type == "loaded":
        state.status = event["message"]
        state.load_info = event.get("load_info", {})
        messages.append(event["message"])
    elif event_type == "chunk":
        state.status = "Streaming"
        state.response_text = event.get("response_text", "")
        state.timeline = event.get("timeline", [])
        state.metrics.update(event.get("metrics", {}))
    elif event_type == "done":
        state.status = "Complete"
        state.completed = True
        state.response_text = event.get("response_text", "")
        state.timeline = event.get("timeline", [])
        state.metrics.update(event.get("metrics", {}))
        state.load_info = event.get("load_info", state.load_info)
        stop_reason = state.metrics.get("stop_reason")
        if stop_reason == "length":
            messages.append(f"{event['message']} Hit the output-token cap before EOS.")
        else:
            messages.append(event["message"])
    elif event_type == "error":
        state.status = "Error"
        state.errored = True
        state.error_text = event.get("error_text")
        state.response_text = event.get("error_text", "")
        messages.append(event["message"])
    return messages


def render_outputs(states: Dict[str, RunState], messages: List[str], concurrent: bool, benchmark_mode: bool):
    dense = states["dense"]
    sparse = states["sparse"]
    token_df = build_token_df(states.values())
    throughput_df = build_throughput_df(states.values())
    comparison_df = build_comparison_df(dense, sparse) if (dense.completed or dense.errored) and (sparse.completed or sparse.errored) else empty_metrics_df()
    comparison_html = render_comparison_table(comparison_df)
    if not benchmark_mode:
        return (
            gr.update(value=render_quality_log(messages), visible=True),
            gr.update(value="", visible=False),
            gr.update(value="", visible=False),
            dense.response_text,
            sparse.response_text,
            gr.update(value=render_model_overview(dense), visible=True),
            gr.update(value=render_model_overview(sparse), visible=True),
            gr.update(value="", visible=False),
            gr.update(value="", visible=False),
            gr.update(value=empty_timeseries_df(), visible=False),
            gr.update(value=empty_timeseries_df(), visible=False),
            gr.update(value="", visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        )
    return (
        gr.update(value="", visible=False),
        gr.update(value=render_log(messages), visible=True),
        gr.update(value=render_scoreboard(dense, sparse, concurrent=concurrent, benchmark_mode=benchmark_mode), visible=True),
        dense.response_text,
        sparse.response_text,
        gr.update(value=render_model_overview(dense), visible=True),
        gr.update(value=render_model_overview(sparse), visible=True),
        gr.update(value=render_metrics_card(dense, sparse), visible=True),
        gr.update(value=render_metrics_card(sparse, dense), visible=True),
        gr.update(value=empty_timeseries_df(), visible=False),
        gr.update(value=empty_timeseries_df(), visible=False),
        gr.update(value=comparison_html, visible=True),
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=True),
    )


def drive_run(
    manager: ModelManager,
    specs: List[ModelSpec],
    prompt: str,
    task: str,
    max_new_tokens: int,
    min_new_tokens: Optional[int],
    ignore_eos: bool,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    seed: int,
    concurrent: bool,
    benchmark_mode: bool,
):
    states = initial_states(specs)
    event_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
    messages = [f"Prompt accepted with {len(prompt)} characters."]
    yield render_outputs(states, messages, concurrent=concurrent, benchmark_mode=benchmark_mode)

    threads: List[threading.Thread] = []
    preloaded_bundles: Dict[str, LoadedBundle] = {}

    def start_worker(spec: ModelSpec, bundle: Optional[LoadedBundle] = None) -> threading.Thread:
        thread = threading.Thread(
            target=worker_run,
            kwargs={
                "manager": manager,
                "spec": spec,
                "prompt": prompt,
                "task": task,
                "max_new_tokens": max_new_tokens,
                "min_new_tokens": min_new_tokens,
                "ignore_eos": ignore_eos,
                "temperature": temperature,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
                "no_repeat_ngram_size": no_repeat_ngram_size,
                "seed": seed,
                "event_queue": event_queue,
                "bundle": bundle,
            },
            daemon=True,
        )
        thread.start()
        return thread

    def preload_worker(spec: ModelSpec) -> threading.Thread:
        def _target() -> None:
            try:
                bundle = manager.get_or_load(spec)
                event_queue.put(
                    {
                        "type": "loaded",
                        "key": spec.key,
                        "message": f"{spec.label}: ready on {spec.device}.",
                        "load_info": bundle.load_info,
                        "bundle": bundle,
                    }
                )
            except Exception as exc:
                event_queue.put(
                    {
                        "type": "error",
                        "key": spec.key,
                        "message": f"{spec.label}: {serialize_error(exc)}",
                        "error_text": serialize_error(exc),
                    }
                )

        thread = threading.Thread(target=_target, daemon=True)
        thread.start()
        return thread

    if concurrent:
        preload_threads: List[threading.Thread] = []
        for spec in specs:
            states[spec.key].status = "Preloading"
            messages.append(f"{spec.label}: preloading model.")
            preload_threads.append(preload_worker(spec))
        yield render_outputs(states, messages, concurrent=concurrent, benchmark_mode=benchmark_mode)

        while any(thread.is_alive() for thread in preload_threads) or not event_queue.empty():
            try:
                event = event_queue.get(timeout=0.2)
                if event["type"] == "loaded":
                    bundle = event.pop("bundle")
                    preloaded_bundles[event["key"]] = bundle
                messages.extend(apply_event(states, event))
                yield render_outputs(states, messages, concurrent=concurrent, benchmark_mode=benchmark_mode)
            except queue.Empty:
                pass

        if any(state.errored for state in states.values()):
            yield render_outputs(states, messages, concurrent=concurrent, benchmark_mode=benchmark_mode)
            return

        for spec in specs:
            threads.append(start_worker(spec, bundle=preloaded_bundles.get(spec.key)))
    else:
        benchmark_devices = [parse_device(spec.device)[0] for spec in specs]
        can_preload_all = len(set(benchmark_devices)) == len(benchmark_devices)
        if can_preload_all:
            preload_threads = []
            for spec in specs:
                states[spec.key].status = "Preloading"
                messages.append(f"{spec.label}: preloading model for benchmark.")
                preload_threads.append(preload_worker(spec))
            yield render_outputs(states, messages, concurrent=concurrent, benchmark_mode=benchmark_mode)

            while any(thread.is_alive() for thread in preload_threads) or not event_queue.empty():
                try:
                    event = event_queue.get(timeout=0.2)
                    if event["type"] == "loaded":
                        bundle = event.pop("bundle")
                        preloaded_bundles[event["key"]] = bundle
                    messages.extend(apply_event(states, event))
                    yield render_outputs(states, messages, concurrent=concurrent, benchmark_mode=benchmark_mode)
                except queue.Empty:
                    pass

            if any(state.errored for state in states.values()):
                yield render_outputs(states, messages, concurrent=concurrent, benchmark_mode=benchmark_mode)
                return

        seen_devices: set[str] = set()
        for spec in specs:
            normalized_device, _ = parse_device(spec.device)
            if normalized_device in seen_devices:
                manager.clear()
                messages.append(f"Shared-device benchmark: cleared cached models before loading {spec.label}.")
                yield render_outputs(states, messages, concurrent=concurrent, benchmark_mode=benchmark_mode)
            seen_devices.add(normalized_device)
            thread = start_worker(spec, bundle=preloaded_bundles.get(spec.key))
            threads = [thread]
            while thread.is_alive() or not event_queue.empty():
                try:
                    event = event_queue.get(timeout=0.2)
                    messages.extend(apply_event(states, event))
                    yield render_outputs(states, messages, concurrent=concurrent, benchmark_mode=benchmark_mode)
                except queue.Empty:
                    pass
        threads = []

    while any(thread.is_alive() for thread in threads) or not event_queue.empty():
        try:
            event = event_queue.get(timeout=0.2)
            messages.extend(apply_event(states, event))
            yield render_outputs(states, messages, concurrent=concurrent, benchmark_mode=benchmark_mode)
        except queue.Empty:
            pass

    yield render_outputs(states, messages, concurrent=concurrent, benchmark_mode=benchmark_mode)


def sanitize_prompt(prompt: str, max_input_chars: int) -> str:
    cleaned = (prompt or "").strip()
    if not cleaned:
        raise gr.Error("Enter a prompt before running the comparison.")
    if len(cleaned) > max_input_chars:
        raise gr.Error(f"Prompt is too long. Limit it to {max_input_chars} characters.")
    return cleaned


def warm_start_models(
    dense_model: str,
    sparse_model: str,
    dense_label: str,
    sparse_label: str,
    dense_device: str,
    sparse_device: str,
    dense_dtype_name: str,
    sparse_dtype_name: str,
    manager: ModelManager,
) -> str:
    specs = build_specs(
        dense_model,
        sparse_model,
        dense_label,
        sparse_label,
        dense_device,
        sparse_device,
        dense_dtype_name,
        sparse_dtype_name,
    )
    messages = []
    for spec in specs:
        bundle = manager.get_or_load(spec)
        messages.append(
            f"{spec.label}: ready on {spec.device} in {bundle.load_info['model_load_time_s']:.2f} s."
        )
    return render_log(messages)


def unload_models(manager: ModelManager) -> str:
    manager.clear()
    return render_log(["Model cache cleared and CUDA cache released."])


def build_demo(args: argparse.Namespace) -> gr.Blocks:
    manager = ModelManager(args.hf_cache_dir, tokenizer_model=args.tokenizer_model or args.dense_model)
    dense_dtype_name = args.dense_dtype or args.dtype or "auto"
    sparse_dtype_name = resolve_sparse_dtype_name(args)
    color_map = {
        args.dense_label: "#7cc6ff",
        args.sparse_label: "#8bf0bf",
    }
    default_specs = build_specs(
        args.dense_model,
        args.sparse_model,
        args.dense_label,
        args.sparse_label,
        args.dense_device,
        args.sparse_device,
        dense_dtype_name,
        sparse_dtype_name,
    )
    default_dense_state = RunState(spec=default_specs[0])
    default_sparse_state = RunState(spec=default_specs[1])

    summarization_benchmark_tokens = min(
        DEFAULT_BENCHMARK_NEW_TOKENS,
        BENCHMARK_MAX_NEW_TOKENS,
        args.max_new_tokens_cap,
    )

    def run_mode(
        prompt_text: str,
        *,
        response_quality: bool,
        selected_benchmark_task: str = "Financial Sentiment Analysis",
    ):
        prompt_clean = sanitize_prompt(prompt_text, args.max_input_chars)
        if response_quality:
            detected_task = classify_prompt_task(prompt_clean)
            task = "finance_sentiment" if detected_task == "finance_sentiment" else "response_quality"
        else:
            task = benchmark_task_key(selected_benchmark_task)
        prompt_for_model = prepare_prompt_for_task(prompt_clean, task)
        benchmark_mode = not response_quality
        if response_quality and task == "finance_sentiment":
            concurrent = len({parse_device(spec.device)[0] for spec in default_specs}) == len(default_specs)
            max_new_tokens = min(FINANCE_SENTIMENT_MAX_NEW_TOKENS, args.max_new_tokens_cap)
            min_new_tokens = None
            ignore_eos = False
            temperature = DEFAULT_TEMPERATURE
            top_p = DEFAULT_TOP_P
            repetition_penalty = DEFAULT_REPETITION_PENALTY
            no_repeat_ngram_size = DEFAULT_NO_REPEAT_NGRAM_SIZE
        elif response_quality:
            response_devices = [parse_device(spec.device)[0] for spec in default_specs]
            concurrent = len(set(response_devices)) == len(response_devices)
            max_new_tokens = max(
                QUALITY_MIN_CAP_NEW_TOKENS,
                min(QUALITY_MAX_NEW_TOKENS, args.max_new_tokens_cap),
            )
            min_new_tokens = min(QUALITY_MIN_OUTPUT_TOKENS, max_new_tokens)
            ignore_eos = False
            temperature = QUALITY_TEMPERATURE
            top_p = QUALITY_TOP_P
            repetition_penalty = DEFAULT_REPETITION_PENALTY
            no_repeat_ngram_size = DEFAULT_NO_REPEAT_NGRAM_SIZE
        else:
            benchmark_devices = [parse_device(spec.device)[0] for spec in default_specs]
            concurrent = len(set(benchmark_devices)) == len(benchmark_devices)
            ignore_eos = False
            temperature = DEFAULT_TEMPERATURE
            top_p = DEFAULT_TOP_P
            repetition_penalty = DEFAULT_REPETITION_PENALTY
            no_repeat_ngram_size = DEFAULT_NO_REPEAT_NGRAM_SIZE
            if task == "finance_sentiment":
                max_new_tokens = min(FINANCE_SENTIMENT_MAX_NEW_TOKENS, args.max_new_tokens_cap)
                min_new_tokens = None
            else:
                max_new_tokens = summarization_benchmark_tokens
                min_new_tokens = summarization_benchmark_tokens
        yield from drive_run(
            manager=manager,
            specs=default_specs,
            prompt=prompt_for_model,
            task=task,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            ignore_eos=ignore_eos,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            seed=DEFAULT_SEED,
            concurrent=concurrent,
            benchmark_mode=benchmark_mode,
        )

    def run_live_comparison(prompt_text: str):
        yield from run_mode(prompt_text, response_quality=True)

    def run_benchmark(prompt_text: str, selected_benchmark_task: str):
        yield from run_mode(
            prompt_text,
            response_quality=False,
            selected_benchmark_task=selected_benchmark_task,
        )

    def update_prompt_counter(prompt_text: str):
        return render_prompt_counter(prompt_text, args.max_input_chars)

    with gr.Blocks(title=APP_TITLE, theme=APP_THEME, css=CSS, head=HEAD, fill_width=True) as demo:
        gr.HTML(render_hero())

        with gr.Group(elem_classes="control-shell"):
            gr.HTML(
                """
                <div class="section-head">
                  <p class="section-kicker">Evaluation Input</p>
                  <h2>Compare the baseline and compressed model</h2>
                  <p>Use the same input to assess response quality or measure latency, throughput, and memory in benchmark mode.</p>
                </div>
                """
            )
            prompt = gr.Textbox(
                show_label=False,
                placeholder="Enter the question or instruction to send to both models.",
                lines=7,
                elem_id="prompt-box",
                value=DEFAULT_PROMPT,
            )
            prompt_counter = gr.HTML(render_prompt_counter(DEFAULT_PROMPT, args.max_input_chars))

            with gr.Row(elem_classes=["action-row", "dual-action-row"]):
                with gr.Column(scale=1, min_width=280, elem_classes="action-panel"):
                    comparison_btn = gr.Button("Assess Response Quality", variant="primary")
                with gr.Column(scale=1, min_width=280, elem_classes="action-panel"):
                    benchmark_task = gr.Radio(
                        choices=["Financial Sentiment Analysis", "Summarization"],
                        value="Financial Sentiment Analysis",
                        label="Benchmark Task",
                        elem_id="benchmark-task-radio",
                    )
                    benchmark_btn = gr.Button("Run Performance Benchmark", variant="secondary")

            prompt.input(
                fn=update_prompt_counter,
                inputs=prompt,
                outputs=prompt_counter,
            )
            prompt.change(
                fn=update_prompt_counter,
                inputs=prompt,
                outputs=prompt_counter,
            )

        gr.HTML(
            render_notes(
                max(QUALITY_MIN_CAP_NEW_TOKENS, min(QUALITY_MAX_NEW_TOKENS, args.max_new_tokens_cap)),
                summarization_benchmark_tokens,
            )
        )

        quality_run_log = gr.HTML(render_log([]), visible=False)

        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=320):
                run_score = gr.HTML(render_scoreboard(default_dense_state, default_sparse_state, concurrent=True, benchmark_mode=False))
            with gr.Column(scale=1, min_width=320):
                run_log = gr.HTML(render_log([]))

        with gr.Row(equal_height=False):
            with gr.Column(elem_classes="model-shell"):
                dense_panel = gr.HTML(render_model_overview(default_dense_state))
                dense_box = gr.Textbox(show_label=False, lines=18, elem_id="dense-box")
            with gr.Column(elem_classes="model-shell"):
                sparse_panel = gr.HTML(render_model_overview(default_sparse_state))
                sparse_box = gr.Textbox(show_label=False, lines=18, elem_id="sparse-box")

        with gr.Row(equal_height=False) as metrics_row:
            with gr.Column():
                dense_metrics = gr.HTML(render_metrics_card(default_dense_state, default_sparse_state))
            with gr.Column():
                sparse_metrics = gr.HTML(render_metrics_card(default_sparse_state, default_dense_state))

        with gr.Row(equal_height=False, visible=False) as viz_row:
            with gr.Column(elem_classes="viz-shell"):
                gr.HTML(
                    """
                    <div class="section-head compact">
                      <p class="section-kicker">Token Growth</p>
                    </div>
                    """
                )
                tokens_plot = gr.LinePlot(
                    value=empty_timeseries_df(),
                    x="elapsed_s",
                    y="value",
                    color="model",
                    title="Cumulative output tokens",
                    x_title="Elapsed seconds",
                    y_title="Generated tokens",
                    color_map=color_map,
                    height=320,
                )
            with gr.Column(elem_classes="viz-shell"):
                gr.HTML(
                    """
                    <div class="section-head compact">
                      <p class="section-kicker">Decode Throughput</p>
                    </div>
                    """
                )
                throughput_plot = gr.LinePlot(
                    value=empty_timeseries_df(),
                    x="elapsed_s",
                    y="value",
                    color="model",
                    title="Decode throughput",
                    x_title="Elapsed seconds",
                    y_title="Tokens / second",
                    color_map=color_map,
                    height=320,
                )

        with gr.Group(elem_classes="table-shell") as comparison_table_group:
            gr.HTML(
                """
                <div class="section-head compact">
                  <p class="section-kicker">Metric Comparison</p>
                </div>
                """
            )
            comparison_df = gr.HTML(render_comparison_table(empty_metrics_df()))

        comparison_btn.click(
            fn=run_live_comparison,
            inputs=prompt,
            outputs=[
                quality_run_log,
                run_log,
                run_score,
                dense_box,
                sparse_box,
                dense_panel,
                sparse_panel,
                dense_metrics,
                sparse_metrics,
                tokens_plot,
                throughput_plot,
                comparison_df,
                metrics_row,
                viz_row,
                comparison_table_group,
            ],
        )

        benchmark_btn.click(
            fn=run_benchmark,
            inputs=[prompt, benchmark_task],
            outputs=[
                quality_run_log,
                run_log,
                run_score,
                dense_box,
                sparse_box,
                dense_panel,
                sparse_panel,
                dense_metrics,
                sparse_metrics,
                tokens_plot,
                throughput_plot,
                comparison_df,
                metrics_row,
                viz_row,
                comparison_table_group,
            ],
        )

        demo.queue(default_concurrency_limit=1, max_size=8)

    return demo


def register_pwa_routes(app: FastAPI) -> None:
    if getattr(app.state, "capstone_pwa_registered", False):
        return
    app.state.capstone_pwa_registered = True
    app.mount("/pwa", StaticFiles(directory=PWA_DIR), name="pwa")

    @app.get("/manifest.webmanifest", include_in_schema=False)
    async def manifest() -> FileResponse:
        return FileResponse(PWA_DIR / "manifest.webmanifest", media_type="application/manifest+json")

    @app.get("/service-worker.js", include_in_schema=False)
    async def service_worker() -> FileResponse:
        return FileResponse(PWA_DIR / "service-worker.js", media_type="application/javascript")


def main() -> None:
    args = parse_args()
    if args.reload:
        print("Warning: --reload is not supported in direct demo.launch mode. Starting without autoreload.")

    base.set_seed(42)
    base.configure_torch()
    demo = build_demo(args)
    server_app, local_url, share_url = demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        favicon_path=str(PWA_DIR / "icon.svg"),
        allowed_paths=[str(PWA_DIR)],
        show_error=True,
        prevent_thread_lock=True,
    )
    register_pwa_routes(server_app)

    if local_url:
        print(f"Local URL: {local_url}")
    if share_url:
        print(f"Share URL: {share_url}")

    demo.block_thread()


if __name__ == "__main__":
    main()
