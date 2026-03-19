"""Interactive Plotly visualization of dimensionality-reduced latent spaces."""

from __future__ import annotations

import base64
import json
import tempfile
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any, Literal

import numpy as np
import plotly.graph_objects as go
from PIL import Image


def _encode_image_for_hover(
    image: np.ndarray,
    *,
    size: int = 64,
) -> str:
    """Encode a single image as a base64 data URI for Plotly hover."""
    image = np.asarray(image)
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = np.transpose(image, (1, 2, 0))
        if image.shape[-1] == 1:
            image = image.squeeze(-1)
    if image.ndim == 2:
        img = Image.fromarray(image, mode="L")
    else:
        img = Image.fromarray(image, mode="RGB")
    img = img.resize((size, size), Image.Resampling.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _reduce_dimensions(
    embeddings: np.ndarray,
    reducer: Literal["umap", "tsne"],
    n_components: int = 2,
    **reducer_kwargs,
) -> tuple[np.ndarray, Any]:
    """Reduce high-dimensional embeddings to 2D. Returns (coords, mapper) for UMAP or (coords, None) for t-SNE."""
    if reducer == "umap":
        import umap

        default_kwargs = dict(
            n_neighbors=15,
            min_dist=0.1,
            n_components=n_components,
            metric="cosine",
            random_state=0,
        )
        default_kwargs.update(reducer_kwargs)
        mapper = umap.UMAP(**default_kwargs)
        coords = mapper.fit_transform(embeddings)
        return coords, mapper

    if reducer == "tsne":
        from sklearn.manifold import TSNE

        default_kwargs = dict(
            n_components=n_components,
            init="pca",
            random_state=0,
        )
        default_kwargs.update(reducer_kwargs)
        tsne = TSNE(**default_kwargs)
        return tsne.fit_transform(embeddings), None

    raise ValueError(f"Unknown reducer: {reducer}. Use 'umap' or 'tsne'.")


def _build_point_customdata(
    labels: np.ndarray,
    *,
    sample_indices: np.ndarray | None = None,
    images: np.ndarray | None = None,
    point_metadata: dict[str, np.ndarray] | None = None,
    hover_image_size: int = 64,
) -> tuple[np.ndarray, list[str]]:
    """Build Plotly customdata rows plus field names for the frontend."""
    labels = np.asarray(labels)
    n_samples = len(labels)

    if sample_indices is None:
        sample_indices = np.arange(n_samples)
    sample_indices = np.asarray(sample_indices)
    if len(sample_indices) != n_samples:
        raise ValueError(
            f"sample_indices length ({len(sample_indices)}) must match labels ({n_samples})"
        )

    fields = ["label", "index"]
    columns: list[np.ndarray] = [
        labels.astype(object, copy=False),
        sample_indices.astype(object, copy=False),
    ]

    if images is not None:
        images = np.asarray(images)
        if len(images) != n_samples:
            raise ValueError(
                f"images length ({len(images)}) must match labels ({n_samples})"
            )
        data_uris = [
            _encode_image_for_hover(img, size=hover_image_size) for img in images
        ]
        fields.append("image_b64")
        columns.append(np.asarray(data_uris, dtype=object))

    if point_metadata is not None:
        for field_name, values in point_metadata.items():
            values = np.asarray(values)
            if len(values) != n_samples:
                raise ValueError(
                    f"point_metadata[{field_name!r}] length ({len(values)}) must match labels ({n_samples})"
                )
            fields.append(field_name)
            columns.append(values.astype(object, copy=False))

    customdata = np.empty((n_samples, len(columns)), dtype=object)
    for idx, column in enumerate(columns):
        customdata[:, idx] = column

    return customdata, fields


_SIDE_PANEL_HTML = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    * { box-sizing: border-box; }
    body { margin: 0; font-family: system-ui, sans-serif; }
    .container { display: flex; height: 100vh; }
    .main { flex: 1; display: flex; flex-direction: column; min-width: 0; }
    .controls {
      display: flex;
      align-items: center;
      gap: 1rem;
      padding: 0.5rem 1rem;
      background: #f1f3f5;
      border-bottom: 1px solid #dee2e6;
    }
    .controls label { font-size: 0.9rem; }
    .controls input[type="number"] { width: 4rem; padding: 0.25rem; }
    .controls button { padding: 0.35rem 0.75rem; cursor: pointer; }
    .plot-pane { flex: 1; min-height: 0; display: flex; padding: 8px; }
    #plot { flex: 1; width: 100%; min-height: 0; overflow: visible; }
    .preview-pane {
      width: 320px;
      flex-shrink: 0;
      background: #f8f9fa;
      border-left: 1px solid #dee2e6;
      display: flex;
      flex-direction: column;
      overflow-y: auto;
      padding: 1rem;
    }
    .preview-pane img {
      max-width: 100%;
      border-radius: 8px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }
    .preview-placeholder {
      color: #6c757d;
      font-size: 0.9rem;
      text-align: center;
    }
    .preview-label { margin-top: 0.25rem; font-size: 0.75rem; font-weight: 600; }
    .preview-meta { margin-top: 0.35rem; font-size: 0.9rem; }
    .cluster-grid {
      display: grid;
      gap: 0.75rem;
      grid-template-columns: repeat(auto-fill, minmax(80px, 1fr));
    }
    .cluster-item {
      text-align: center;
      border: 2px solid transparent;
      border-radius: 10px;
      padding: 0.35rem;
      background: white;
    }
    .cluster-item img { width: 100%; height: auto; }
    .cluster-item.correct { border-color: #2a9d8f; }
    .cluster-item.wrong { border-color: #e76f51; }
    .cluster-item.parse-fail { border-color: #6c757d; }
    .summary-box {
      margin-bottom: 0.9rem;
      padding: 0.75rem;
      border-radius: 10px;
      background: white;
      box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }
    .badge {
      display: inline-block;
      padding: 0.2rem 0.45rem;
      border-radius: 999px;
      font-size: 0.72rem;
      font-weight: 700;
      margin-left: 0.35rem;
    }
    .badge-correct { background: #d8f3dc; color: #1b4332; }
    .badge-wrong { background: #ffe8d6; color: #9c6644; }
    .badge-invalid { background: #e9ecef; color: #495057; }
    .summary-list {
      margin: 0.35rem 0 0 0;
      padding-left: 1rem;
      font-size: 0.9rem;
    }
    .summary-list li { margin: 0.2rem 0; }
  </style>
</head>
<body>
  <div class="container">
    <div class="main">
      <div class="controls">
        <span><strong>Cluster mode:</strong></span>
        <label id="embedding-view-label" style="display: none;">Display: <select id="embedding-view-select"></select></label>
        <label id="mode-label" style="display: none;">View: <select id="mode-select"><option value="regular">Regular</option><option value="clip_over_dino">CLIP &gt; DINO</option><option value="dino_over_clip">DINO &gt; CLIP</option></select></label>
        <label id="qwen-filter-label" style="display: none;">Qwen: <select id="qwen-filter-select"><option value="all">All</option><option value="correct">Correct</option><option value="wrong">Wrong</option><option value="parse_fail">Parse fail</option></select></label>
        <label id="point-color-mode-label" style="display: none;">Point colors: <select id="point-color-mode-select"><option value="digit">Digit</option><option value="outcome">Outcome</option><option value="true_label_gap">True-label gap</option><option value="qwen_outcome">Qwen outcome</option></select></label>
        <label>K neighbors: <input type="number" id="k-input" value="9" min="1" max="50"></label>
        <label id="nn-mode-label" style="display: none;">NN in: <select id="nn-mode"><option value="embedding">Embedding space</option><option value="umap">UMAP space</option></select></label>
        <button id="heatmap-toggle-btn" type="button">Show heatmap</button>
        <label id="heatmap-metric-label" style="display: none;">Heatmap: <select id="heatmap-metric-select"><option value="density">Density</option><option value="local_advantage">Local advantage</option><option value="local_error_rate">Local error rate</option></select></label>
        <button id="scatter-toggle-btn" type="button">Hide scatter</button>
        <label>Point size: <input type="range" id="scatter-size-input" value="8" min="2" max="24" step="1"><span id="scatter-size-value">8</span></label>
        <button id="home-scale-toggle-btn" type="button">Home scale: current</button>
        <label>Heatmap bins: <input type="number" id="heatmap-resolution-input" value="40" min="5" max="200" style="width: 72px;"></label>
        <label>Heatmap opacity: <input type="range" id="heatmap-opacity-input" value="0.65" min="0.15" max="1" step="0.05"><span id="heatmap-opacity-value">0.65</span></label>
        <span id="cluster-hint" class="preview-placeholder">Click a point to zoom to its cluster</span>
        <button id="reset-btn">Reset view</button>
        <button id="home-btn">Home</button>
        <span id="query-sep" style="display: none;">|</span>
        <span id="query-section" style="display: none;">
          <label>Query text: <input type="text" id="query-input" placeholder="e.g. an image of the number 3" style="width: 220px;"></label>
          <button id="add-query-btn">Add to plot</button>
        </span>
      </div>
      <div id="plot" class="plot-pane"></div>
    </div>
    <div class="preview-pane">
      <div id="preview-content" class="preview-placeholder">
        Hover over a point to preview
      </div>
    </div>
  </div>
  <script>
    const figure = JSON.parse(atob('FIGURE_JSON'));
    figure.layout.autosize = true;
    delete figure.layout.height;
    const plotDiv = document.getElementById('plot');
    const previewDiv = document.getElementById('preview-content');
    const kInput = document.getElementById('k-input');
    const nnModeSelect = document.getElementById('nn-mode');
    const heatmapToggleBtn = document.getElementById('heatmap-toggle-btn');
    const scatterToggleBtn = document.getElementById('scatter-toggle-btn');
    const scatterSizeInput = document.getElementById('scatter-size-input');
    const scatterSizeValue = document.getElementById('scatter-size-value');
    const homeScaleToggleBtn = document.getElementById('home-scale-toggle-btn');
    const heatmapResolutionInput = document.getElementById('heatmap-resolution-input');
    const heatmapOpacityInput = document.getElementById('heatmap-opacity-input');
    const heatmapOpacityValue = document.getElementById('heatmap-opacity-value');
    const embeddingViewLabel = document.getElementById('embedding-view-label');
    const embeddingViewSelect = document.getElementById('embedding-view-select');
    const modeLabel = document.getElementById('mode-label');
    const modeSelect = document.getElementById('mode-select');
    const qwenFilterLabel = document.getElementById('qwen-filter-label');
    const qwenFilterSelect = document.getElementById('qwen-filter-select');
    const pointColorModeLabel = document.getElementById('point-color-mode-label');
    const pointColorModeSelect = document.getElementById('point-color-mode-select');
    const heatmapMetricLabel = document.getElementById('heatmap-metric-label');
    const heatmapMetricSelect = document.getElementById('heatmap-metric-select');
    const resetBtn = document.getElementById('reset-btn');
    const homeBtn = document.getElementById('home-btn');
    const clusterHint = document.getElementById('cluster-hint');
    const customdataFields = (figure.layout.meta && figure.layout.meta.customdata_fields) || [];
    const initialFilterMode = (figure.layout.meta && figure.layout.meta.initial_filter_mode) || 'regular';
    const baseTitle = (figure.layout.meta && figure.layout.meta.base_title) || 'Latent Space (2D)';
    const analysisMetadata = (figure.layout.meta && figure.layout.meta.analysis_metadata) || {};
    const embedServerUrl = 'EMBED_SERVER_URL';
    const embeddingViewCoords = (figure.layout.meta && figure.layout.meta.embedding_view_coords) || {
      default: {
        x: figure.data[0].x.slice(),
        y: figure.data[0].y.slice(),
      }
    };
    const embeddingViewLabels = (figure.layout.meta && figure.layout.meta.embedding_view_labels) || {};
    const availableEmbeddingViews = Object.keys(embeddingViewCoords);
    const configuredInitialEmbeddingView = (figure.layout.meta && figure.layout.meta.initial_embedding_view) || availableEmbeddingViews[0] || 'default';
    const queryableEmbeddingView = (figure.layout.meta && figure.layout.meta.queryable_embedding_view) || null;
    const DIGIT_LEGEND_TRACE_COUNT = 10;
    const OUTCOME_LEGEND_TRACE_COUNT = 4;
    const OUTCOME_COLOR_BY_KEY = {
      both_correct: '#2a9d8f',
      both_wrong: '#6c757d',
      clip_only_correct: '#e76f51',
      dino_only_correct: '#3a86ff',
    };
    const OUTCOME_LABEL_BY_KEY = {
      both_correct: 'Both correct',
      both_wrong: 'Both wrong',
      clip_only_correct: 'CLIP only correct',
      dino_only_correct: 'DINO only correct',
    };
    const QWEN_OUTCOME_COLOR_BY_KEY = {
      correct: '#2a9d8f',
      wrong: '#e76f51',
      parse_fail: '#6c757d',
    };
    const QWEN_OUTCOME_LABEL_BY_KEY = {
      correct: 'Correct',
      wrong: 'Wrong',
      parse_fail: 'Parse fail',
    };
    const TRUE_LABEL_GAP_COLORSCALE = [
      [0, '#2166ac'],
      [0.5, '#f7f7f7'],
      [1, '#b2182b'],
    ];
    const QWEN_ERROR_RATE_COLORSCALE = [
      [0, '#d8f3dc'],
      [0.5, '#ffe8d6'],
      [1, '#b02a37'],
    ];

    const fullData = {
      customdata: figure.data[0].customdata ? figure.data[0].customdata.map(c => c ? c.slice() : c) : null,
      marker: JSON.parse(JSON.stringify(figure.data[0].marker || {}))
    };
    const comparisonEnabled = customdataFields.indexOf('clip_prediction') !== -1 && customdataFields.indexOf('dino_prediction') !== -1;
    const qwenEnabled = customdataFields.indexOf('qwen_prediction') !== -1
      && customdataFields.indexOf('qwen_parse_valid') !== -1
      && customdataFields.indexOf('qwen_raw_response') !== -1
      && customdataFields.indexOf('qwen_correct') !== -1;
    const trueLabelProbabilityEnabled = comparisonEnabled
      && customdataFields.indexOf('clip_true_label_probability') !== -1
      && customdataFields.indexOf('dino_true_label_probability') !== -1;
    const HEATMAP_TRACE_INDEX = 0;
    const SCATTER_TRACE_INDEX = 1;
    const DIGIT_LEGEND_TRACE_INDICES = Array.from(
      {length: DIGIT_LEGEND_TRACE_COUNT},
      (_, idx) => SCATTER_TRACE_INDEX + 1 + idx
    );
    const OUTCOME_LEGEND_TRACE_INDICES = Array.from(
      {length: OUTCOME_LEGEND_TRACE_COUNT},
      (_, idx) => SCATTER_TRACE_INDEX + 1 + DIGIT_LEGEND_TRACE_COUNT + idx
    );

    let clusterIndices = null;
    let clusterRenderMode = null;
    let clusterCenterIdx = null;
    let currentEmbeddingView = availableEmbeddingViews.indexOf(configuredInitialEmbeddingView) !== -1
      ? configuredInitialEmbeddingView
      : (availableEmbeddingViews[0] || 'default');
    const allIndices = Array.from({length: (embeddingViewCoords[currentEmbeddingView] || {x: []}).x.length}, (_, idx) => idx);
    let modeIndices = allIndices.slice();
    let displayIndices = allIndices.slice();
    let queryTraceIdx = null;
    let heatmapEnabled = false;
    let heatmapResolution = 40;
    let heatmapOpacity = 0.65;
    let heatmapMetric = 'density';
    let pointColorMode = 'digit';
    let qwenFilterMode = 'all';
    let baseScatterSize = 8;
    let currentHomeRelayout = null;
    let scatterVisible = true;
    let lockHomeToRegularScale = false;

    function clampHeatmapResolution(value) {
      if (!Number.isFinite(value)) return 40;
      return Math.max(5, Math.min(200, Math.round(value)));
    }

    function clampHeatmapOpacity(value) {
      if (!Number.isFinite(value)) return 0.65;
      return Math.max(0.15, Math.min(1, Math.round(value * 100) / 100));
    }

    function clampScatterSize(value) {
      if (!Number.isFinite(value)) return 8;
      return Math.max(2, Math.min(24, Math.round(value)));
    }

    function createEmptyHeatmapTrace() {
      return {
        type: 'heatmap',
        x: [0],
        y: [0],
        z: [[0]],
        visible: false,
        opacity: heatmapOpacity,
        colorscale: [
          [0, '#0b2a8f'],
          [0.2, '#1368ce'],
          [0.4, '#1fbad6'],
          [0.6, '#f3e55b'],
          [0.8, '#f98e2b'],
          [1, '#c62020'],
        ],
        hoverinfo: 'skip',
        showscale: false,
        colorbar: {
          title: {text: 'Density'},
          thickness: 16,
          len: 0.8,
          y: 0.5,
          x: 1.03,
        },
        zsmooth: 'best',
        name: 'Density heatmap',
        showlegend: false,
      };
    }

    function buildNormalizedGaussianKernel(radius, sigma) {
      const kernel = [];
      let total = 0;
      for (let offset = -radius; offset <= radius; offset++) {
        const weight = Math.exp(-(offset * offset) / (2 * sigma * sigma));
        kernel.push(weight);
        total += weight;
      }
      return kernel.map(weight => weight / total);
    }

    function convolveGridSeparable(grid, kernel) {
      const height = grid.length;
      const width = height > 0 ? grid[0].length : 0;
      const radius = Math.floor(kernel.length / 2);
      const temp = Array.from({length: height}, () => Array(width).fill(0));
      const output = Array.from({length: height}, () => Array(width).fill(0));

      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          let weightedSum = 0;
          let weightTotal = 0;
          for (let offset = -radius; offset <= radius; offset++) {
            const sampleX = x + offset;
            if (sampleX < 0 || sampleX >= width) continue;
            const weight = kernel[offset + radius];
            weightedSum += grid[y][sampleX] * weight;
            weightTotal += weight;
          }
          temp[y][x] = weightTotal > 0 ? weightedSum / weightTotal : 0;
        }
      }

      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          let weightedSum = 0;
          let weightTotal = 0;
          for (let offset = -radius; offset <= radius; offset++) {
            const sampleY = y + offset;
            if (sampleY < 0 || sampleY >= height) continue;
            const weight = kernel[offset + radius];
            weightedSum += temp[sampleY][x] * weight;
            weightTotal += weight;
          }
          output[y][x] = weightTotal > 0 ? weightedSum / weightTotal : 0;
        }
      }

      return output;
    }

    function buildInterpolatedLocalAdvantageGrid(sums, counts) {
      const radius = Math.max(1, Math.min(6, Math.round(heatmapResolution / 18)));
      const sigma = Math.max(0.8, radius / 1.5);
      const kernel = buildNormalizedGaussianKernel(radius, sigma);
      const smoothedSums = convolveGridSeparable(sums, kernel);
      const smoothedCounts = convolveGridSeparable(counts, kernel);
      return smoothedSums.map((row, yBin) => row.map((value, xBin) => {
        const support = smoothedCounts[yBin][xBin];
        if (support <= 1e-6) {
          return 0;
        }
        return Math.max(-1, Math.min(1, value / support));
      }));
    }

    function getCustomValue(customdata, fieldName) {
      if (!customdata) return null;
      const idx = customdataFields.indexOf(fieldName);
      if (idx === -1 || idx >= customdata.length) return null;
      return customdata[idx];
    }

    function getComparableValue(value) {
      if (value === null || value === undefined || value === '') return null;
      const numericValue = Number(value);
      return Number.isNaN(numericValue) ? String(value) : numericValue;
    }

    function getNumericValue(value) {
      if (value === null || value === undefined || value === '') return null;
      const numericValue = Number(value);
      return Number.isNaN(numericValue) ? null : numericValue;
    }

    function getBooleanValue(value) {
      if (value === null || value === undefined || value === '') return null;
      if (value === true || value === false) return value;
      if (typeof value === 'number') return value !== 0;
      const normalized = String(value).trim().toLowerCase();
      if (normalized === 'true' || normalized === '1') return true;
      if (normalized === 'false' || normalized === '0') return false;
      return null;
    }

    function formatProbability(value) {
      return value === null ? '' : value.toFixed(3);
    }

    function formatSignedValue(value) {
      if (value === null) return '';
      return (value >= 0 ? '+' : '') + value.toFixed(3);
    }

    function resolveOutcomeKey(clipCorrect, dinoCorrect) {
      if (clipCorrect && dinoCorrect) return 'both_correct';
      if (!clipCorrect && !dinoCorrect) return 'both_wrong';
      if (clipCorrect) return 'clip_only_correct';
      return 'dino_only_correct';
    }

    const comparisonPointStats = comparisonEnabled
      ? allIndices.map(idx => {
          const cd = fullData.customdata ? fullData.customdata[idx] : null;
          const label = getComparableValue(getCustomValue(cd, 'label'));
          const clipPrediction = getComparableValue(getCustomValue(cd, 'clip_prediction'));
          const dinoPrediction = getComparableValue(getCustomValue(cd, 'dino_prediction'));
          const clipTrueLabelProbability = getNumericValue(getCustomValue(cd, 'clip_true_label_probability'));
          const dinoTrueLabelProbability = getNumericValue(getCustomValue(cd, 'dino_true_label_probability'));
          const clipCorrect = clipPrediction === label;
          const dinoCorrect = dinoPrediction === label;
          return {
            label: label,
            clipPrediction: clipPrediction,
            dinoPrediction: dinoPrediction,
            clipCorrect: clipCorrect,
            dinoCorrect: dinoCorrect,
            outcomeKey: resolveOutcomeKey(clipCorrect, dinoCorrect),
            localAdvantageScore: clipCorrect === dinoCorrect ? 0 : (clipCorrect ? 1 : -1),
            trueLabelGap: (
              clipTrueLabelProbability === null || dinoTrueLabelProbability === null
            )
              ? null
              : clipTrueLabelProbability - dinoTrueLabelProbability,
            clipTrueLabelProbability: clipTrueLabelProbability,
            dinoTrueLabelProbability: dinoTrueLabelProbability,
          };
        })
      : [];

    const qwenPointStats = qwenEnabled
      ? allIndices.map(idx => {
          const cd = fullData.customdata ? fullData.customdata[idx] : null;
          const label = getComparableValue(getCustomValue(cd, 'label'));
          const qwenPrediction = getComparableValue(getCustomValue(cd, 'qwen_prediction'));
          const qwenParseValid = getBooleanValue(getCustomValue(cd, 'qwen_parse_valid')) === true;
          const qwenCorrect = getBooleanValue(getCustomValue(cd, 'qwen_correct')) === true;
          const qwenRawResponse = getCustomValue(cd, 'qwen_raw_response');
          return {
            label: label,
            qwenPrediction: qwenPrediction,
            qwenParseValid: qwenParseValid,
            qwenCorrect: qwenCorrect,
            qwenRawResponse: qwenRawResponse,
            outcomeKey: !qwenParseValid ? 'parse_fail' : (qwenCorrect ? 'correct' : 'wrong'),
            localErrorScore: (!qwenParseValid || !qwenCorrect) ? 1 : 0,
          };
        })
      : [];

    function getComparisonStats(idx) {
      if (!comparisonEnabled || idx < 0 || idx >= comparisonPointStats.length) return null;
      return comparisonPointStats[idx];
    }

    function getQwenStats(idx) {
      if (!qwenEnabled || idx < 0 || idx >= qwenPointStats.length) return null;
      return qwenPointStats[idx];
    }

    function escapeHtml(value) {
      const div = document.createElement('div');
      div.textContent = value == null ? '' : String(value);
      return div.innerHTML;
    }

    function formatPercent(numerator, denominator) {
      if (!denominator) return '0.0%';
      return ((100 * numerator) / denominator).toFixed(1) + '%';
    }

    function qwenBadgeHtml(outcomeKey) {
      if (outcomeKey === 'correct') {
        return '<span class="badge badge-correct">Correct</span>';
      }
      if (outcomeKey === 'wrong') {
        return '<span class="badge badge-wrong">Wrong</span>';
      }
      return '<span class="badge badge-invalid">Parse fail</span>';
    }

    function computeQwenSubsetStats(indices) {
      if (!qwenEnabled) return null;
      const stats = {
        total: indices.length,
        correct: 0,
        wrong: 0,
        parseFail: 0,
        confusionCounts: {},
      };
      indices.forEach(idx => {
        const pointStats = getQwenStats(idx);
        if (!pointStats) return;
        if (!pointStats.qwenParseValid) {
          stats.parseFail += 1;
          return;
        }
        if (pointStats.qwenCorrect) {
          stats.correct += 1;
        } else {
          stats.wrong += 1;
        }
        const key = String(pointStats.label) + '→' + String(pointStats.qwenPrediction);
        stats.confusionCounts[key] = (stats.confusionCounts[key] || 0) + 1;
      });
      return stats;
    }

    function renderQwenConfusions(confusionCounts) {
      const entries = Object.entries(confusionCounts || {})
        .sort((a, b) => b[1] - a[1] || String(a[0]).localeCompare(String(b[0])))
        .slice(0, 3);
      if (!entries.length) {
        return '<li>No wrong parsed predictions in this view.</li>';
      }
      return entries.map(([pair, count]) => '<li>' + escapeHtml(pair) + ' (' + escapeHtml(count) + ')</li>').join('');
    }

    function renderQwenSummary(indices) {
      const stats = computeQwenSubsetStats(indices);
      if (!stats) {
        return '<span class="preview-placeholder">Hover over a point to preview</span>';
      }
      const parsedCount = stats.total - stats.parseFail;
      return ''
        + '<div class="summary-box">'
        + '<div class="preview-label">Qwen summary</div>'
        + '<div class="preview-meta"><strong>Visible points:</strong> ' + escapeHtml(stats.total) + '</div>'
        + '<div class="preview-meta"><strong>Accuracy:</strong> ' + escapeHtml(formatPercent(stats.correct, stats.total)) + '</div>'
        + '<div class="preview-meta"><strong>Parse rate:</strong> ' + escapeHtml(formatPercent(parsedCount, stats.total)) + '</div>'
        + '<div class="preview-meta"><strong>Correct / wrong / parse fail:</strong> '
        + escapeHtml(stats.correct) + ' / ' + escapeHtml(stats.wrong) + ' / ' + escapeHtml(stats.parseFail) + '</div>'
        + '<div class="preview-meta"><strong>Repo:</strong> ' + escapeHtml(analysisMetadata.qwen_repo || '') + '</div>'
        + '<div class="preview-meta"><strong>Prompt:</strong> ' + escapeHtml(analysisMetadata.qwen_prompt || '') + '</div>'
        + '<div class="preview-meta"><strong>System prompt:</strong> ' + escapeHtml(analysisMetadata.qwen_system_prompt || '') + '</div>'
        + '<div class="preview-meta"><strong>Top confusions:</strong></div>'
        + '<ul class="summary-list">' + renderQwenConfusions(stats.confusionCounts) + '</ul>'
        + '</div>';
    }

    function renderPointPreview(customdata) {
      const imageB64 = getCustomValue(customdata, 'image_b64');
      if (!imageB64) {
        return '<span class="preview-placeholder">Hover over a point to preview</span>';
      }

      const label = getCustomValue(customdata, 'label');
      const index = getCustomValue(customdata, 'index');
      const clipPrediction = getCustomValue(customdata, 'clip_prediction');
      const dinoPrediction = getCustomValue(customdata, 'dino_prediction');
      const clipTrueLabelProbability = getNumericValue(getCustomValue(customdata, 'clip_true_label_probability'));
      const dinoTrueLabelProbability = getNumericValue(getCustomValue(customdata, 'dino_true_label_probability'));
      const qwenPrediction = getComparableValue(getCustomValue(customdata, 'qwen_prediction'));
      const qwenParseValid = getBooleanValue(getCustomValue(customdata, 'qwen_parse_valid')) === true;
      const qwenCorrect = getBooleanValue(getCustomValue(customdata, 'qwen_correct')) === true;
      const qwenRawResponse = getCustomValue(customdata, 'qwen_raw_response');

      let html = '<img src="data:image/png;base64,' + imageB64 + '" alt="">';
      html += '<div class="preview-label">Ground truth: ' + escapeHtml(label) + ' &middot; Index: ' + escapeHtml(index) + '</div>';

      if (clipPrediction !== null && clipPrediction !== '') {
        html += '<div class="preview-meta"><strong>CLIP:</strong> ' + escapeHtml(clipPrediction) + '</div>';
      }
      if (dinoPrediction !== null && dinoPrediction !== '') {
        html += '<div class="preview-meta"><strong>DINO:</strong> ' + escapeHtml(dinoPrediction) + '</div>';
      }
      if (clipTrueLabelProbability !== null) {
        html += '<div class="preview-meta"><strong>CLIP p(true):</strong> ' + escapeHtml(formatProbability(clipTrueLabelProbability)) + '</div>';
      }
      if (dinoTrueLabelProbability !== null) {
        html += '<div class="preview-meta"><strong>DINO p(true):</strong> ' + escapeHtml(formatProbability(dinoTrueLabelProbability)) + '</div>';
      }
      if (clipTrueLabelProbability !== null && dinoTrueLabelProbability !== null) {
        html += '<div class="preview-meta"><strong>Gap:</strong> ' + escapeHtml(
          formatSignedValue(clipTrueLabelProbability - dinoTrueLabelProbability)
        ) + '</div>';
      }
      if (qwenEnabled) {
        const qwenOutcomeKey = !qwenParseValid ? 'parse_fail' : (qwenCorrect ? 'correct' : 'wrong');
        html += '<div class="preview-meta"><strong>Qwen:</strong> '
          + escapeHtml(qwenParseValid ? qwenPrediction : 'invalid')
          + qwenBadgeHtml(qwenOutcomeKey) + '</div>';
        html += '<div class="preview-meta"><strong>Raw response:</strong> ' + escapeHtml(qwenRawResponse || '') + '</div>';
      }

      return html;
    }

    function getModeDisplayName(mode) {
      if (mode === 'clip_over_dino') return 'CLIP > DINO';
      if (mode === 'dino_over_clip') return 'DINO > CLIP';
      return 'Regular';
    }

    function getCurrentMode() {
      return comparisonEnabled ? modeSelect.value : 'regular';
    }

    function getEmbeddingViewDisplayName(view) {
      return embeddingViewLabels[view] || String(view).toUpperCase();
    }

    function getCurrentCoords() {
      return embeddingViewCoords[currentEmbeddingView] || embeddingViewCoords[availableEmbeddingViews[0]];
    }

    function currentViewSupportsTextQueries() {
      return !!embedServerUrl && queryableEmbeddingView !== null && currentEmbeddingView === queryableEmbeddingView;
    }

    function clearQueryTrace() {
      if (queryTraceIdx === null) return;
      if (queryTraceIdx >= plotDiv.data.length) {
        queryTraceIdx = null;
        return;
      }
      Plotly.deleteTraces(plotDiv, [queryTraceIdx]);
      queryTraceIdx = null;
    }

    function updateQueryControls() {
      const queryAvailable = currentViewSupportsTextQueries();
      document.getElementById('query-section').style.display = queryAvailable ? 'inline' : 'none';
      document.getElementById('query-sep').style.display = queryAvailable ? 'inline' : 'none';
      if (!queryAvailable) {
        clearQueryTrace();
      }
    }

    function updateHeatmapControls() {
      heatmapToggleBtn.textContent = heatmapEnabled ? 'Hide heatmap' : 'Show heatmap';
      heatmapResolutionInput.disabled = !heatmapEnabled;
      heatmapOpacityInput.disabled = !heatmapEnabled;
      heatmapResolutionInput.value = String(heatmapResolution);
      heatmapOpacityInput.value = heatmapOpacity.toFixed(2);
      heatmapOpacityValue.textContent = heatmapOpacity.toFixed(2);
      if (comparisonEnabled || qwenEnabled) {
        heatmapMetricSelect.value = heatmapMetric;
      }
    }

    function updateScatterControls() {
      scatterToggleBtn.textContent = scatterVisible ? 'Hide scatter' : 'Show scatter';
    }

    function updateScatterSizeControls() {
      scatterSizeInput.value = String(baseScatterSize);
      scatterSizeValue.textContent = String(baseScatterSize);
    }

    function updateHomeScaleControls() {
      homeScaleToggleBtn.textContent = lockHomeToRegularScale
        ? 'Home scale: regular'
        : 'Home scale: current';
    }

    function updatePointColorControls() {
      if (!comparisonEnabled && !qwenEnabled) return;
      pointColorModeSelect.value = pointColorMode;
    }

    function updateQwenFilterControls() {
      if (!qwenEnabled) return;
      qwenFilterSelect.value = qwenFilterMode;
    }

    function updateLegendVisibility() {
      if (!plotDiv.data || plotDiv.data.length === 0) return;
      const showDigitLegend = pointColorMode === 'digit';
      const showOutcomeLegend = (comparisonEnabled && pointColorMode === 'outcome')
        || (qwenEnabled && pointColorMode === 'qwen_outcome');
      Plotly.restyle(
        plotDiv,
        {visible: DIGIT_LEGEND_TRACE_INDICES.map(() => showDigitLegend)},
        DIGIT_LEGEND_TRACE_INDICES
      );
      Plotly.restyle(
        plotDiv,
        {visible: OUTCOME_LEGEND_TRACE_INDICES.map(() => showOutcomeLegend)},
        OUTCOME_LEGEND_TRACE_INDICES
      );
      if (showOutcomeLegend) {
        if (comparisonEnabled && pointColorMode === 'outcome') {
          const legendKeys = ['both_correct', 'both_wrong', 'clip_only_correct', 'dino_only_correct'];
          Plotly.restyle(
            plotDiv,
            {
              name: legendKeys.map(key => OUTCOME_LABEL_BY_KEY[key]),
              marker: legendKeys.map(key => ({
                size: 10,
                color: OUTCOME_COLOR_BY_KEY[key],
                symbol: 'circle',
                line: {width: 0.5, color: 'white'},
              })),
              visible: legendKeys.map(() => true),
            },
            OUTCOME_LEGEND_TRACE_INDICES,
          );
        } else if (qwenEnabled && pointColorMode === 'qwen_outcome') {
          const legendKeys = ['correct', 'wrong', 'parse_fail'];
          const names = legendKeys
            .map(key => QWEN_OUTCOME_LABEL_BY_KEY[key])
            .concat(['Unused']);
          const markers = legendKeys
            .map(key => ({
              size: 10,
              color: QWEN_OUTCOME_COLOR_BY_KEY[key],
              symbol: 'circle',
              line: {width: 0.5, color: 'white'},
            }))
            .concat([{
              size: 10,
              color: '#ffffff',
              symbol: 'circle',
              line: {width: 0, color: '#ffffff'},
            }]);
          const visible = [true, true, true, false];
          Plotly.restyle(
            plotDiv,
            {name: names, marker: markers, visible: visible},
            OUTCOME_LEGEND_TRACE_INDICES,
          );
        }
      }
    }

    function getHomeReferenceIndices(fallbackIndices) {
      return lockHomeToRegularScale ? allIndices : fallbackIndices;
    }

    function resolveCurrentHomeRelayout() {
      if (lockHomeToRegularScale) {
        return buildRangeRelayout(allIndices);
      }
      return currentHomeRelayout || buildRangeRelayout(displayIndices);
    }

    function getDefaultScatterSize() {
      return baseScatterSize;
    }

    function getHighlightedScatterSize() {
      return Math.max(getDefaultScatterSize() + 8, getDefaultScatterSize() * 2);
    }

    function defaultPreviewHtml() {
      if (modeIndices.length === 0) {
        return '<span class="preview-placeholder">No points match the current mode.</span>';
      }
      if (!scatterVisible) {
        return '<span class="preview-placeholder">Scatter hidden. Show scatter to inspect individual points.</span>';
      }
      if (qwenEnabled) {
        return renderQwenSummary(displayIndices);
      }
      return '<span class="preview-placeholder">Hover over a point to preview</span>';
    }

    function showClusterGrid(indices, centerIdx = null) {
      if (!indices.length) {
        previewDiv.innerHTML = defaultPreviewHtml();
        return;
      }
      if (fullData.customdata && fullData.customdata[0] && getCustomValue(fullData.customdata[0], 'image_b64')) {
        let html = '';
        if (qwenEnabled) {
          const focusIdx = centerIdx === null ? indices[0] : centerIdx;
          const focusCustomdata = fullData.customdata[focusIdx];
          const focusStats = getQwenStats(focusIdx);
          const neighborhoodStats = computeQwenSubsetStats(indices);
          html += '<div class="summary-box">';
          html += renderPointPreview(focusCustomdata);
          if (focusStats) {
            html += '<div class="preview-meta"><strong>Neighborhood:</strong> '
              + escapeHtml(indices.length) + ' samples &middot; '
              + escapeHtml(neighborhoodStats.correct) + ' correct / '
              + escapeHtml(neighborhoodStats.wrong) + ' wrong / '
              + escapeHtml(neighborhoodStats.parseFail) + ' parse fail</div>';
          }
          html += '</div>';
        }
        html += '<div class="cluster-grid">';
        indices.forEach(idx => {
          const cd = fullData.customdata[idx];
          let classes = 'cluster-item';
          if (qwenEnabled) {
            const stats = getQwenStats(idx);
            if (stats) {
              classes += ' ' + (stats.outcomeKey === 'parse_fail' ? 'parse-fail' : stats.outcomeKey);
            }
          }
          html += '<div class="' + classes + '"><img src="data:image/png;base64,' + getCustomValue(cd, 'image_b64') + '" alt=""><div class="preview-label">' + escapeHtml(getCustomValue(cd, 'label')) + '</div>';
          if (qwenEnabled) {
            const stats = getQwenStats(idx);
            if (stats) {
              html += '<div class="preview-meta">Pred: ' + escapeHtml(stats.qwenParseValid ? stats.qwenPrediction : 'invalid') + '</div>';
            }
          }
          html += '</div>';
        });
        html += '</div>';
        previewDiv.innerHTML = html;
        return;
      }
      previewDiv.innerHTML = defaultPreviewHtml();
    }

    function updateClusterHint() {
      if (modeIndices.length === 0) {
        clusterHint.textContent = 'No points match the current mode.';
        return;
      }
      if (!scatterVisible) {
        clusterHint.textContent = 'Scatter hidden. Show scatter to inspect individual points.';
        return;
      }
      if (clusterIndices) {
        if (clusterRenderMode === 'embedding') {
          const clusterScopeLabel = qwenEnabled
            ? ('Qwen ' + qwenFilterMode.replace('_', ' '))
            : getModeDisplayName(getCurrentMode());
          clusterHint.textContent = 'Showing ' + clusterIndices.length + ' neighbors in ' + clusterScopeLabel + '. Click Reset to return to the current mode.';
          return;
        }
        clusterHint.textContent = 'Showing ' + clusterIndices.length + ' points. Click Reset to return to the current mode.';
        return;
      }
      clusterHint.textContent = 'Click a visible point to zoom to its cluster';
    }

    function updateTitle() {
      let title = baseTitle;
      if (availableEmbeddingViews.length > 1) {
        title += ' - ' + getEmbeddingViewDisplayName(currentEmbeddingView);
      }
      if (comparisonEnabled && getCurrentMode() !== 'regular') {
        title += ' - ' + getModeDisplayName(getCurrentMode());
      }
      if (qwenEnabled && qwenFilterMode !== 'all') {
        title += ' - Qwen ' + qwenFilterMode.replace('_', ' ');
      }
      title += ' (' + modeIndices.length + ' points)';
      Plotly.relayout(plotDiv, {title: title});
    }

    function buildTraceData(indices) {
      const coords = getCurrentCoords();
      return {
        x: indices.map(i => coords.x[i]),
        y: indices.map(i => coords.y[i]),
        customdata: fullData.customdata ? indices.map(i => fullData.customdata[i]) : null,
      };
    }

    function buildScatterMarkerConfig(indices, opacity, size) {
      const marker = JSON.parse(JSON.stringify(fullData.marker || {}));
      marker.size = size;
      marker.opacity = opacity;
      marker.line = marker.line || {width: 0.5, color: 'white'};

      if (comparisonEnabled && pointColorMode === 'outcome') {
        marker.color = indices.map(idx => {
          const stats = getComparisonStats(idx);
          return stats ? OUTCOME_COLOR_BY_KEY[stats.outcomeKey] : '#6c757d';
        });
        marker.showscale = false;
        delete marker.colorscale;
        delete marker.cmin;
        delete marker.cmax;
        delete marker.cmid;
        delete marker.colorbar;
        return marker;
      }

      if (comparisonEnabled && pointColorMode === 'true_label_gap' && trueLabelProbabilityEnabled) {
        marker.color = indices.map(idx => {
          const stats = getComparisonStats(idx);
          return stats && stats.trueLabelGap !== null ? stats.trueLabelGap : 0;
        });
        marker.colorscale = TRUE_LABEL_GAP_COLORSCALE;
        marker.cmin = -1;
        marker.cmax = 1;
        marker.cmid = 0;
        marker.showscale = true;
        marker.colorbar = {
          title: {text: 'CLIP - DINO p(true)'},
          thickness: 16,
          len: 0.8,
          y: 0.5,
          x: 1.14,
        };
        return marker;
      }

      if (qwenEnabled && pointColorMode === 'qwen_outcome') {
        marker.color = indices.map(idx => {
          const stats = getQwenStats(idx);
          return stats ? QWEN_OUTCOME_COLOR_BY_KEY[stats.outcomeKey] : '#6c757d';
        });
        marker.showscale = false;
        delete marker.colorscale;
        delete marker.cmin;
        delete marker.cmax;
        delete marker.cmid;
        delete marker.colorbar;
        return marker;
      }

      marker.color = indices.map(i => fullData.marker.color[i]);
      marker.showscale = false;
      delete marker.colorscale;
      delete marker.cmin;
      delete marker.cmax;
      delete marker.cmid;
      delete marker.colorbar;
      return marker;
    }

    function buildHeatmapTrace(indices) {
      if (!heatmapEnabled || indices.length === 0) {
        return createEmptyHeatmapTrace();
      }

      const coords = getCurrentCoords();
      const xs = indices.map(i => coords.x[i]);
      const ys = indices.map(i => coords.y[i]);
      let xMin = Math.min.apply(null, xs);
      let xMax = Math.max.apply(null, xs);
      let yMin = Math.min.apply(null, ys);
      let yMax = Math.max.apply(null, ys);

      if (xMin === xMax) {
        xMin -= 0.5;
        xMax += 0.5;
      }
      if (yMin === yMax) {
        yMin -= 0.5;
        yMax += 0.5;
      }

      const bins = heatmapResolution;
      const xStep = (xMax - xMin) / bins;
      const yStep = (yMax - yMin) / bins;

      const xCenters = Array.from({length: bins}, (_, i) => xMin + (i + 0.5) * xStep);
      const yCenters = Array.from({length: bins}, (_, i) => yMin + (i + 0.5) * yStep);

      if (comparisonEnabled && heatmapMetric === 'local_advantage') {
        const sums = Array.from({length: bins}, () => Array(bins).fill(0));
        const counts = Array.from({length: bins}, () => Array(bins).fill(0));
        indices.forEach(idx => {
          const x = coords.x[idx];
          const y = coords.y[idx];
          const xBin = Math.min(bins - 1, Math.max(0, Math.floor((x - xMin) / xStep)));
          const yBin = Math.min(bins - 1, Math.max(0, Math.floor((y - yMin) / yStep)));
          const stats = getComparisonStats(idx);
          const score = stats ? stats.localAdvantageScore : 0;
          sums[yBin][xBin] += score;
          counts[yBin][xBin] += 1;
        });

        return {
          type: 'heatmap',
          x: xCenters,
          y: yCenters,
          z: buildInterpolatedLocalAdvantageGrid(sums, counts),
          visible: true,
          opacity: heatmapOpacity,
          colorscale: TRUE_LABEL_GAP_COLORSCALE,
          hoverinfo: 'skip',
          showscale: true,
          colorbar: {
            title: {text: 'CLIP advantage'},
            thickness: 16,
            len: 0.8,
            y: 0.5,
            x: 1.03,
          },
          zsmooth: 'best',
          zmin: -1,
          zmax: 1,
          zmid: 0,
        };
      }

      if (qwenEnabled && heatmapMetric === 'local_error_rate') {
        const errorCounts = Array.from({length: bins}, () => Array(bins).fill(0));
        const counts = Array.from({length: bins}, () => Array(bins).fill(0));
        indices.forEach(idx => {
          const x = coords.x[idx];
          const y = coords.y[idx];
          const xBin = Math.min(bins - 1, Math.max(0, Math.floor((x - xMin) / xStep)));
          const yBin = Math.min(bins - 1, Math.max(0, Math.floor((y - yMin) / yStep)));
          const stats = getQwenStats(idx);
          const errorScore = stats && stats.qwenParseValid && stats.qwenCorrect ? 0 : 1;
          errorCounts[yBin][xBin] += errorScore;
          counts[yBin][xBin] += 1;
        });
        const z = errorCounts.map((row, yBin) => row.map((value, xBin) => {
          const support = counts[yBin][xBin];
          return support > 0 ? value / support : 0;
        }));
        return {
          type: 'heatmap',
          x: xCenters,
          y: yCenters,
          z: z,
          visible: true,
          opacity: heatmapOpacity,
          colorscale: QWEN_ERROR_RATE_COLORSCALE,
          hoverinfo: 'skip',
          showscale: true,
          colorbar: {
            title: {text: 'Local error rate'},
            thickness: 16,
            len: 0.8,
            y: 0.5,
            x: 1.03,
          },
          zsmooth: 'best',
          zmin: 0,
          zmax: 1,
        };
      }

      const z = Array.from({length: bins}, () => Array(bins).fill(0));
      indices.forEach(idx => {
        const x = coords.x[idx];
        const y = coords.y[idx];
        const xBin = Math.min(bins - 1, Math.max(0, Math.floor((x - xMin) / xStep)));
        const yBin = Math.min(bins - 1, Math.max(0, Math.floor((y - yMin) / yStep)));
        z[yBin][xBin] += 1;
      });

      return {
        type: 'heatmap',
        x: xCenters,
        y: yCenters,
        z: z,
        visible: true,
        opacity: heatmapOpacity,
        colorscale: [
          [0, '#0b2a8f'],
          [0.2, '#1368ce'],
          [0.4, '#1fbad6'],
          [0.6, '#f3e55b'],
          [0.8, '#f98e2b'],
          [1, '#c62020'],
        ],
        hoverinfo: 'skip',
        showscale: true,
        colorbar: {
          title: {text: 'Density'},
          thickness: 16,
          len: 0.8,
          y: 0.5,
          x: 1.03,
        },
        zsmooth: 'best',
      };
    }

    function buildRangeRelayout(indices) {
      if (!indices.length) {
        return {
          'xaxis.autorange': true,
          'yaxis.autorange': true,
        };
      }

      const coords = getCurrentCoords();
      const xs = indices.map(i => coords.x[i]);
      const ys = indices.map(i => coords.y[i]);
      const xMin = Math.min.apply(null, xs);
      const xMax = Math.max.apply(null, xs);
      const yMin = Math.min.apply(null, ys);
      const yMax = Math.max.apply(null, ys);
      const span = Math.max(xMax - xMin, yMax - yMin) || 1;
      const halfRange = (span * 0.5) + (span * 0.15);
      const xCenter = (xMin + xMax) * 0.5;
      const yCenter = (yMin + yMax) * 0.5;
      return {
        'xaxis.autorange': false,
        'yaxis.autorange': false,
        'xaxis.range': [xCenter - halfRange, xCenter + halfRange],
        'yaxis.range': [yCenter - halfRange, yCenter + halfRange],
      };
    }

    function updateHeatmap() {
      const heatmapTrace = buildHeatmapTrace(displayIndices);
      Plotly.restyle(plotDiv, {
        x: [heatmapTrace.x],
        y: [heatmapTrace.y],
        z: [heatmapTrace.z],
        visible: [heatmapTrace.visible],
        opacity: [heatmapTrace.opacity],
        colorscale: [heatmapTrace.colorscale],
        hoverinfo: [heatmapTrace.hoverinfo],
        showscale: [heatmapTrace.showscale],
        colorbar: [heatmapTrace.colorbar],
        zsmooth: [heatmapTrace.zsmooth],
      }, [HEATMAP_TRACE_INDEX]);
    }

    function updateScatterVisibility() {
      Plotly.restyle(plotDiv, {
        visible: [scatterVisible],
      }, [SCATTER_TRACE_INDEX]);
      if (queryTraceIdx !== null && queryTraceIdx < plotDiv.data.length) {
        Plotly.restyle(plotDiv, {
          visible: [scatterVisible],
        }, [queryTraceIdx]);
      }
    }

    function renderTrace(indices, options = {}) {
      displayIndices = indices.slice();
      const traceData = buildTraceData(displayIndices);
      const opacity = options.opacity || Array(displayIndices.length).fill(1);
      const size = options.size || Array(displayIndices.length).fill(getDefaultScatterSize());
      const marker = buildScatterMarkerConfig(displayIndices, opacity, size);
      currentHomeRelayout = buildRangeRelayout(options.homeIndices || displayIndices);

      Plotly.restyle(plotDiv, {
        x: [traceData.x],
        y: [traceData.y],
        visible: [scatterVisible],
        customdata: traceData.customdata ? [traceData.customdata] : undefined,
        marker: [marker],
      }, [SCATTER_TRACE_INDEX]);
      updateHeatmap();

      if (options.relayout) {
        Plotly.relayout(plotDiv, options.relayout);
      }
    }

    function rerenderCurrentScatterState() {
      if (clusterIndices && clusterRenderMode === 'embedding') {
        const highlighted = new Set(clusterIndices);
        const opacity = modeIndices.map(idx => highlighted.has(idx) ? 1 : 0.04);
        const size = modeIndices.map(idx => highlighted.has(idx) ? getHighlightedScatterSize() : getDefaultScatterSize());
        const homeIndices = getHomeReferenceIndices(modeIndices);
        renderTrace(modeIndices, {
          opacity: opacity,
          size: size,
          homeIndices: homeIndices,
          relayout: null,
        });
        showClusterGrid(clusterIndices, clusterCenterIdx);
        updateClusterHint();
        return;
      }

      if (clusterIndices && clusterRenderMode === 'umap') {
        const homeIndices = getHomeReferenceIndices(clusterIndices);
        renderTrace(clusterIndices, {
          homeIndices: homeIndices,
          relayout: null,
        });
        showClusterGrid(clusterIndices, clusterCenterIdx);
        updateClusterHint();
        return;
      }

      const homeIndices = getHomeReferenceIndices(modeIndices);
      renderTrace(modeIndices, {
        homeIndices: homeIndices,
        relayout: null,
      });
      previewDiv.innerHTML = defaultPreviewHtml();
      updateClusterHint();
    }

    function computeModeIndices(mode) {
      if (qwenEnabled) {
        return allIndices.filter(idx => {
          const stats = getQwenStats(idx);
          if (!stats) return false;
          if (qwenFilterMode === 'correct') return stats.qwenCorrect;
          if (qwenFilterMode === 'wrong') return stats.qwenParseValid && !stats.qwenCorrect;
          if (qwenFilterMode === 'parse_fail') return !stats.qwenParseValid;
          return true;
        });
      }

      if (!comparisonEnabled || mode === 'regular') {
        return allIndices.slice();
      }

      return allIndices.filter(idx => {
        const stats = getComparisonStats(idx);
        if (!stats) return false;

        if (mode === 'clip_over_dino') {
          return stats.clipCorrect && !stats.dinoCorrect;
        }
        if (mode === 'dino_over_clip') {
          return stats.dinoCorrect && !stats.clipCorrect;
        }
        return true;
      });
    }

    function dist(i, j) {
      const coords = getCurrentCoords();
      const dx = coords.x[i] - coords.x[j];
      const dy = coords.y[i] - coords.y[j];
      return dx * dx + dy * dy;
    }

    function getKNearest(centerIdx, k, allowedIndices) {
      const candidates = (allowedIndices && allowedIndices.length > 0) ? allowedIndices : allIndices;
      const distances = [];
      for (let j = 0; j < candidates.length; j++) {
        const idx = candidates[j];
        if (idx === centerIdx) continue;
        distances.push({ idx: idx, d: dist(centerIdx, idx) });
      }
      distances.sort((a, b) => a.d - b.d);
      const neighbors = distances.slice(0, k).map(d => d.idx);
      return [centerIdx].concat(neighbors);
    }

    function showEmbeddingNeighbors(indices, centerIdx = null) {
      clusterIndices = indices.slice();
      clusterRenderMode = 'embedding';
      clusterCenterIdx = centerIdx;
      const highlighted = new Set(clusterIndices);
      const opacity = modeIndices.map(idx => highlighted.has(idx) ? 1 : 0.04);
      const size = modeIndices.map(idx => highlighted.has(idx) ? getHighlightedScatterSize() : getDefaultScatterSize());
      const homeIndices = getHomeReferenceIndices(modeIndices);
      renderTrace(modeIndices, {
        opacity: opacity,
        size: size,
        homeIndices: homeIndices,
        relayout: buildRangeRelayout(homeIndices),
      });
      showClusterGrid(clusterIndices, clusterCenterIdx);
      updateClusterHint();
    }

    function zoomToCluster(indices, centerIdx = null) {
      clusterIndices = indices.slice();
      clusterRenderMode = 'umap';
      clusterCenterIdx = centerIdx;
      if (!clusterIndices.length) {
        const homeIndices = getHomeReferenceIndices([]);
        renderTrace([], {
          homeIndices: homeIndices,
          relayout: buildRangeRelayout([]),
        });
        previewDiv.innerHTML = defaultPreviewHtml();
        updateClusterHint();
        return;
      }

      const homeIndices = getHomeReferenceIndices(clusterIndices);
      renderTrace(clusterIndices, {
        homeIndices: homeIndices,
        relayout: buildRangeRelayout(clusterIndices),
      });
      showClusterGrid(clusterIndices, clusterCenterIdx);
      updateClusterHint();
    }

    function applyModeFilter(resetRanges = true) {
      clusterIndices = null;
      clusterRenderMode = null;
      clusterCenterIdx = null;
      modeIndices = computeModeIndices(getCurrentMode());
      const homeIndices = getHomeReferenceIndices(modeIndices);
      renderTrace(modeIndices, {
        homeIndices: homeIndices,
        relayout: resetRanges ? buildRangeRelayout(homeIndices) : null,
      });
      previewDiv.innerHTML = defaultPreviewHtml();
      updateClusterHint();
      updateTitle();
    }

    function applyEmbeddingViewChange(resetRanges = true) {
      clusterIndices = null;
      clusterRenderMode = null;
      clusterCenterIdx = null;
      const homeIndices = getHomeReferenceIndices(modeIndices);
      renderTrace(modeIndices, {
        homeIndices: homeIndices,
        relayout: resetRanges ? buildRangeRelayout(homeIndices) : null,
      });
      previewDiv.innerHTML = defaultPreviewHtml();
      updateClusterHint();
      updateTitle();
      updateQueryControls();
    }

    function resetView() {
      clusterIndices = null;
      clusterRenderMode = null;
      clusterCenterIdx = null;
      const homeIndices = getHomeReferenceIndices(modeIndices);
      renderTrace(modeIndices, {
        homeIndices: homeIndices,
        relayout: buildRangeRelayout(homeIndices),
      });
      previewDiv.innerHTML = defaultPreviewHtml();
      updateClusterHint();
    }

    const config = {responsive: true, displayModeBar: true, modeBarButtonsToRemove: []};
    baseScatterSize = clampScatterSize(parseInt(scatterSizeInput.value, 10));
    heatmapResolution = clampHeatmapResolution(parseInt(heatmapResolutionInput.value, 10));
    heatmapOpacity = clampHeatmapOpacity(parseFloat(heatmapOpacityInput.value));
    if (!trueLabelProbabilityEnabled) {
      pointColorModeSelect.querySelector('option[value="true_label_gap"]').disabled = true;
    }
    if (!comparisonEnabled) {
      pointColorModeSelect.querySelector('option[value="outcome"]').disabled = true;
    }
    if (!qwenEnabled) {
      pointColorModeSelect.querySelector('option[value="qwen_outcome"]').disabled = true;
      heatmapMetricSelect.querySelector('option[value="local_error_rate"]').disabled = true;
    }
    updateHeatmapControls();
    updateScatterControls();
    updateScatterSizeControls();
    updateHomeScaleControls();
    updatePointColorControls();
    updateQwenFilterControls();
    Plotly.newPlot(plotDiv, [createEmptyHeatmapTrace()].concat(figure.data), figure.layout, config);
    updateLegendVisibility();
    window.addEventListener('resize', function() { Plotly.Plots.resize(plotDiv); });

    function getKNearestToPoint(px, py, k, allowedIndices) {
      const candidates = (allowedIndices && allowedIndices.length > 0) ? allowedIndices : allIndices;
      const coords = getCurrentCoords();
      const distances = [];
      for (let j = 0; j < candidates.length; j++) {
        const idx = candidates[j];
        const dx = coords.x[idx] - px;
        const dy = coords.y[idx] - py;
        distances.push({ idx: idx, d: dx * dx + dy * dy });
      }
      distances.sort((a, b) => a.d - b.d);
      return distances.slice(0, k).map(d => d.idx);
    }

    plotDiv.on('plotly_click', async function(data) {
      if (!data.points || data.points.length === 0) return;
      const pt = data.points[0];
      const curveNumber = pt.curveNumber;
      const k = Math.max(1, parseInt(kInput.value) || 9);
      const useEmbedding = nnModeSelect.value === 'embedding';
      const allowedIndices = modeIndices.slice();

      if (curveNumber === SCATTER_TRACE_INDEX) {
        if (pt.pointNumber >= displayIndices.length) return;
        const centerIdx = displayIndices[pt.pointNumber];
        if (useEmbedding && embedServerUrl) {
          try {
            const resp = await fetch('/k_neighbors', {
              method: 'POST',
              headers: {'Content-Type': 'application/json'},
              body: JSON.stringify({index: centerIdx, k: k, view: currentEmbeddingView, allowed_indices: allowedIndices})
            });
            if (!resp.ok) throw new Error(resp.statusText);
            const {neighbors} = await resp.json();
            showEmbeddingNeighbors(neighbors, centerIdx);
          } catch (e) {
            alert('Failed to get neighbors: ' + e.message);
          }
        } else {
          zoomToCluster(getKNearest(centerIdx, k, allowedIndices), centerIdx);
        }
      } else if (plotDiv.data[curveNumber].name === 'Query') {
        const queryText = pt.customdata && pt.customdata[0];
        if (!queryText) return;
        if (useEmbedding && embedServerUrl) {
          try {
            const resp = await fetch('/k_neighbors_from_text', {
              method: 'POST',
              headers: {'Content-Type': 'application/json'},
              body: JSON.stringify({text: queryText, k: k, view: currentEmbeddingView, allowed_indices: allowedIndices})
            });
            if (!resp.ok) throw new Error(resp.statusText);
            const {neighbors} = await resp.json();
            showEmbeddingNeighbors(neighbors);
          } catch (e) {
            alert('Failed to get neighbors: ' + e.message);
          }
        } else {
          zoomToCluster(getKNearestToPoint(pt.x, pt.y, k, allowedIndices));
        }
      }
    });

    plotDiv.on('plotly_hover', function(data) {
      if (clusterIndices) return;
      if (data.points && data.points.length > 0) {
        const pt = data.points[0];
        const cd = pt.customdata;
        previewDiv.innerHTML = renderPointPreview(cd);
        return;
      }
      previewDiv.innerHTML = defaultPreviewHtml();
    });

    plotDiv.on('plotly_unhover', function() {
      if (clusterIndices) return;
      previewDiv.innerHTML = defaultPreviewHtml();
    });

    resetBtn.onclick = resetView;

    homeBtn.onclick = function() {
      Plotly.relayout(
        plotDiv,
        resolveCurrentHomeRelayout(),
      );
    };

    heatmapToggleBtn.onclick = function() {
      heatmapEnabled = !heatmapEnabled;
      updateHeatmapControls();
      updateHeatmap();
    };

    scatterToggleBtn.onclick = function() {
      scatterVisible = !scatterVisible;
      updateScatterControls();
      updateScatterVisibility();
      previewDiv.innerHTML = defaultPreviewHtml();
      updateClusterHint();
    };

    homeScaleToggleBtn.onclick = function() {
      lockHomeToRegularScale = !lockHomeToRegularScale;
      updateHomeScaleControls();
      currentHomeRelayout = resolveCurrentHomeRelayout();
      updateClusterHint();
    };

    function syncHeatmapResolution() {
      const parsed = parseInt(heatmapResolutionInput.value, 10);
      if (Number.isNaN(parsed)) return;
      const nextResolution = clampHeatmapResolution(parsed);
      if (nextResolution === heatmapResolution) {
        heatmapResolutionInput.value = String(nextResolution);
        return;
      }
      heatmapResolution = nextResolution;
      heatmapResolutionInput.value = String(nextResolution);
      updateHeatmap();
    }

    heatmapResolutionInput.onchange = syncHeatmapResolution;
    heatmapResolutionInput.oninput = syncHeatmapResolution;

    function syncHeatmapOpacity() {
      const parsed = parseFloat(heatmapOpacityInput.value);
      if (Number.isNaN(parsed)) return;
      const nextOpacity = clampHeatmapOpacity(parsed);
      if (nextOpacity === heatmapOpacity) {
        heatmapOpacityInput.value = nextOpacity.toFixed(2);
        heatmapOpacityValue.textContent = nextOpacity.toFixed(2);
        return;
      }
      heatmapOpacity = nextOpacity;
      heatmapOpacityInput.value = nextOpacity.toFixed(2);
      heatmapOpacityValue.textContent = nextOpacity.toFixed(2);
      updateHeatmap();
    }

    heatmapOpacityInput.onchange = syncHeatmapOpacity;
    heatmapOpacityInput.oninput = syncHeatmapOpacity;

    function syncScatterSize() {
      const parsed = parseInt(scatterSizeInput.value, 10);
      if (Number.isNaN(parsed)) return;
      const nextSize = clampScatterSize(parsed);
      if (nextSize === baseScatterSize) {
        updateScatterSizeControls();
        return;
      }
      baseScatterSize = nextSize;
      updateScatterSizeControls();
      rerenderCurrentScatterState();
    }

    scatterSizeInput.onchange = syncScatterSize;
    scatterSizeInput.oninput = syncScatterSize;

    if (availableEmbeddingViews.length > 1) {
      embeddingViewLabel.style.display = 'inline';
      embeddingViewSelect.innerHTML = '';
      availableEmbeddingViews.forEach(view => {
        const option = document.createElement('option');
        option.value = view;
        option.textContent = getEmbeddingViewDisplayName(view);
        embeddingViewSelect.appendChild(option);
      });
      embeddingViewSelect.value = currentEmbeddingView;
      embeddingViewSelect.onchange = function() {
        currentEmbeddingView = embeddingViewSelect.value;
        applyEmbeddingViewChange(true);
      };
    }

    if (comparisonEnabled) {
      modeLabel.style.display = 'inline';
      pointColorModeLabel.style.display = 'inline';
      heatmapMetricLabel.style.display = 'inline';
      modeSelect.value = initialFilterMode;
      modeSelect.onchange = function() {
        applyModeFilter(true);
      };
      pointColorModeSelect.onchange = function() {
        pointColorMode = pointColorModeSelect.value;
        if (pointColorMode === 'true_label_gap' && !trueLabelProbabilityEnabled) {
          pointColorMode = 'digit';
        }
        updatePointColorControls();
        updateLegendVisibility();
        rerenderCurrentScatterState();
      };
      heatmapMetricSelect.onchange = function() {
        heatmapMetric = heatmapMetricSelect.value;
        updateHeatmapControls();
        updateHeatmap();
      };
    } else if (qwenEnabled) {
      qwenFilterLabel.style.display = 'inline';
      pointColorModeLabel.style.display = 'inline';
      heatmapMetricLabel.style.display = 'inline';
      qwenFilterSelect.onchange = function() {
        qwenFilterMode = qwenFilterSelect.value;
        updateQwenFilterControls();
        applyModeFilter(true);
      };
      pointColorModeSelect.onchange = function() {
        pointColorMode = pointColorModeSelect.value;
        if (pointColorMode !== 'digit' && pointColorMode !== 'qwen_outcome') {
          pointColorMode = 'digit';
        }
        updatePointColorControls();
        updateLegendVisibility();
        rerenderCurrentScatterState();
      };
      heatmapMetricSelect.onchange = function() {
        heatmapMetric = heatmapMetricSelect.value;
        if (heatmapMetric !== 'density' && heatmapMetric !== 'local_error_rate') {
          heatmapMetric = 'density';
        }
        updateHeatmapControls();
        updateHeatmap();
      };
    } else {
      modeSelect.value = 'regular';
    }
    applyModeFilter(false);
    updateQueryControls();

    if (embedServerUrl) {
      document.getElementById('nn-mode-label').style.display = 'inline';
      const queryInput = document.getElementById('query-input');
      const addQueryBtn = document.getElementById('add-query-btn');

      addQueryBtn.onclick = async function() {
        if (!currentViewSupportsTextQueries()) return;
        const text = queryInput.value.trim();
        if (!text) return;
        addQueryBtn.disabled = true;
        try {
          const resp = await fetch('/embed_text', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({text: text, view: currentEmbeddingView})
          });
          if (!resp.ok) throw new Error(resp.statusText);
          const {x, y} = await resp.json();
          if (queryTraceIdx === null) {
            Plotly.addTraces(plotDiv, [{
              x: [x], y: [y],
              customdata: [[text]],
              mode: 'markers',
              marker: {size: 16, symbol: 'diamond', color: '#e74c3c', line: {width: 2, color: 'white'}},
              name: 'Query',
              visible: scatterVisible,
              showlegend: true,
              hovertemplate: '<b>Query</b>: %{customdata[0]}<extra></extra>'
            }]);
            queryTraceIdx = plotDiv.data.length - 1;
          } else {
            const trace = plotDiv.data[queryTraceIdx];
            const newX = trace.x.concat([x]);
            const newY = trace.y.concat([y]);
            const newCd = (trace.customdata || []).concat([[text]]);
            Plotly.restyle(plotDiv, {x: [newX], y: [newY], customdata: [newCd]}, [queryTraceIdx]);
          }
        } catch (e) {
          alert('Failed to embed text: ' + e.message);
        }
        addQueryBtn.disabled = false;
      };
    }
  </script>
</body>
</html>
"""


def _make_embed_server(
    model: Any,
    processor: Any,
    mappers_by_view: dict[str, Any],
    device: str,
    html_content: bytes,
    embedding_views: dict[str, np.ndarray],
    *,
    queryable_view: str | None = None,
    port: int = 8765,
) -> tuple[HTTPServer, str]:
    """Create an HTTP server that serves the HTML, embeds text, and computes K-NN in embedding space."""
    import torch
    import torch.nn.functional as F

    normalized_embedding_views: dict[str, np.ndarray] = {}
    for view_name, view_embeddings in embedding_views.items():
        view_embeddings = np.asarray(view_embeddings, dtype=np.float32)
        if view_embeddings.ndim == 1:
            view_embeddings = view_embeddings.reshape(1, -1)
        normalized_embedding_views[view_name] = view_embeddings
    embedding_views = normalized_embedding_views
    default_view = next(iter(embedding_views))

    def resolve_view(view: str | None) -> str:
        resolved_view = view or default_view
        if resolved_view not in embedding_views:
            raise ValueError(f"Unknown embedding view: {resolved_view}")
        return resolved_view

    def resolve_allowed_indices(
        view: str,
        allowed_indices: list[int] | None,
    ) -> np.ndarray:
        view_embeddings = embedding_views[view]
        if not allowed_indices:
            return np.arange(len(view_embeddings), dtype=np.int64)
        allowed = np.asarray(allowed_indices, dtype=np.int64)
        if allowed.ndim != 1:
            raise ValueError("allowed_indices must be a 1D list of integers")
        if np.any(allowed < 0) or np.any(allowed >= len(view_embeddings)):
            raise ValueError("allowed_indices contains an out-of-range sample index")
        return allowed

    def k_neighbors_embedding(
        center_idx: int,
        k: int,
        view: str | None = None,
        allowed_indices: list[int] | None = None,
    ) -> list[int]:
        """K nearest neighbors in embedding space (cosine distance)."""
        resolved_view = resolve_view(view)
        view_embeddings = embedding_views[resolved_view]
        candidates = resolve_allowed_indices(resolved_view, allowed_indices)
        center = view_embeddings[center_idx : center_idx + 1]
        sim = view_embeddings[candidates] @ center.T
        sim = np.squeeze(sim)
        nearest = candidates[np.argsort(-sim)]
        nearest = [int(idx) for idx in nearest if int(idx) != center_idx][:k]
        return [int(center_idx)] + nearest

    def embed_text(text: str, view: str | None = None) -> tuple[np.ndarray, str]:
        resolved_view = resolve_view(view)
        if queryable_view is None or resolved_view != queryable_view:
            raise ValueError(f"Text queries are not supported for embedding view {resolved_view}")
        mapper = mappers_by_view.get(resolved_view)
        if mapper is None:
            raise ValueError(f"Embedding view {resolved_view} does not support text projection")

        with torch.inference_mode():
            text_inputs = processor(
                text=[text],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            )
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            text_latents = model.text_model(**text_inputs)
            text_features = model.text_projection(text_latents.pooler_output)
            normalized = F.normalize(text_features, dim=-1)
            emb = normalized.cpu().numpy()
        return emb, resolved_view

    def k_neighbors_from_text(
        text: str,
        k: int,
        view: str | None = None,
        allowed_indices: list[int] | None = None,
    ) -> tuple[list[int], np.ndarray]:
        """K nearest image neighbors to a text embedding. Returns (neighbor_indices, text_embedding)."""
        center, resolved_view = embed_text(text, view=view)
        view_embeddings = embedding_views[resolved_view]
        candidates = resolve_allowed_indices(resolved_view, allowed_indices)
        if len(candidates) == 0:
            return [], center
        sim = view_embeddings[candidates] @ center.T
        sim = np.squeeze(sim)
        nearest = candidates[np.argsort(-sim)[:k]]
        return [int(i) for i in nearest], center

    def embed_text_coords(text: str, view: str | None = None) -> tuple[float, float]:
        emb, resolved_view = embed_text(text, view=view)
        mapper = mappers_by_view[resolved_view]
        coords = mapper.transform(emb)
        return float(coords[0, 0]), float(coords[0, 1])

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path in ("/", ""):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(html_content)
            else:
                self.send_response(404)
                self.end_headers()

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                data = json.loads(body.decode()) if body else {}
            except json.JSONDecodeError:
                data = {}
            try:
                if self.path == "/embed_text":
                    text = data.get("text", "")
                    x, y = embed_text_coords(text, view=data.get("view"))
                    result = {"x": x, "y": y}
                elif self.path == "/k_neighbors":
                    idx = int(data.get("index", 0))
                    k = int(data.get("k", 9))
                    neighbors = k_neighbors_embedding(
                        idx,
                        k,
                        view=data.get("view"),
                        allowed_indices=data.get("allowed_indices"),
                    )
                    result = {"neighbors": neighbors}
                elif self.path == "/k_neighbors_from_text":
                    text = data.get("text", "")
                    k = int(data.get("k", 9))
                    neighbors, _ = k_neighbors_from_text(
                        text,
                        k,
                        view=data.get("view"),
                        allowed_indices=data.get("allowed_indices"),
                    )
                    result = {"neighbors": neighbors}
                else:
                    self.send_response(404)
                    self.end_headers()
                    return
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())

        def log_message(self, format, *args):
            pass

    server = HTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}"
    return server, url


def _write_side_panel_html(
    fig: go.Figure,
    output_path: Path,
    embed_server_url: str = "",
) -> None:
    """Write interactive HTML with scatter plot and right-side image preview panel."""
    fig_json = fig.to_json()
    fig_b64 = base64.b64encode(fig_json.encode()).decode()
    html = _SIDE_PANEL_HTML.replace("FIGURE_JSON", fig_b64).replace(
        "EMBED_SERVER_URL", embed_server_url
    )
    output_path.write_text(html, encoding="utf-8")


def plot_latent_space(
    embeddings: np.ndarray,
    labels: np.ndarray,
    images: np.ndarray | None = None,
    *,
    reducer: Literal["umap", "tsne"] = "umap",
    reducer_kwargs: dict | None = None,
    hover_image_size: int = 64,
    color_continuous_scale: str = "turbo",
    height: int = 700,
    output_path: Path | str | None = None,
    open_browser: bool = True,
    model: Any = None,
    processor: Any = None,
    device: str = "cpu",
    embed_server_port: int = 8765,
    sample_indices: np.ndarray | None = None,
    point_metadata: dict[str, np.ndarray] | None = None,
    analysis_metadata: dict[str, Any] | None = None,
    title: str = "Latent Space (2D)",
    embedding_views: dict[str, np.ndarray] | None = None,
    initial_embedding_view: str | None = None,
    embedding_view_labels: dict[str, str] | None = None,
    queryable_embedding_view: str | None = None,
    initial_filter_mode: str = "regular",
) -> go.Figure | Path:
    """
    Create an interactive Plotly scatter plot of dimensionality-reduced embeddings.

    When images are provided, writes an HTML file with a right-side preview panel
    that shows the hovered point's image. Otherwise returns a Plotly Figure.

    Args:
        embeddings: High-dimensional embeddings, shape (n_samples, n_features).
        labels: Integer labels for each sample, shape (n_samples,).
        images: Optional images for preview panel, shape (n_samples, H, W) or
            (n_samples, H, W, C). Must align with embeddings by index.
        reducer: Dimensionality reduction method: "umap" or "tsne".
        reducer_kwargs: Extra arguments passed to the reducer.
        hover_image_size: Size of preview image in pixels.
        color_continuous_scale: Plotly color scale for labels.
        height: Figure height in pixels.
        output_path: Path for HTML output when images provided. Default: temp file.
        open_browser: Whether to open the HTML in the browser.
        model: Optional CLIP model for text embedding (enables query textbox when with processor).
        processor: Optional CLIP processor for text embedding.
        device: Device for model inference.
        embed_server_port: Port for the text embedding server (default 8765).
        sample_indices: Optional original dataset indices for each plotted point.
        point_metadata: Optional extra per-point metadata shown in the side panel.
        analysis_metadata: Optional global analysis metadata consumed by the frontend.
        title: Plot title.
        embedding_views: Optional mapping from view name to embedding matrix. When
            provided, the frontend can switch the plotted UMAP/t-SNE coordinates
            and embedding-space KNN between these views on the fly.
        initial_embedding_view: Initial embedding view to display.
        embedding_view_labels: Optional mapping from view name to display label.
        queryable_embedding_view: Optional embedding view that supports text queries.
        initial_filter_mode: Initial frontend filter mode when comparison metadata is available.

    Returns:
        Path to the HTML file when images are provided, else the Plotly Figure.
    """
    reducer_kwargs = reducer_kwargs or {}
    embeddings = np.asarray(embeddings)
    labels = np.asarray(labels)

    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got shape {embeddings.shape}")
    if len(labels) != len(embeddings):
        raise ValueError(
            f"labels length ({len(labels)}) must match embeddings ({len(embeddings)})"
        )

    if embedding_views is None:
        resolved_initial_embedding_view = initial_embedding_view or "default"
        normalized_embedding_views = {resolved_initial_embedding_view: embeddings}
    else:
        normalized_embedding_views = {}
        for view_name, view_embeddings in embedding_views.items():
            view_embeddings = np.asarray(view_embeddings)
            if view_embeddings.ndim != 2:
                raise ValueError(
                    f"embedding_views[{view_name!r}] must be 2D, got shape {view_embeddings.shape}"
                )
            if len(view_embeddings) != len(labels):
                raise ValueError(
                    f"embedding_views[{view_name!r}] length ({len(view_embeddings)}) "
                    f"must match labels ({len(labels)})"
                )
            normalized_embedding_views[view_name] = view_embeddings
        resolved_initial_embedding_view = (
            initial_embedding_view or next(iter(normalized_embedding_views))
        )

    if resolved_initial_embedding_view not in normalized_embedding_views:
        raise ValueError(
            f"initial_embedding_view {resolved_initial_embedding_view!r} is not available"
        )
    resolved_queryable_embedding_view = queryable_embedding_view
    if (
        resolved_queryable_embedding_view is None
        and model is not None
        and processor is not None
    ):
        resolved_queryable_embedding_view = resolved_initial_embedding_view

    if (
        resolved_queryable_embedding_view is not None
        and resolved_queryable_embedding_view not in normalized_embedding_views
    ):
        raise ValueError(
            f"queryable_embedding_view {resolved_queryable_embedding_view!r} is not available"
        )

    if embedding_view_labels is None:
        embedding_view_labels = {
            view_name: view_name.upper() for view_name in normalized_embedding_views
        }

    coords_by_view: dict[str, np.ndarray] = {}
    mappers_by_view: dict[str, Any] = {}
    for view_name, view_embeddings in normalized_embedding_views.items():
        view_coords, view_mapper = _reduce_dimensions(
            view_embeddings,
            reducer,
            **reducer_kwargs,
        )
        coords_by_view[view_name] = view_coords
        mappers_by_view[view_name] = view_mapper

    coords = coords_by_view[resolved_initial_embedding_view]
    x, y = coords[:, 0], coords[:, 1]

    hover_parts = ["<b>Label</b>: %{customdata[0]}", "<b>Index</b>: %{customdata[1]}"]
    customdata, customdata_fields = _build_point_customdata(
        labels,
        sample_indices=sample_indices,
        images=images,
        point_metadata=point_metadata,
        hover_image_size=hover_image_size,
    )
    # Image preview is shown in side panel, not in hover tooltip.

    hovertemplate = "<br>".join(hover_parts) + "<extra></extra>"

    # Discrete palette for digits 0-9 (Plotly qualitative)
    _PALETTE = [
        "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
        "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
    ]
    _COMPARISON_OUTCOME_LEGEND = [
        ("Both correct", "#2a9d8f"),
        ("Both wrong", "#6c757d"),
        ("CLIP only correct", "#e76f51"),
        ("DINO only correct", "#3a86ff"),
    ]
    marker_colors = [_PALETTE[int(l) % 10] for l in labels]

    traces = [
        go.Scatter(
            x=x.tolist(),
            y=y.tolist(),
            mode="markers",
            marker=dict(
                size=8,
                color=marker_colors,
                line=dict(width=0.5, color="white"),
            ),
            customdata=customdata.tolist() if hasattr(customdata, "tolist") else customdata,
            hovertemplate=hovertemplate,
            showlegend=False,
        )
    ]
    for digit in range(10):
        traces.append(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=10, color=_PALETTE[digit], symbol="square", line=dict(width=0.5, color="white")),
                name=str(digit),
                showlegend=True,
                legendgroup=str(digit),
            )
        )
    for legend_name, legend_color in _COMPARISON_OUTCOME_LEGEND:
        traces.append(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(
                    size=10,
                    color=legend_color,
                    symbol="circle",
                    line=dict(width=0.5, color="white"),
                ),
                name=legend_name,
                showlegend=True,
                visible=False,
                legendgroup=f"comparison-{legend_name.lower().replace(' ', '-')}",
            )
        )

    fig = go.Figure(
        data=traces,
        layout=go.Layout(
            title=title,
            xaxis=dict(
                title=f"{reducer.upper()} 1",
                scaleanchor="y",
                scaleratio=1,
            ),
            yaxis=dict(
                title=f"{reducer.upper()} 2",
            ),
            height=height,
            hovermode="closest",
            template="plotly_white",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=80, r=150, b=60, l=60),
            meta=dict(
                customdata_fields=customdata_fields,
                base_title=title,
                analysis_metadata=analysis_metadata or {},
                embedding_view_coords={
                    view_name: dict(
                        x=view_coords[:, 0].tolist(),
                        y=view_coords[:, 1].tolist(),
                    )
                    for view_name, view_coords in coords_by_view.items()
                },
                embedding_view_labels=embedding_view_labels,
                initial_embedding_view=resolved_initial_embedding_view,
                queryable_embedding_view=resolved_queryable_embedding_view,
                initial_filter_mode=initial_filter_mode,
            ),
        ),
    )

    if images is not None:
        out = Path(output_path) if output_path else Path(tempfile.gettempdir()) / "latent_space.html"
        query_view = resolved_queryable_embedding_view
        query_mapper = mappers_by_view.get(query_view) if query_view is not None else None
        if model is not None and processor is not None and query_view is not None and query_mapper is not None:
            embed_url = f"http://127.0.0.1:{embed_server_port}"
            _write_side_panel_html(fig, out, embed_server_url=embed_url)
            html_bytes = out.read_bytes()
            server, _ = _make_embed_server(
                model,
                processor,
                mappers_by_view,
                device,
                html_bytes,
                normalized_embedding_views,
                queryable_view=query_view,
                port=embed_server_port,
            )
            if open_browser:

                def _open_after_delay():
                    import time

                    time.sleep(0.5)
                    webbrowser.open(embed_url)

                threading.Thread(target=_open_after_delay, daemon=True).start()
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
        else:
            _write_side_panel_html(fig, out, embed_server_url="")
            if open_browser:
                webbrowser.open(f"file://{out.resolve()}")
        return out
    return fig
