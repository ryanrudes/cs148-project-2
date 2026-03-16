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
    .cluster-grid {
      display: grid;
      gap: 0.75rem;
      grid-template-columns: repeat(auto-fill, minmax(80px, 1fr));
    }
    .cluster-item { text-align: center; }
    .cluster-item img { width: 100%; height: auto; }
  </style>
</head>
<body>
  <div class="container">
    <div class="main">
      <div class="controls">
        <span><strong>Cluster mode:</strong></span>
        <label>K neighbors: <input type="number" id="k-input" value="9" min="1" max="50"></label>
        <label id="nn-mode-label" style="display: none;">NN in: <select id="nn-mode"><option value="embedding">Embedding space</option><option value="umap">UMAP space</option></select></label>
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
    const resetBtn = document.getElementById('reset-btn');
    const homeBtn = document.getElementById('home-btn');
    const clusterHint = document.getElementById('cluster-hint');

    const fullData = {
      x: figure.data[0].x.slice(),
      y: figure.data[0].y.slice(),
      customdata: figure.data[0].customdata ? figure.data[0].customdata.map(c => c ? c.slice() : c) : null,
      marker: JSON.parse(JSON.stringify(figure.data[0].marker || {}))
    };

    let clusterIndices = null;

    function dist(i, j) {
      const dx = fullData.x[i] - fullData.x[j];
      const dy = fullData.y[i] - fullData.y[j];
      return dx * dx + dy * dy;
    }

    function getKNearest(centerIdx, k) {
      const n = fullData.x.length;
      const distances = [];
      for (let j = 0; j < n; j++) {
        if (j === centerIdx) continue;
        distances.push({ idx: j, d: dist(centerIdx, j) });
      }
      distances.sort((a, b) => a.d - b.d);
      const neighbors = distances.slice(0, k).map(d => d.idx);
      return [centerIdx].concat(neighbors);
    }

    function showEmbeddingNeighbors(indices) {
      const n = fullData.x.length;
      const opacity = Array(n).fill(0.04);
      const sizes = Array(n).fill(8);
      indices.forEach(i => { opacity[i] = 1; sizes[i] = 16; });
      Plotly.restyle(plotDiv, {
        x: [fullData.x],
        y: [fullData.y],
        customdata: fullData.customdata ? [fullData.customdata] : undefined,
        'marker.color': [fullData.marker.color],
        'marker.opacity': [opacity],
        'marker.size': [sizes]
      }, [0]);
      Plotly.relayout(plotDiv, { 'xaxis.range': undefined, 'yaxis.range': undefined });
      if (fullData.customdata && fullData.customdata[0] && fullData.customdata[0][2]) {
        let html = '<div class="cluster-grid">';
        indices.forEach(idx => {
          const cd = fullData.customdata[idx];
          html += '<div class="cluster-item"><img src="data:image/png;base64,' + cd[2] + '" alt=""><div class="preview-label">' + cd[0] + '</div></div>';
        });
        html += '</div>';
        previewDiv.innerHTML = html;
      }
      clusterHint.textContent = 'Showing ' + indices.length + ' neighbors (embedding space). Click Reset to show all.';
    }

    function zoomToCluster(indices) {
      const xs = indices.map(i => fullData.x[i]);
      const ys = indices.map(i => fullData.y[i]);
      const xMin = Math.min.apply(null, xs);
      const xMax = Math.max.apply(null, xs);
      const yMin = Math.min.apply(null, ys);
      const yMax = Math.max.apply(null, ys);
      const pad = Math.max((xMax - xMin), (yMax - yMin)) * 0.15 || 1;
      const xRange = [xMin - pad, xMax + pad];
      const yRange = [yMin - pad, yMax + pad];

      const filteredX = indices.map(i => fullData.x[i]);
      const filteredY = indices.map(i => fullData.y[i]);
      const filteredCd = fullData.customdata ? indices.map(i => fullData.customdata[i]) : null;
      const filteredColor = indices.map(i => fullData.marker.color[i]);

      Plotly.restyle(plotDiv, {
        x: [filteredX],
        y: [filteredY],
        customdata: filteredCd ? [filteredCd] : undefined,
        'marker.color': [filteredColor]
      }, [0]);

      Plotly.relayout(plotDiv, {
        'xaxis.range': xRange,
        'yaxis.range': yRange
      });

      if (fullData.customdata && fullData.customdata[0] && fullData.customdata[0][2]) {
        let html = '<div class="cluster-grid">';
        indices.forEach(idx => {
          const cd = fullData.customdata[idx];
          html += '<div class="cluster-item"><img src="data:image/png;base64,' + cd[2] + '" alt=""><div class="preview-label">' + cd[0] + '</div></div>';
        });
        html += '</div>';
        previewDiv.innerHTML = html;
      }
      clusterHint.textContent = 'Showing ' + indices.length + ' points. Click Reset to show all.';
    }

    function resetView() {
      clusterIndices = null;
      const n = fullData.x.length;
      Plotly.restyle(plotDiv, {
        x: [fullData.x],
        y: [fullData.y],
        customdata: fullData.customdata ? [fullData.customdata] : undefined,
        'marker.color': [fullData.marker.color],
        'marker.opacity': [Array(n).fill(1)],
        'marker.size': [Array(n).fill(8)]
      }, [0]);
      Plotly.relayout(plotDiv, {
        'xaxis.range': undefined,
        'yaxis.range': undefined
      });
      previewDiv.innerHTML = '<span class="preview-placeholder">Hover over a point to preview</span>';
      clusterHint.textContent = 'Click a point to zoom to its cluster';
    }

    const config = {responsive: true, displayModeBar: true, modeBarButtonsToRemove: []};
    Plotly.newPlot(plotDiv, figure.data, figure.layout, config);
    window.addEventListener('resize', function() { Plotly.Plots.resize(plotDiv); });

    function getKNearestToPoint(px, py, k) {
      const n = fullData.x.length;
      const distances = [];
      for (let j = 0; j < n; j++) {
        const dx = fullData.x[j] - px;
        const dy = fullData.y[j] - py;
        distances.push({ idx: j, d: dx * dx + dy * dy });
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

      if (curveNumber === 0) {
        let centerIdx = pt.pointNumber;
        if (clusterIndices) centerIdx = clusterIndices[centerIdx];
        if (useEmbedding && embedServerUrl) {
          try {
            const resp = await fetch('/k_neighbors', {
              method: 'POST',
              headers: {'Content-Type': 'application/json'},
              body: JSON.stringify({index: centerIdx, k: k})
            });
            if (!resp.ok) throw new Error(resp.statusText);
            const {neighbors} = await resp.json();
            clusterIndices = neighbors;
            showEmbeddingNeighbors(clusterIndices);
          } catch (e) {
            alert('Failed to get neighbors: ' + e.message);
          }
        } else {
          clusterIndices = getKNearest(centerIdx, k);
          zoomToCluster(clusterIndices);
        }
      } else if (plotDiv.data[curveNumber].name === 'Query') {
        const queryText = pt.customdata && pt.customdata[0];
        if (!queryText) return;
        if (useEmbedding && embedServerUrl) {
          try {
            const resp = await fetch('/k_neighbors_from_text', {
              method: 'POST',
              headers: {'Content-Type': 'application/json'},
              body: JSON.stringify({text: queryText, k: k})
            });
            if (!resp.ok) throw new Error(resp.statusText);
            const {neighbors} = await resp.json();
            clusterIndices = neighbors;
            showEmbeddingNeighbors(clusterIndices);
          } catch (e) {
            alert('Failed to get neighbors: ' + e.message);
          }
        } else {
          clusterIndices = getKNearestToPoint(pt.x, pt.y, k);
          zoomToCluster(clusterIndices);
        }
      }
    });

    plotDiv.on('plotly_hover', function(data) {
      if (clusterIndices) return;
      if (data.points && data.points.length > 0) {
        const pt = data.points[0];
        const cd = pt.customdata;
        if (cd && cd[2]) {
          previewDiv.innerHTML = '<img src="data:image/png;base64,' + cd[2] + '" alt=""><div class="preview-label">Label: ' + cd[0] + ' &middot; Index: ' + cd[1] + '</div>';
          return;
        }
      }
      previewDiv.innerHTML = '<span class="preview-placeholder">Hover over a point to preview</span>';
    });

    plotDiv.on('plotly_unhover', function() {
      if (clusterIndices) return;
      previewDiv.innerHTML = '<span class="preview-placeholder">Hover over a point to preview</span>';
    });

    resetBtn.onclick = resetView;

    homeBtn.onclick = function() {
      Plotly.relayout(plotDiv, {
        'xaxis.range': undefined,
        'yaxis.range': undefined
      });
    };

    const embedServerUrl = 'EMBED_SERVER_URL';
    if (embedServerUrl) {
      document.getElementById('query-section').style.display = 'inline';
      document.getElementById('query-sep').style.display = 'inline';
      document.getElementById('nn-mode-label').style.display = 'inline';
      const queryInput = document.getElementById('query-input');
      const addQueryBtn = document.getElementById('add-query-btn');
      let queryTraceIdx = null;

      addQueryBtn.onclick = async function() {
        const text = queryInput.value.trim();
        if (!text) return;
        addQueryBtn.disabled = true;
        try {
          const resp = await fetch('/embed_text', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({text: text})
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
    mapper: Any,
    device: str,
    html_content: bytes,
    embeddings: np.ndarray,
    port: int = 8765,
) -> tuple[HTTPServer, str]:
    """Create an HTTP server that serves the HTML, embeds text, and computes K-NN in embedding space."""
    import torch
    import torch.nn.functional as F

    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)

    def k_neighbors_embedding(center_idx: int, k: int) -> list[int]:
        """K nearest neighbors in embedding space (cosine distance)."""
        n = len(embeddings)
        center = embeddings[center_idx : center_idx + 1]
        sim = embeddings @ center.T
        sim = np.squeeze(sim)
        sim[center_idx] = -np.inf
        nearest = np.argsort(-sim)[:k]
        return [int(center_idx)] + [int(i) for i in nearest]

    def k_neighbors_from_text(text: str, k: int) -> tuple[list[int], np.ndarray]:
        """K nearest image neighbors to a text embedding. Returns (neighbor_indices, text_embedding)."""
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
            center = F.normalize(text_features, dim=-1).cpu().numpy()
        sim = embeddings @ center.T
        sim = np.squeeze(sim)
        nearest = np.argsort(-sim)[:k]
        return [int(i) for i in nearest], center

    def embed_text(text: str) -> tuple[float, float]:
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
                    x, y = embed_text(text)
                    result = {"x": x, "y": y}
                elif self.path == "/k_neighbors":
                    idx = int(data.get("index", 0))
                    k = int(data.get("k", 9))
                    neighbors = k_neighbors_embedding(idx, k)
                    result = {"neighbors": neighbors}
                elif self.path == "/k_neighbors_from_text":
                    text = data.get("text", "")
                    k = int(data.get("k", 9))
                    neighbors, _ = k_neighbors_from_text(text, k)
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
    if images is not None and len(images) != len(embeddings):
        raise ValueError(
            f"images length ({len(images)}) must match embeddings ({len(embeddings)})"
        )

    coords, mapper = _reduce_dimensions(embeddings, reducer, **reducer_kwargs)
    x, y = coords[:, 0], coords[:, 1]

    hover_parts = ["<b>Label</b>: %{customdata[0]}", "<b>Index</b>: %{customdata[1]}"]
    customdata = np.column_stack([labels, np.arange(len(labels))])

    if images is not None:
        images = np.asarray(images)
        data_uris = [
            _encode_image_for_hover(img, size=hover_image_size) for img in images
        ]
        customdata = np.column_stack([labels, np.arange(len(labels)), data_uris])
        # Image preview is shown in side panel, not in hover tooltip

    hovertemplate = "<br>".join(hover_parts) + "<extra></extra>"

    # Discrete palette for digits 0-9 (Plotly qualitative)
    _PALETTE = [
        "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
        "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
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

    fig = go.Figure(
        data=traces,
        layout=go.Layout(
            title="Latent Space (2D)",
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
            margin=dict(t=80, r=80, b=60, l=60),
        ),
    )

    if images is not None:
        out = Path(output_path) if output_path else Path(tempfile.gettempdir()) / "latent_space.html"
        if model is not None and processor is not None and mapper is not None:
            embed_url = f"http://127.0.0.1:{embed_server_port}"
            _write_side_panel_html(fig, out, embed_server_url=embed_url)
            html_bytes = out.read_bytes()
            server, _ = _make_embed_server(
                model, processor, mapper, device, html_bytes, embeddings, embed_server_port
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
