// Web map for netseedf. Python drives it through the `netseedf` object below
// and answers hover and click queries through the QWebChannel object `bridge`.
"use strict";

const netseedf = (() => {
  const osmAttribution =
    '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors';
  // Coastlines and borders from the bundled Natural Earth files (see setOutlines): the
  // "None" basemap, which works offline. Drawn in their own pane, above the data.
  const outlineRenderer = L.canvas({ pane: "outlines" });
  const coastlines = L.geoJSON(null, {
    style: { color: "#333", weight: 1, fill: false },
    renderer: outlineRenderer,
    interactive: false,
  });
  const borders = L.geoJSON(null, {
    style: { color: "#777", weight: 0.7, fill: false },
    renderer: outlineRenderer,
    interactive: false,
  });
  const basemaps = {
    "OpenStreetMap": L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19, attribution: osmAttribution,
    }),
    // The 2016 mosaic: CC BY 4.0 (later years are non-commercial only). 10 m pixels,
    // so tiles past zoom 15 would only be upsampled: Leaflet scales those up itself.
    "Sentinel-2 cloudless": L.tileLayer(
      "https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless_3857/default/g/{z}/{y}/{x}.jpg", {
        maxZoom: 19, maxNativeZoom: 15,
        attribution: '<a href="https://s2maps.eu">Sentinel-2 cloudless</a> by ' +
          '<a href="https://eox.at">EOX IT Services GmbH</a> (Contains modified Copernicus ' +
          "Sentinel data 2016)",
      }),
    "OpenTopoMap": L.tileLayer("https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png", {
      maxZoom: 17,
      attribution: osmAttribution + ', SRTM | Style: &copy; <a href="https://opentopomap.org">' +
        "OpenTopoMap</a> (CC-BY-SA)",
    }),
    "None": L.layerGroup([coastlines, borders]),
  };

  const map = L.map("map", { worldCopyJump: true, preferCanvas: true, zoomSnap: 0.25 })
    .setView([20, 0], 2);
  const outlinePane = map.createPane("outlines");
  outlinePane.style.zIndex = 440; // above the data (overlayPane, 400), below the grid lines
  outlinePane.style.pointerEvents = "none";
  basemaps["OpenStreetMap"].addTo(map);
  L.control.layers(basemaps).addTo(map);
  L.control.scale().addTo(map);

  const info = L.control({ position: "bottomleft" });
  info.onAdd = () => L.DomUtil.create("div", "info");
  info.addTo(map);
  const legend = L.control({ position: "bottomright" });
  legend.onAdd = () => L.DomUtil.create("div", "legend");
  legend.addTo(map);

  // Grid cell outlines go in their own pane, above the data images (which would
  // otherwise cover the canvas the lines are drawn on).
  const gridPane = map.createPane("gridlines");
  gridPane.style.zIndex = 450;
  gridPane.style.pointerEvents = "none"; // clicks and hovering go to the map beneath
  const gridLines = L.layerGroup().addTo(map);
  const gridRenderer = L.canvas({ pane: "gridlines" });

  let dataLayer = null; // overview of all the data
  let detailLayer = null; // the visible part in more detail, after zooming in
  let detailCovers = null; // area the detail layer holds all data for
  let opacity = 0.75;
  let bridge = null;

  // Data drawn on a canvas, redrawn for every view (subclasses implement _draw), so
  // it's exact at any zoom. The canvas covers the viewport plus `padding` (a fraction
  // of its size) on each side. With `redrawWhileMoving` it's redrawn every frame of a
  // drag; otherwise it moves along with the map and is redrawn when the move ends
  // (or when dragged past the padding): for layers too slow to draw every frame.
  // The map wraps around (and jumps by 360° when panned across the antimeridian), so
  // the data is drawn on every copy of the world on the canvas: subclasses set
  // `_lonRange` (its west and east) and draw with `_shift` added to longitudes.
  const CanvasOverlay = L.Layer.extend({
    options: { padding: 0, redrawWhileMoving: true },
    _lonRange: null,
    _shift: 0,

    onAdd() {
      this._canvas = L.DomUtil.create("canvas", "data-overlay leaflet-zoom-animated");
      this._canvas.style.pointerEvents = "none";
      this.getPane().appendChild(this._canvas);
      this._frame = null;
      this.setOpacity(this.options.opacity);
      this._reset();
    },

    onRemove() {
      L.Util.cancelAnimFrame(this._frame);
      this._frame = null;
      L.DomUtil.remove(this._canvas);
    },

    getEvents() {
      return {
        zoomanim: this._animateZoom, move: this._onMove, moveend: this._reset,
        viewreset: this._reset, resize: this._reset,
      };
    },

    // A hidden layer (the overview under a detail, or the other way round) isn't
    // drawn until it's shown.
    setOpacity(value) {
      this.options.opacity = value;
      if (this._canvas) {
        this._canvas.style.opacity = value;
        if (value > 0 && this._dirty) {
          this._redraw();
        }
      }
      return this;
    },

    // Redraw at most once a frame, and not during the zoom animation (which scales
    // the last drawing instead).
    _onMove() {
      if (this._frame || this._map._animatingZoom) {
        return;
      }
      if (!this.options.redrawWhileMoving && this._coversView()) {
        return;
      }
      this._frame = L.Util.requestAnimFrame(() => {
        this._frame = null;
        this._reset();
      });
    },

    _coversView() {
      const topLeft = this._map.containerPointToLayerPoint([0, 0]);
      return this._drawn.contains(topLeft) && this._drawn.contains(topLeft.add(this._map.getSize()));
    },

    _reset() {
      const map = this._map;
      if (!map) {
        return;
      }
      const size = map.getSize();
      const pad = size.multiplyBy(this.options.padding).round();
      const full = size.add(pad.multiplyBy(2));
      this._origin = map.containerPointToLayerPoint([0, 0]).subtract(pad); // canvas's top left
      this._drawn = L.bounds(this._origin, this._origin.add(full));
      this._topLeft = map.layerPointToLatLng(this._origin);
      this._ratio = window.devicePixelRatio || 1;
      L.DomUtil.setPosition(this._canvas, this._origin);
      this._canvas.style.width = `${full.x}px`;
      this._canvas.style.height = `${full.y}px`;
      this._canvas.width = Math.round(full.x * this._ratio);
      this._canvas.height = Math.round(full.y * this._ratio);
      this._redraw();
    },

    _redraw() {
      if (!this._map || !this._canvas) {
        return;
      }
      this._dirty = this.options.opacity === 0;
      const ctx = this._canvas.getContext("2d");
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, this._canvas.width, this._canvas.height);
      if (this._dirty) {
        return;
      }
      // A detail is drawn only inside the area it holds (`clip`), the overview only
      // outside it (`setHole`): together they fill the view without overlapping.
      const clip = this.options.clip;
      if (clip) {
        ctx.save();
        ctx.beginPath();
        ctx.rect(...this._rect(clip));
        ctx.clip();
      }
      for (const shift of this._shifts()) {
        this._shift = shift;
        this._draw(ctx);
      }
      this._shift = 0;
      if (clip) {
        ctx.restore();
      }
      if (this._hole) {
        ctx.clearRect(...this._rect(this._hole));
      }
    },

    // Multiples of 360° that put a copy of the data on the canvas.
    _shifts() {
      if (!this._lonRange) {
        return [0];
      }
      const [west, east] = this._lonRange;
      const left = this._map.layerPointToLatLng(this._origin).lng;
      const right = this._map.layerPointToLatLng(this._drawn.max).lng;
      const shifts = [];
      for (let k = Math.ceil((left - east) / 360); k <= Math.floor((right - west) / 360); k++) {
        shifts.push(360 * k);
      }
      return shifts;
    },

    setHole(bounds) {
      if (bounds !== this._hole) {
        this._hole = bounds;
        this._redraw();
      }
    },

    // [x, y, width, height] of a LatLngBounds in canvas pixels. Rounded, so a hole and
    // a clip of the same bounds meet exactly.
    _rect(bounds) {
      const nw = this._map.latLngToLayerPoint(bounds.getNorthWest()).subtract(this._origin);
      const se = this._map.latLngToLayerPoint(bounds.getSouthEast()).subtract(this._origin);
      const x = Math.round(nw.x * this._ratio);
      const y = Math.round(nw.y * this._ratio);
      return [x, y, Math.round(se.x * this._ratio) - x, Math.round(se.y * this._ratio) - y];
    },

    // Canvas pixels of a position (on the copy of the world being drawn).
    _x(lon, lat) {
      return (this._map.latLngToLayerPoint([lat, lon + this._shift]).x - this._origin.x) * this._ratio;
    },

    _y(lat, lon) {
      return (this._map.latLngToLayerPoint([lat, lon]).y - this._origin.y) * this._ratio;
    },

    _animateZoom(e) {
      const scale = this._map.getZoomScale(e.zoom);
      const offset = this._map._latLngToNewLayerPoint(this._topLeft, e.zoom, e.center);
      L.DomUtil.setTransform(this._canvas, offset, scale);
    },
  });

  // A regular grid: an image with one pixel per cell. Each image row and column is
  // stretched between its own edges (uneven in Web Mercator, and for thinned-out
  // grids), so cells land exactly where they belong.
  const GridOverlay = CanvasOverlay.extend({
    initialize(payload, options) {
      L.setOptions(this, options);
      this._rows = payload.rows; // latitude edges, north to south
      this._cols = payload.cols; // longitude edges, west to east
      this._lonRange = [this._cols[0], this._cols[this._cols.length - 1]];
      this._strip = document.createElement("canvas"); // see _draw
      this._image = new Image();
      this._image.onload = () => this._redraw();
      this._image.src = payload.url;
      // Decoded before the layer is shown (see setData): no blank frame in between.
      this.ready = this._image.decode().catch(() => {});
    },

    _draw(ctx) {
      const image = this._image;
      if (!image.complete || !image.naturalWidth) {
        return;
      }
      const { width, height } = this._canvas;
      // Rounding each edge (rather than each cell's size) keeps neighbouring cells
      // touching, without gaps or overlaps.
      const lat0 = this._rows[0];
      const lon0 = this._cols[0];
      const xs = this._cols.map((lon) => Math.round(this._x(lon, lat0)));
      const ys = this._rows.map((lat) => Math.round(this._y(lat, lon0)));

      // Only the columns in view: far zoomed in, the whole grid would be huge.
      let c0 = 0;
      while (c0 < xs.length - 1 && xs[c0 + 1] <= 0) {
        c0++;
      }
      let c1 = xs.length - 1;
      while (c1 > c0 && xs[c1 - 1] >= width) {
        c1--;
      }
      if (c0 >= c1) {
        return;
      }

      // First the rows, each between its edges, into a strip one pixel per column wide;
      // then each column of the strip between its edges. Two passes of a few thousand
      // draws at most, where drawing every cell could take millions.
      const strip = this._strip;
      strip.width = c1 - c0;
      strip.height = height;
      const sctx = strip.getContext("2d");
      sctx.imageSmoothingEnabled = false;
      for (let r = 0; r < ys.length - 1; r++) {
        const top = Math.max(ys[r], 0);
        const bottom = Math.min(ys[r + 1], height);
        if (bottom > top) {
          sctx.drawImage(image, c0, r, c1 - c0, 1, 0, top, c1 - c0, bottom - top);
        }
      }
      ctx.imageSmoothingEnabled = false;
      for (let c = c0; c < c1; c++) {
        const left = Math.max(xs[c], 0);
        const right = Math.min(xs[c + 1], width);
        if (right > left) {
          ctx.drawImage(strip, c - c0, 0, 1, height, left, 0, right - left, height);
        }
      }
    },
  });

  // WebGL for the cell layers below, shared (browsers allow only a few contexts) and
  // made on first use: {canvas, gl, program, position, color, transform}, or null
  // where there's no WebGL.
  let cellGL;

  function cellRenderer() {
    if (cellGL === undefined || (cellGL && cellGL.gl.isContextLost())) {
      cellGL = makeCellRenderer();
    }
    return cellGL;
  }

  function makeCellRenderer() {
    const canvas = document.createElement("canvas");
    // Not antialiased: triangles sharing an edge then meet exactly, without seams.
    const gl = canvas.getContext("webgl", { antialias: false, preserveDrawingBuffer: true });
    if (!gl) {
      return null;
    }
    const program = gl.createProgram();
    for (const [type, source] of [
      // A position is projected at zoom 0 (relative to its layer's `_ref`, see
      // _vertices); `transform` takes it to clip space: position * xy + zw.
      [gl.VERTEX_SHADER, `
        attribute vec2 position;
        attribute vec4 color;
        uniform vec4 transform;
        varying vec4 v_color;
        void main() {
          v_color = color;
          gl_Position = vec4(position * transform.xy + transform.zw, 0.0, 1.0);
        }`],
      [gl.FRAGMENT_SHADER, `
        precision mediump float;
        varying vec4 v_color;
        void main() {
          gl_FragColor = v_color;
        }`],
    ]) {
      const shader = gl.createShader(type);
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      gl.attachShader(program, shader);
    }
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      console.log(`WebGL: ${gl.getProgramInfoLog(program)}`);
      return null;
    }
    return {
      canvas, gl, program,
      position: gl.getAttribLocation(program, "position"),
      color: gl.getAttribLocation(program, "color"),
      transform: gl.getUniformLocation(program, "transform"),
    };
  }

  // A curvilinear grid with few enough cells: every cell a quadrilateral, with the
  // same corners as the grid lines. Drawn with WebGL, two triangles a cell, and copied
  // onto the layer's canvas: that takes a millisecond or so, so it's redrawn while
  // dragging. Without WebGL, a 2D canvas takes about 10 microseconds a cell, far too
  // slow for that: a padding of half the view each way moves along instead.
  const CellsOverlay = CanvasOverlay.extend({
    initialize(payload, options) {
      L.setOptions(this, options);
      const nx = payload.shape[1];
      this._nx = nx;
      this._lat = payload.lat; // corners, (ny + 1) x (nx + 1)
      this._lon = payload.lon;
      this._lonRange = [payload.bounds[0][1], payload.bounds[1][1]];
      // Cells by colour: one path to fill per colour, not per cell.
      this._byColor = new Map();
      payload.colors.forEach((color, k) => {
        if (color) {
          const cells = this._byColor.get(color) || [];
          cells.push(Math.floor(k / nx) * (nx + 1) + (k % nx)); // its first corner
          this._byColor.set(color, cells);
        }
      });
      this._buffers = null; // see _upload
      if (!cellRenderer()) {
        L.setOptions(this, { padding: 0.5, redrawWhileMoving: false });
      }
    },

    onRemove() {
      CanvasOverlay.prototype.onRemove.call(this);
      if (this._buffers && !this._buffers.gl.isContextLost()) {
        this._buffers.gl.deleteBuffer(this._buffers.position);
        this._buffers.gl.deleteBuffer(this._buffers.color);
      }
      this._buffers = null;
    },

    _draw(ctx) {
      const r = cellRenderer();
      if (r) {
        this._drawGL(ctx, r);
      } else {
        this._draw2D(ctx);
      }
    },

    _drawGL(ctx, r) {
      const { canvas, gl } = r;
      if (!this._buffers || this._buffers.gl !== gl) {
        this._upload(gl);
      }
      if (!this._buffers.count) {
        return;
      }
      const { width, height } = this._canvas;
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }
      gl.viewport(0, 0, width, height);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.useProgram(r.program);
      gl.bindBuffer(gl.ARRAY_BUFFER, this._buffers.position);
      gl.enableVertexAttribArray(r.position);
      gl.vertexAttribPointer(r.position, 2, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, this._buffers.color);
      gl.enableVertexAttribArray(r.color);
      gl.vertexAttribPointer(r.color, 4, gl.UNSIGNED_BYTE, true, 0, 0);
      // Canvas pixels of a position p: ((p + ref) * scale - pixel origin - origin) * ratio.
      const map = this._map;
      const s = map.getZoomScale(map.getZoom(), 0) * this._ratio;
      const o = map.getPixelOrigin().add(this._origin).multiplyBy(this._ratio);
      const crs = map.options.crs;
      const shift = crs.latLngToPoint(L.latLng(0, this._shift), 0).x - crs.latLngToPoint(L.latLng(0, 0), 0).x;
      const x0 = (this._ref.x + shift) * s - o.x;
      const y0 = this._ref.y * s - o.y;
      gl.uniform4f(r.transform, 2 * s / width, -2 * s / height, 2 * x0 / width - 1, 1 - 2 * y0 / height);
      gl.drawArrays(gl.TRIANGLES, 0, this._buffers.count);
      ctx.drawImage(canvas, 0, 0);
    },

    // The cells' triangles into WebGL buffers of `gl`.
    _upload(gl) {
      const { positions, colors } = this._vertices();
      const buffer = (data) => {
        const b = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, b);
        gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
        return b;
      };
      this._buffers = {
        gl, position: buffer(positions), color: buffer(colors), count: positions.length / 2,
      };
    },

    // Six vertices a cell, each with its cell's colour. Positions are projected at
    // zoom 0 and taken relative to `_ref`, a corner of the grid: as absolute ones,
    // 32-bit floats would be off by pixels far zoomed in.
    _vertices() {
      const crs = this._map.options.crs;
      const n = this._lat.length;
      const xs = new Float64Array(n);
      const ys = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        const p = crs.latLngToPoint(L.latLng(this._lat[i], this._lon[i]), 0);
        xs[i] = p.x;
        ys[i] = p.y;
      }
      let count = 0;
      for (const cells of this._byColor.values()) {
        count += cells.length;
      }
      const positions = new Float32Array(count * 12);
      const colors = new Uint8Array(count * 24);
      const first = count ? this._byColor.values().next().value[0] : 0; // corners all valid
      this._ref = L.point(xs[first], ys[first]);
      const row = this._nx + 1;
      let v = 0;
      for (const [color, cells] of this._byColor) {
        const rgb = [1, 3, 5].map((i) => parseInt(color.slice(i, i + 2), 16)); // "#rrggbb"
        for (const a of cells) {
          for (const k of [a, a + 1, a + row + 1, a, a + row + 1, a + row]) {
            positions[2 * v] = xs[k] - this._ref.x;
            positions[2 * v + 1] = ys[k] - this._ref.y;
            colors.set(rgb, 4 * v);
            colors[4 * v + 3] = 255;
            v++;
          }
        }
      }
      return { positions, colors };
    },

    _draw2D(ctx) {
      const n = this._lat.length;
      const xs = new Float64Array(n);
      const ys = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        xs[i] = this._x(this._lon[i], this._lat[i]);
        ys[i] = this._y(this._lat[i], this._lon[i]);
      }
      const row = this._nx + 1;
      // Filled edges are antialiased, which would leave faint seams between cells;
      // outlining each cell in its own colour closes them.
      ctx.lineWidth = 1;
      ctx.lineJoin = "round";
      for (const [color, cells] of this._byColor) {
        ctx.beginPath();
        for (const a of cells) {
          ctx.moveTo(xs[a], ys[a]);
          ctx.lineTo(xs[a + 1], ys[a + 1]);
          ctx.lineTo(xs[a + row + 1], ys[a + row + 1]);
          ctx.lineTo(xs[a + row], ys[a + row]);
          ctx.closePath();
        }
        ctx.fillStyle = ctx.strokeStyle = color;
        ctx.fill();
        ctx.stroke();
      }
    },
  });

  // A curvilinear grid with too many cells for polygons: an image stretched linearly
  // between its bounds in Web Mercator (its rows are laid out for that).
  const ImageCanvasOverlay = CanvasOverlay.extend({
    initialize(payload, options) {
      L.setOptions(this, options);
      this._bounds = L.latLngBounds(payload.bounds);
      this._lonRange = [this._bounds.getWest(), this._bounds.getEast()];
      this._image = new Image();
      this._image.onload = () => this._redraw();
      this._image.src = payload.url;
      // Decoded before the layer is shown (see setData): no blank frame in between.
      this.ready = this._image.decode().catch(() => {});
    },

    _draw(ctx) {
      const image = this._image;
      if (!image.complete || !image.naturalWidth) {
        return;
      }
      const b = this._bounds;
      const [x, y] = [this._x(b.getWest(), b.getNorth()), this._y(b.getNorth(), b.getWest())];
      const [w, h] = [this._x(b.getEast(), b.getSouth()) - x, this._y(b.getSouth(), b.getEast()) - y];
      // Only the part on the canvas: far zoomed in, the whole image would be huge.
      const x0 = Math.max(x, 0);
      const y0 = Math.max(y, 0);
      const x1 = Math.min(x + w, this._canvas.width);
      const y1 = Math.min(y + h, this._canvas.height);
      if (x1 <= x0 || y1 <= y0) {
        return;
      }
      const sx = image.naturalWidth / w;
      const sy = image.naturalHeight / h;
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(image, (x0 - x) * sx, (y0 - y) * sy, (x1 - x0) * sx, (y1 - y0) * sy,
        x0, y0, x1 - x0, y1 - y0);
    },
  });

  function makeLayer(payload) {
    // A detail's `covers`: the area it holds all the data for.
    const options = {
      opacity, pane: "overlayPane", clip: payload.covers ? L.latLngBounds(payload.covers) : null,
    };
    if (payload.kind === "grid") {
      return new GridOverlay(payload, options);
    }
    if (payload.kind === "cells") {
      return new CellsOverlay(payload, options);
    }
    if (payload.kind === "image") {
      return new ImageCanvasOverlay(payload, options);
    }
    if (payload.kind === "points") {
      // On the neighbouring copies of the world too, for panning across the antimeridian.
      return L.layerGroup([-360, 0, 360].flatMap((shift) => payload.points.map(([lat, lon, color]) =>
        L.circleMarker([lat, lon + shift], {
          radius: 5, weight: 0.6, color: "#222", opacity,
          fillColor: color || "#000", fillOpacity: color ? opacity : 0, missing: !color,
        }))));
    }
    return null;
  }

  function layerOpacity(layer, value) {
    if (!layer) {
      return;
    }
    if (layer.setOpacity) {
      layer.setOpacity(value);
    } else {
      layer.eachLayer((m) => m.setStyle({
        opacity: value, fillOpacity: m.options.missing ? 0 : value,
      }));
    }
  }

  // Show either the detail (when it covers the whole view) or the overview, never
  // both: two half-transparent layers on top of each other would look more opaque.
  function updateOverview() {
    if (dataLayer && dataLayer.setHole && (!detailLayer || detailLayer.setHole)) {
      // Canvas layers: the detail where it has data, the overview around it (e.g.
      // while panning, until the detail for the new view arrives).
      dataLayer.setHole(detailLayer ? detailCovers : null);
      layerOpacity(dataLayer, opacity);
      layerOpacity(detailLayer, opacity);
      return;
    }
    const covered = detailLayer && detailCovers && detailCovers.contains(map.getBounds());
    layerOpacity(dataLayer, covered ? 0 : opacity);
    layerOpacity(detailLayer, covered ? opacity : 0);
  }

  // Resolves once a layer can be drawn in full (its image decoded).
  function whenReady(layer) {
    return (layer && layer.ready) || Promise.resolve();
  }

  // New data (e.g. after moving a slider) replaces the old only once it's ready to
  // draw, together with its detail when one follows (`awaitDetail`): the old layers
  // stay up until then, rather than leaving the map blank or coarse for a moment.
  let staged = null; // {payload, data, detail, covers, detailArrived}
  let detailToken = 0; // the latest setDetail, or setData (which drops pending details)

  function setData(payload) {
    detailToken++;
    let arrived;
    const s = staged = {
      payload, data: makeLayer(payload), detail: null, covers: null,
      detailArrived: new Promise((resolve) => { arrived = resolve; }),
    };
    s.arrived = arrived;
    if (!payload.awaitDetail) {
      arrived();
    } else {
      setTimeout(arrived, 1000); // in case it never comes
    }
    Promise.all([whenReady(s.data), s.detailArrived]).then(() => commit(s));
  }

  function commit(s) {
    if (staged !== s) {
      return; // newer data came along
    }
    staged = null;
    for (const layer of [dataLayer, detailLayer]) {
      if (layer) {
        map.removeLayer(layer);
      }
    }
    dataLayer = s.data;
    detailLayer = s.detail;
    detailCovers = s.covers;
    for (const layer of [dataLayer, detailLayer]) {
      if (layer) {
        layer.addTo(map);
      }
    }
    updateOverview();
    const payload = s.payload;
    let bounds = null;
    if (payload.kind === "image" || payload.kind === "grid" || payload.kind === "cells") {
      bounds = L.latLngBounds(payload.bounds);
    } else if (payload.kind === "points" && payload.points.length) {
      bounds = L.latLngBounds(payload.points.map(([lat, lon]) => [lat, lon]));
    }
    if (bounds && bounds.getEast() - bounds.getWest() > 300) {
      // Global: centred on Greenwich, also when stored as 0..360.
      bounds = L.latLngBounds([[bounds.getSouth(), -180], [bounds.getNorth(), 180]]);
    }
    if (payload.fit && bounds) {
      map.fitBounds(bounds, { padding: [20, 20], maxZoom: 12 });
    }
    setLegend(payload.legend);
    refreshPopup();
  }

  // The old detail also stays up until the new one is ready to draw.
  function setDetail(payload) {
    const token = ++detailToken;
    const layer = payload ? makeLayer(payload) : null;
    const covers = payload ? L.latLngBounds(payload.covers) : null;
    whenReady(layer).then(() => {
      if (token !== detailToken) {
        return; // superseded
      }
      if (staged) { // goes up with the data it belongs to
        staged.detail = layer;
        staged.covers = covers;
        staged.arrived();
        return;
      }
      if (detailLayer) {
        map.removeLayer(detailLayer);
      }
      detailLayer = layer;
      detailCovers = covers;
      if (detailLayer) {
        detailLayer.addTo(map);
      }
      updateOverview();
    });
  }

  function setLegend(l) {
    const div = legend.getContainer();
    if (!l) {
      div.style.display = "none";
      return;
    }
    div.style.display = "";
    div.innerHTML = "";
    const title = L.DomUtil.create("div", "title", div);
    title.textContent = l.title;
    const bar = L.DomUtil.create("div", "bar", div);
    bar.style.background = `linear-gradient(to right, ${l.colors.join(", ")})`;
    const labels = L.DomUtil.create("div", "labels", div);
    for (const text of [l.min, l.max]) {
      L.DomUtil.create("span", "", labels).textContent = text;
    }
  }

  function setGridLines(lines) {
    gridLines.clearLayers();
    if (lines && lines.length) {
      gridLines.addLayer(L.polyline(lines, {
        renderer: gridRenderer, color: "#222", weight: 0.7, opacity: 0.6, interactive: false,
      }));
    }
  }

  function setOpacity(value) {
    opacity = value;
    updateOverview();
  }

  // {coastlines, borders}: GeoJSON of each (or null if it couldn't be loaded).
  function setOutlines(outlines) {
    for (const [layer, geojson] of [[coastlines, outlines.coastlines], [borders, outlines.borders]]) {
      layer.clearLayers();
      if (geojson) {
        layer.addData(geojson);
      }
    }
  }

  function clear() {
    closePopup();
    setGridLines(null);
    setData({ kind: "none", legend: null });
  }

  // Hover readout: ask Python for the value under the mouse, at most every 40 ms.
  let pending = null;
  let busy = false;
  function query() {
    if (!bridge || busy || !pending) {
      return;
    }
    const { lat, lng } = pending;
    pending = null;
    busy = true;
    bridge.valueAt(lat, lng, (text) => {
      info.getContainer().textContent = text;
      busy = false;
      setTimeout(query, 40);
    });
  }
  map.on("mousemove", (e) => {
    pending = e.latlng;
    query();
  });
  map.on("mouseout", () => {
    pending = null;
    info.getContainer().textContent = "";
  });

  // Clicking shows the cell's value in a popup, with a button to export its time series.
  const popup = L.popup({ maxWidth: 360 });

  function popupContent(p) {
    const div = L.DomUtil.create("div", "pick");
    const value = L.DomUtil.create("div", "value", div);
    value.textContent = `${p.title} = ${p.value}`;
    L.DomUtil.create("div", "", div).textContent = p.position;
    if (p.at) {
      L.DomUtil.create("div", "", div).textContent = p.at;
    }
    L.DomUtil.create("div", "cell", div).textContent = p.cell;
    if (p.export) {
      const button = L.DomUtil.create("button", "export", div);
      button.type = "button";
      button.textContent = p.export;
      L.DomEvent.on(button, "click", (e) => {
        L.DomEvent.stop(e);
        bridge.exportPoint();
      });
    }
    L.DomEvent.disableClickPropagation(div);
    return div;
  }

  function pick(latlng) {
    if (!bridge) {
      return;
    }
    bridge.pick(latlng.lat, latlng.lng, (json) => {
      const p = JSON.parse(json);
      if (!p) {
        closePopup();
        return;
      }
      // At the cell's centre, on the copy of the world that was clicked.
      const lng = p.lon + 360 * Math.round((latlng.lng - p.lon) / 360);
      popup.setLatLng([p.lat, lng]).setContent(popupContent(p));
      if (!map.hasLayer(popup)) {
        popup.openOn(map);
      }
    });
  }

  function closePopup() {
    map.closePopup(popup);
  }

  // New data (e.g. another time step): show the value at the same place.
  function refreshPopup() {
    if (map.hasLayer(popup)) {
      pick(popup.getLatLng());
    }
  }

  map.on("click", (e) => pick(e.latlng));

  // Tell Python what's visible, so it can load that part in more detail.
  map.on("move", updateOverview);
  map.on("moveend", () => {
    updateOverview();
    if (bridge) {
      const b = map.getBounds();
      bridge.viewChanged(b.getSouth(), b.getWest(), b.getNorth(), b.getEast());
    }
  });

  new QWebChannel(qt.webChannelTransport, (channel) => {
    bridge = channel.objects.bridge;
    bridge.ready();
    const b = map.getBounds();
    bridge.viewChanged(b.getSouth(), b.getWest(), b.getNorth(), b.getEast());
  });

  return { setData, setDetail, setGridLines, setOpacity, setOutlines, closePopup, clear };
})();
