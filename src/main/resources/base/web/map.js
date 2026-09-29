// Web map for netseedf. Python drives it through the `netseedf` object below
// and answers hover and click queries through the QWebChannel object `bridge`.
"use strict";

const netseedf = (() => {
  const osmAttribution =
    '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors';
  const basemaps = {
    "OpenStreetMap": L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19, attribution: osmAttribution,
    }),
    "CARTO Positron": L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png", {
      subdomains: "abcd", maxZoom: 20,
      attribution: osmAttribution + ' &copy; <a href="https://carto.com/attributions">CARTO</a>',
    }),
    "Esri World Imagery": L.tileLayer(
      "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", {
        maxZoom: 19,
        attribution: "Tiles &copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics, " +
          "and the GIS User Community",
      }),
    "OpenTopoMap": L.tileLayer("https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png", {
      maxZoom: 17,
      attribution: osmAttribution + ', SRTM | Style: &copy; <a href="https://opentopomap.org">' +
        "OpenTopoMap</a> (CC-BY-SA)",
    }),
    "None (offline)": L.layerGroup(),
  };
  const coastlines = L.geoJSON(null, {
    style: { color: "#333", weight: 1, fill: false },
    interactive: false,
  });

  const map = L.map("map", { worldCopyJump: true, preferCanvas: true, zoomSnap: 0.25 })
    .setView([20, 0], 2);
  basemaps["OpenStreetMap"].addTo(map);
  L.control.layers(basemaps, { "Coastlines (offline)": coastlines }).addTo(map);
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

  function makeLayer(payload) {
    if (payload.kind === "image") {
      return L.imageOverlay(payload.url, L.latLngBounds(payload.bounds), {
        opacity, className: "data-overlay", interactive: false,
      });
    }
    if (payload.kind === "points") {
      return L.layerGroup(payload.points.map(([lat, lon, color]) => L.circleMarker([lat, lon], {
        radius: 5, weight: 0.6, color: "#222", opacity,
        fillColor: color || "#000", fillOpacity: color ? opacity : 0, missing: !color,
      })));
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
    const covered = detailLayer && detailCovers && detailCovers.contains(map.getBounds());
    layerOpacity(dataLayer, covered ? 0 : opacity);
    layerOpacity(detailLayer, covered ? opacity : 0);
  }

  function setData(payload) {
    for (const layer of [dataLayer, detailLayer]) {
      if (layer) {
        map.removeLayer(layer);
      }
    }
    detailLayer = detailCovers = null;
    dataLayer = makeLayer(payload);
    if (dataLayer) {
      dataLayer.addTo(map);
    }
    let bounds = null;
    if (payload.kind === "image") {
      bounds = L.latLngBounds(payload.bounds);
    } else if (payload.kind === "points" && payload.points.length) {
      bounds = L.latLngBounds(payload.points.map(([lat, lon]) => [lat, lon]));
    }
    if (payload.fit && bounds) {
      map.fitBounds(bounds, { padding: [20, 20], maxZoom: 12 });
    }
    setLegend(payload.legend);
    refreshPopup();
  }

  function setDetail(payload) {
    if (detailLayer) {
      map.removeLayer(detailLayer);
    }
    detailLayer = payload ? makeLayer(payload) : null;
    detailCovers = payload ? L.latLngBounds(payload.covers) : null;
    if (detailLayer) {
      detailLayer.addTo(map);
    }
    updateOverview();
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

  function setCoastlines(geojson) {
    coastlines.clearLayers();
    coastlines.addData(geojson);
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
      popup.setLatLng(latlng).setContent(popupContent(p));
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

  return { setData, setDetail, setGridLines, setOpacity, setCoastlines, closePopup, clear };
})();
