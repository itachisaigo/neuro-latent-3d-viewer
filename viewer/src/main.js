import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import "./styles.css";

const canvas = document.querySelector("#scene");
const statusCount = document.querySelector("#status-count");
const statusSelection = document.querySelector("#status-selection");
const details = document.querySelector("#details");
const sourcebar = document.querySelector("#sourcebar");
const datasetSelect = document.querySelector("#dataset-select");
const methodGroup = document.querySelector("#method-group");
const fitButton = document.querySelector("#fit-view");
const cameraModeButton = document.querySelector("#camera-mode");
const imageButton = document.querySelector("#toggle-images");
const pointButton = document.querySelector("#toggle-points");
const scaleInput = document.querySelector("#sprite-scale");
const jumpInput = document.querySelector("#jump-index");
let methodButtons = [...document.querySelectorAll("[data-method]")];

const VIEWER_DATA_ROOT = new URL("viewer-data/", document.baseURI);
const MANIFEST_URL = new URL("datasets.json", VIEWER_DATA_ROOT);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d1117);

const renderer = new THREE.WebGLRenderer({
  canvas,
  antialias: true,
  alpha: false,
  powerPreference: "high-performance",
});
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;

const perspectiveCamera = new THREE.PerspectiveCamera(45, 1, 0.01, 1000);
const orthographicCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, -1000, 1000);
let camera = perspectiveCamera;
let controls;
let useOrthographic = false;

const root = new THREE.Group();
scene.add(root);

const spriteGroup = new THREE.Group();
const helperGroup = new THREE.Group();
root.add(spriteGroup);
scene.add(helperGroup);

const raycaster = new THREE.Raycaster();
raycaster.params.Points.threshold = 0.055;
const pointer = new THREE.Vector2();
const bounds = new THREE.Box3();
const center = new THREE.Vector3();
const size = new THREE.Vector3();
const selectedPosition = new THREE.Vector3();

let dataset;
let points;
let activeMethod = "pca";
let activeCoordinates = [];
let selectedIndex = -1;
let spriteScale = Number(scaleInput.value);
let imagesVisible = true;
let pointsVisible = true;
let boundingRadius = 4;
let manifest;
let datasetEntry;
let datasetBasePath = VIEWER_DATA_ROOT;
let loadedTextures = [];
let pointColorLookup = new Map();

const pc1Color = new THREE.Color(0x38bdf8);
const pc2Color = new THREE.Color(0xa3e635);
const pc3Color = new THREE.Color(0xf97316);
const fallbackCategoryColors = ["#38bdf8", "#f97316", "#a3e635", "#e879f9", "#facc15", "#14b8a6", "#f43f5e", "#8b5cf6"];
const pointTexture = makePointTexture();

init();

async function init() {
  setupLights();
  setupControls();
  setupEvents();

  manifest = await loadManifest();
  populateDatasetSelect();
  const initialId = manifest.defaultDataset || manifest.datasets[0]?.id;
  await loadDataset(initialId, "iso");
  animate();
}

async function loadManifest() {
  const response = await fetch(MANIFEST_URL);
  if (!response.ok) {
    throw new Error(`Failed to load dataset manifest: ${response.status}`);
  }
  return response.json();
}

function populateDatasetSelect() {
  datasetSelect.innerHTML = "";
  for (const entry of manifest.datasets) {
    const option = document.createElement("option");
    option.value = entry.id;
    option.textContent = entry.shortLabel || entry.label || entry.id;
    datasetSelect.append(option);
  }
}

async function loadDataset(datasetId, preferredView) {
  const entry = manifest.datasets.find((candidate) => candidate.id === datasetId) || manifest.datasets[0];
  if (!entry) {
    throw new Error("No datasets are configured.");
  }

  datasetEntry = entry;
  datasetSelect.value = entry.id;
  statusCount.textContent = `Loading ${entry.label || entry.id}`;
  statusSelection.textContent = "No selection";
  details.innerHTML = `<div class="details-empty">Loading</div>`;

  const datasetUrl = new URL(entry.href, VIEWER_DATA_ROOT);
  datasetBasePath = new URL(".", datasetUrl).href;
  const response = await fetch(datasetUrl);
  if (!response.ok) {
    throw new Error(`Failed to load dataset ${entry.id}: ${response.status}`);
  }

  dataset = await response.json();
  normalizeDataset();
  applyDatasetDisplayPrefs();
  activeMethod = dataset.defaultEmbedding && dataset.embeddings[dataset.defaultEmbedding] ? dataset.defaultEmbedding : Object.keys(dataset.embeddings)[0];
  activeCoordinates = getCoordinates(activeMethod);
  selectedIndex = -1;
  jumpInput.value = "";
  jumpInput.max = String(dataset.items.length);
  jumpInput.placeholder = `1-${dataset.items.length}`;

  buildScene(dataset.items);
  renderMethodButtons();
  renderSourceBar();
  updateStatusCount();
  updateMethodButtons();
  details.innerHTML = `<div class="details-empty">No selection</div>`;
  fitView(preferredView);
}

function normalizeDataset() {
  if (!dataset.items) {
    dataset.items = [];
  }
  if (!dataset.embeddings) {
    dataset.embeddings = {
      pca: {
        label: "PCA",
        axes: ["PC1", "PC2", "PC3"],
        coordinates: dataset.items.map((item) => [item.pc1, item.pc2, item.pc3]),
      },
    };
  }
}

function applyDatasetDisplayPrefs() {
  const display = dataset.display || {};
  imagesVisible = display.mediaSprites === false ? false : display.mediaSprites ?? true;
  pointsVisible = display.points ?? true;
  imageButton.classList.toggle("is-active", imagesVisible);
  pointButton.classList.toggle("is-active", pointsVisible);
}

function setupLights() {
  scene.add(new THREE.AmbientLight(0xffffff, 1.9));
  const key = new THREE.DirectionalLight(0xffffff, 1.4);
  key.position.set(4, 6, 8);
  scene.add(key);
}

function setupControls() {
  if (controls) {
    controls.dispose();
  }
  controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.screenSpacePanning = true;
  controls.zoomToCursor = true;
  controls.rotateSpeed = 0.58;
  controls.panSpeed = 0.82;
  controls.zoomSpeed = 0.84;
  controls.minDistance = 0.35;
  controls.maxDistance = 80;
  controls.mouseButtons = {
    LEFT: THREE.MOUSE.ROTATE,
    MIDDLE: THREE.MOUSE.PAN,
    RIGHT: THREE.MOUSE.PAN,
  };
  controls.touches = {
    ONE: THREE.TOUCH.ROTATE,
    TWO: THREE.TOUCH.DOLLY_PAN,
  };
  controls.listenToKeyEvents(window);
}

function clearSceneData() {
  if (points) {
    points.geometry.dispose();
    points.material.dispose();
    root.remove(points);
    points = null;
  }

  for (const sprite of spriteGroup.children) {
    sprite.material.dispose();
  }
  spriteGroup.clear();
  for (const texture of loadedTextures) {
    texture.dispose();
  }
  loadedTextures = [];
  helperGroup.clear();

  const halo = scene.getObjectByName("selection-halo");
  if (halo) {
    halo.material.map?.dispose();
    halo.material.dispose();
    scene.remove(halo);
  }
}

function buildScene(items) {
  clearSceneData();
  const positions = new Float32Array(items.length * 3);
  const colors = new Float32Array(items.length * 3);

  for (let i = 0; i < items.length; i += 1) {
    const coord = activeCoordinates[i];
    positions[i * 3] = coord[0];
    positions[i * 3 + 1] = coord[1];
    positions[i * 3 + 2] = coord[2];
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  rebuildPointColorLookup();
  updatePointColors(geometry, activeCoordinates);
  points = new THREE.Points(
    geometry,
    new THREE.PointsMaterial({
      size: dataset.display?.pointSize ?? 0.038,
      map: pointTexture,
      vertexColors: true,
      transparent: true,
      opacity: dataset.display?.pointOpacity ?? 0.58,
      alphaTest: 0.08,
      sizeAttenuation: true,
      depthWrite: false,
    }),
  );
  points.userData.kind = "points";
  points.visible = pointsVisible;
  root.add(points);
  raycaster.params.Points.threshold = dataset.display?.pointPickRadius ?? 0.055;

  updateBoundsFromCoordinates(activeCoordinates);
  addHelpers();
  addSelectionHalo();
  if (shouldRenderMediaSprites()) {
    addSprites(items);
    spriteGroup.visible = imagesVisible;
  } else {
    spriteGroup.visible = false;
    updateStatusCount();
  }
}

function shouldRenderMediaSprites() {
  return dataset?.display?.mediaSprites !== false;
}

function addSprites(items) {
  const manager = new THREE.LoadingManager();
  manager.onProgress = (_url, loaded, total) => {
    statusCount.textContent = `${getDatasetLabel()} · loading media ${loaded}/${total}`;
  };
  manager.onLoad = () => {
    updateStatusCount();
  };

  const loader = new THREE.TextureLoader(manager);
  const maxAnisotropy = renderer.capabilities.getMaxAnisotropy();
  for (let i = 0; i < items.length; i += 1) {
    const item = items[i];
    const mediaPath = getItemMediaPath(item);
    const texture = mediaPath ? loader.load(resolveAssetPath(mediaPath)) : makeItemGlyphTexture(item, i);
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.anisotropy = Math.min(maxAnisotropy, 8);
    loadedTextures.push(texture);

    const sprite = new THREE.Sprite(
      new THREE.SpriteMaterial({
        map: texture,
        transparent: true,
        depthWrite: false,
      }),
    );
    const coord = activeCoordinates[i] || [0, 0, 0];
    sprite.position.set(coord[0], coord[1], coord[2]);
    sprite.scale.setScalar(spriteScale);
    sprite.userData.item = item;
    sprite.userData.index = i;
    spriteGroup.add(sprite);
  }
}

function resolveAssetPath(assetPath) {
  if (!assetPath) {
    return "";
  }
  if (/^(https?:)?\/\//.test(assetPath) || assetPath.startsWith("/")) {
    return assetPath;
  }
  return `${datasetBasePath}${assetPath}`;
}

function getItemMediaPath(item) {
  return item.thumb || item.thumbnail || item.spectrogram || item.poster || item.preview || "";
}

function getMediaKind(item) {
  if (item.mediaType) {
    return item.mediaType;
  }
  if (item.kind && item.kind !== "points") {
    return item.kind;
  }
  if (item.audio || item.audioSrc) {
    return "audio";
  }
  if (item.text || item.transcript) {
    return "text";
  }
  if (item.thumb || item.thumbnail || item.image) {
    return "image";
  }
  return "item";
}

function getDisplayIndex(item, index) {
  return item.index ?? item.id ?? index + 1;
}

function getItemLabel(item, index) {
  return item.title || item.label || item.image || item.audio || item.audioSrc || item.text || `#${getDisplayIndex(item, index)}`;
}

function makeItemGlyphTexture(item, index) {
  const glyphCanvas = document.createElement("canvas");
  glyphCanvas.width = 160;
  glyphCanvas.height = 160;
  const ctx = glyphCanvas.getContext("2d");
  const kind = getMediaKind(item);
  const label = kind === "audio" ? "Audio" : kind === "text" ? "Text" : "Item";
  ctx.fillStyle = kind === "audio" ? "#1d4ed8" : kind === "text" ? "#166534" : "#334155";
  ctx.fillRect(0, 0, glyphCanvas.width, glyphCanvas.height);
  ctx.strokeStyle = "rgba(255,255,255,0.38)";
  ctx.lineWidth = 6;
  ctx.strokeRect(10, 10, 140, 140);
  ctx.fillStyle = "#f8fafc";
  ctx.font = "700 32px system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(label, 80, 70);
  ctx.font = "600 22px system-ui, sans-serif";
  ctx.fillText(`#${getDisplayIndex(item, index)}`, 80, 108);
  const texture = new THREE.CanvasTexture(glyphCanvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

function addHelpers() {
  helperGroup.clear();
  const gridSize = Math.ceil(Math.max(size.x, size.y, size.z, 6));
  const grid = new THREE.GridHelper(gridSize, gridSize * 2, 0x385063, 0x243447);
  grid.rotation.x = Math.PI / 2;
  grid.position.z = center.z;
  grid.material.transparent = true;
  grid.material.opacity = 0.34;
  helperGroup.add(grid);

  helperGroup.add(axisLine(new THREE.Vector3(-gridSize / 2, 0, 0), new THREE.Vector3(gridSize / 2, 0, 0), pc1Color));
  helperGroup.add(axisLine(new THREE.Vector3(0, -gridSize / 2, 0), new THREE.Vector3(0, gridSize / 2, 0), pc2Color));
  helperGroup.add(axisLine(new THREE.Vector3(0, 0, -gridSize / 2), new THREE.Vector3(0, 0, gridSize / 2), pc3Color));

  const axes = dataset.embeddings[activeMethod]?.axes || ["X", "Y", "Z"];
  addAxisLabel(axes[0], new THREE.Vector3(gridSize / 2 + 0.25, 0, 0), pc1Color);
  addAxisLabel(axes[1], new THREE.Vector3(0, gridSize / 2 + 0.25, 0), pc2Color);
  addAxisLabel(axes[2], new THREE.Vector3(0, 0, gridSize / 2 + 0.25), pc3Color);
}

function addSelectionHalo() {
  if (scene.getObjectByName("selection-halo")) {
    return;
  }
  const haloTexture = makeHaloTexture();
  const halo = new THREE.Sprite(
    new THREE.SpriteMaterial({
      map: haloTexture,
      transparent: true,
      depthWrite: false,
      color: 0xffffff,
    }),
  );
  halo.name = "selection-halo";
  halo.visible = false;
  halo.scale.setScalar(spriteScale * 1.28);
  scene.add(halo);
}

function axisLine(start, end, color) {
  const geometry = new THREE.BufferGeometry().setFromPoints([start, end]);
  const material = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.82 });
  return new THREE.Line(geometry, material);
}

function addAxisLabel(text, position, color) {
  const sprite = new THREE.Sprite(
    new THREE.SpriteMaterial({
      map: makeTextTexture(text, `#${color.getHexString()}`),
      transparent: true,
      depthWrite: false,
    }),
  );
  sprite.position.copy(position);
  sprite.scale.set(0.42, 0.18, 1);
  helperGroup.add(sprite);
}

function makeTextTexture(text, color) {
  const labelCanvas = document.createElement("canvas");
  labelCanvas.width = 256;
  labelCanvas.height = 96;
  const ctx = labelCanvas.getContext("2d");
  ctx.clearRect(0, 0, labelCanvas.width, labelCanvas.height);
  ctx.font = "600 42px system-ui, sans-serif";
  ctx.fillStyle = color;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, 128, 48);
  const texture = new THREE.CanvasTexture(labelCanvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

function makeHaloTexture() {
  const haloCanvas = document.createElement("canvas");
  haloCanvas.width = 128;
  haloCanvas.height = 128;
  const ctx = haloCanvas.getContext("2d");
  ctx.clearRect(0, 0, 128, 128);
  ctx.strokeStyle = "#38bdf8";
  ctx.lineWidth = 9;
  if (dataset?.display?.selectionHalo === "circle") {
    ctx.beginPath();
    ctx.arc(64, 64, 52, 0, Math.PI * 2);
    ctx.stroke();
  } else {
    ctx.strokeRect(9, 9, 110, 110);
  }
  ctx.strokeStyle = "rgba(255,255,255,0.82)";
  ctx.lineWidth = 3;
  if (dataset?.display?.selectionHalo === "circle") {
    ctx.beginPath();
    ctx.arc(64, 64, 43, 0, Math.PI * 2);
    ctx.stroke();
  } else {
    ctx.strokeRect(17, 17, 94, 94);
  }
  const texture = new THREE.CanvasTexture(haloCanvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

function makePointTexture() {
  const pointCanvas = document.createElement("canvas");
  pointCanvas.width = 64;
  pointCanvas.height = 64;
  const ctx = pointCanvas.getContext("2d");
  const gradient = ctx.createRadialGradient(32, 32, 0, 32, 32, 30);
  gradient.addColorStop(0, "rgba(255,255,255,1)");
  gradient.addColorStop(0.62, "rgba(255,255,255,1)");
  gradient.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = gradient;
  ctx.beginPath();
  ctx.arc(32, 32, 30, 0, Math.PI * 2);
  ctx.fill();
  const texture = new THREE.CanvasTexture(pointCanvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

function setupEvents() {
  window.addEventListener("resize", resize);
  resize();

  const viewButtons = document.querySelectorAll("[data-view]");
  for (const button of viewButtons) {
    button.addEventListener("click", () => setView(button.dataset.view));
  }

  fitButton.addEventListener("click", () => fitView());

  datasetSelect.addEventListener("change", () => {
    loadDataset(datasetSelect.value);
  });

  cameraModeButton.addEventListener("click", () => {
    useOrthographic = !useOrthographic;
    switchCamera();
  });

  imageButton.addEventListener("click", () => {
    imagesVisible = !imagesVisible;
    spriteGroup.visible = imagesVisible;
    imageButton.classList.toggle("is-active", imagesVisible);
  });

  pointButton.addEventListener("click", () => {
    pointsVisible = !pointsVisible;
    if (points) {
      points.visible = pointsVisible;
    }
    pointButton.classList.toggle("is-active", pointsVisible);
  });

  scaleInput.addEventListener("input", () => {
    spriteScale = Number(scaleInput.value);
    for (const sprite of spriteGroup.children) {
      sprite.scale.setScalar(spriteScale);
    }
    const halo = scene.getObjectByName("selection-halo");
    halo.scale.setScalar(spriteScale * 1.28);
  });

  jumpInput.addEventListener("change", () => {
    const value = Number(jumpInput.value);
    if (Number.isInteger(value) && value >= 1 && value <= dataset.items.length) {
      selectIndex(value - 1, true);
    }
  });

  let downPosition = null;
  canvas.addEventListener("pointerdown", (event) => {
    downPosition = { x: event.clientX, y: event.clientY };
  });
  canvas.addEventListener("pointerup", (event) => {
    if (!downPosition) {
      return;
    }
    const moved = Math.hypot(event.clientX - downPosition.x, event.clientY - downPosition.y);
    downPosition = null;
    if (moved < 4) {
      pick(event);
    }
  });
}

function resize() {
  const width = window.innerWidth;
  const height = window.innerHeight;
  renderer.setSize(width, height, false);
  perspectiveCamera.aspect = width / height;
  perspectiveCamera.updateProjectionMatrix();

  const aspect = width / height;
  const frustumSize = 7.5;
  orthographicCamera.left = (-frustumSize * aspect) / 2;
  orthographicCamera.right = (frustumSize * aspect) / 2;
  orthographicCamera.top = frustumSize / 2;
  orthographicCamera.bottom = -frustumSize / 2;
  orthographicCamera.updateProjectionMatrix();
}

function getCoordinates(method) {
  const embedding = dataset.embeddings[method] || dataset.embeddings.pca;
  return embedding.coordinates;
}

function getActiveLabel() {
  return dataset.embeddings[activeMethod]?.label || activeMethod.toUpperCase();
}

function getActiveAxes() {
  return dataset.embeddings[activeMethod]?.axes || ["X", "Y", "Z"];
}

function renderMethodButtons() {
  methodGroup.innerHTML = "";
  for (const [method, embedding] of Object.entries(dataset.embeddings)) {
    const button = document.createElement("button");
    button.className = "tool-button method-button";
    button.dataset.method = method;
    button.title = `${embedding.label || method} layout`;
    button.textContent = embedding.label || method.toUpperCase();
    button.addEventListener("click", () => setEmbedding(method));
    methodGroup.append(button);
  }
  methodButtons = [...methodGroup.querySelectorAll("[data-method]")];
}

function updateMethodButtons() {
  for (const button of methodButtons) {
    button.classList.toggle("is-active", button.dataset.method === activeMethod);
  }
}

function updateStatusCount() {
  statusCount.textContent = [getDatasetLabel(), getItemCountLabel(), getSignalLabel(), getActiveLabel()].filter(Boolean).join(" · ");
}

function renderSourceBar() {
  const info = getDatasetSourceInfo();
  const links = [];

  if (info.sourceUrl) {
    links.push(`<a href="${escapeAttribute(info.sourceUrl)}" target="_blank" rel="noreferrer">${escapeHtml(info.sourceName || "Source")}</a>`);
  } else if (info.sourceName) {
    links.push(`<span>${escapeHtml(info.sourceName)}</span>`);
  }
  if (info.doi) {
    links.push(`<a href="${escapeAttribute(getDoiUrl(info.doi))}" target="_blank" rel="noreferrer">DOI ${escapeHtml(info.doi)}</a>`);
  }
  if (info.parentDoi) {
    links.push(`<a href="${escapeAttribute(getDoiUrl(info.parentDoi))}" target="_blank" rel="noreferrer">Parent DOI ${escapeHtml(info.parentDoi)}</a>`);
  }
  if (info.license) {
    const license = escapeHtml(info.license);
    links.push(
      info.licenseUrl
        ? `<a href="${escapeAttribute(info.licenseUrl)}" target="_blank" rel="noreferrer">${license}</a>`
        : `<span>${license}</span>`,
    );
  }

  if (!links.length) {
    sourcebar.hidden = true;
    sourcebar.innerHTML = "";
    return;
  }

  sourcebar.hidden = false;
  sourcebar.innerHTML = `
    <span class="sourcebar-title">${escapeHtml(info.label)}</span>
    <span class="sourcebar-links">${links.join("<span aria-hidden=\"true\">/</span>")}</span>
  `;
}

function getDatasetSourceInfo() {
  return {
    label: datasetEntry?.shortLabel || datasetEntry?.label || dataset?.label || "Dataset",
    sourceName: datasetEntry?.sourceName || dataset?.sourceName || dataset?.source || "",
    sourceUrl: datasetEntry?.sourceUrl || dataset?.sourceUrl || "",
    doi: datasetEntry?.doi || dataset?.doi || "",
    parentDoi: datasetEntry?.parentDoi || dataset?.parentDoi || "",
    license: datasetEntry?.license || dataset?.license || "",
    licenseUrl: datasetEntry?.licenseUrl || dataset?.licenseUrl || "",
  };
}

function getDoiUrl(doi) {
  if (/^https?:\/\//.test(doi)) {
    return doi;
  }
  return `https://doi.org/${doi}`;
}

function getDatasetLabel() {
  return datasetEntry?.shortLabel || datasetEntry?.label || dataset?.label || dataset?.source || "Dataset";
}

function getItemCountLabel() {
  const count = dataset.itemCount || dataset.imageCount || dataset.audioCount || dataset.textCount || dataset.items.length;
  const modality = dataset.itemLabel || dataset.mediaLabel || inferItemPluralLabel();
  return `${count.toLocaleString()} ${modality}`;
}

function inferItemPluralLabel() {
  const firstKind = getMediaKind(dataset.items[0] || {});
  if (dataset.imageCount) {
    return "images";
  }
  if (firstKind === "audio") {
    return "clips";
  }
  if (firstKind === "text") {
    return "texts";
  }
  return "items";
}

function getSignalLabel() {
  if (dataset.signalLabel) {
    return dataset.signalLabel;
  }
  if (Number.isFinite(dataset.unitCount)) {
    return `${dataset.unitCount.toLocaleString()} units`;
  }
  if (Number.isFinite(dataset.voxelCount)) {
    return `${dataset.voxelCount.toLocaleString()} voxels`;
  }
  if (Number.isFinite(dataset.channelCount)) {
    return `${dataset.channelCount.toLocaleString()} channels`;
  }
  return datasetEntry?.signal || dataset.signal || "";
}

function updateBoundsFromCoordinates(coords) {
  bounds.makeEmpty();
  if (!coords.length) {
    center.set(0, 0, 0);
    size.set(1, 1, 1);
    boundingRadius = 2.5;
    return;
  }
  for (const coord of coords) {
    bounds.expandByPoint(new THREE.Vector3(coord[0], coord[1], coord[2]));
  }
  bounds.getCenter(center);
  bounds.getSize(size);
  boundingRadius = Math.max(size.length() * 0.5, 2.5);
}

function updatePointColors(geometry, coords) {
  const color = new THREE.Color();
  const colorAttribute = geometry.getAttribute("color");
  if (dataset?.colorBy?.field || dataset?.items?.some((item) => item.color)) {
    for (let i = 0; i < dataset.items.length; i += 1) {
      color.set(getItemPointColor(dataset.items[i], i));
      colorAttribute.setXYZ(i, color.r, color.g, color.b);
    }
    colorAttribute.needsUpdate = true;
    return;
  }

  if (!coords.length) {
    colorAttribute.needsUpdate = true;
    return;
  }
  let minZ = Infinity;
  let maxZ = -Infinity;
  for (const coord of coords) {
    minZ = Math.min(minZ, coord[2]);
    maxZ = Math.max(maxZ, coord[2]);
  }
  const zSpan = maxZ - minZ || 1;

  for (let i = 0; i < coords.length; i += 1) {
    color.setHSL(0.58 - 0.42 * ((coords[i][2] - minZ) / zSpan), 0.72, 0.56);
    colorAttribute.setXYZ(i, color.r, color.g, color.b);
  }
  colorAttribute.needsUpdate = true;
}

function rebuildPointColorLookup() {
  pointColorLookup = new Map();
  const colorBy = dataset?.colorBy;
  if (!colorBy?.field) {
    return;
  }

  const palette = colorBy.palette || {};
  for (const item of dataset.items || []) {
    const key = getItemColorKey(item);
    if (pointColorLookup.has(key)) {
      continue;
    }
    const color = palette[key] || fallbackCategoryColors[pointColorLookup.size % fallbackCategoryColors.length];
    pointColorLookup.set(key, color);
  }
}

function getItemColorKey(item) {
  const field = dataset?.colorBy?.field;
  const value = field ? item[field] : item.color;
  return value === undefined || value === null || value === "" ? "Unknown" : String(value);
}

function getItemPointColor(item, index) {
  if (item.color) {
    return item.color;
  }
  const key = getItemColorKey(item);
  return pointColorLookup.get(key) || fallbackCategoryColors[index % fallbackCategoryColors.length];
}

function setEmbedding(method) {
  if (!dataset.embeddings[method] || method === activeMethod) {
    return;
  }

  activeMethod = method;
  activeCoordinates = getCoordinates(method);

  const positionAttribute = points.geometry.getAttribute("position");
  for (let i = 0; i < activeCoordinates.length; i += 1) {
    const coord = activeCoordinates[i];
    positionAttribute.setXYZ(i, coord[0], coord[1], coord[2]);
    const sprite = spriteGroup.children[i];
    if (sprite) {
      sprite.position.set(coord[0], coord[1], coord[2]);
    }
  }
  positionAttribute.needsUpdate = true;
  points.geometry.computeBoundingSphere();
  updatePointColors(points.geometry, activeCoordinates);

  updateBoundsFromCoordinates(activeCoordinates);
  addHelpers();
  updateMethodButtons();
  updateStatusCount();
  if (selectedIndex >= 0) {
    updateSelection(selectedIndex, false);
  }
  fitView();
}

function switchCamera() {
  const oldCamera = camera;
  camera = useOrthographic ? orthographicCamera : perspectiveCamera;
  camera.position.copy(oldCamera.position);
  camera.quaternion.copy(oldCamera.quaternion);
  camera.up.copy(oldCamera.up);
  cameraModeButton.textContent = useOrthographic ? "Ortho" : "Persp";
  setupControls();
  controls.target.copy(center);
  fitView();
}

function setView(name) {
  const distance = boundingRadius * 2.35;
  const views = {
    iso: { position: [distance, distance * 0.78, distance], up: [0, 1, 0] },
    front: { position: [0, 0, distance], up: [0, 1, 0] },
    right: { position: [distance, 0, 0], up: [0, 1, 0] },
    left: { position: [-distance, 0, 0], up: [0, 1, 0] },
    top: { position: [0, distance, 0], up: [0, 0, -1] },
    bottom: { position: [0, -distance, 0], up: [0, 0, 1] },
  };
  const view = views[name] || views.iso;
  camera.up.fromArray(view.up);
  camera.position.set(...view.position).add(center);
  controls.target.copy(center);
  camera.lookAt(center);
  controls.update();
}

function fitView(preferredView) {
  if (preferredView) {
    setView(preferredView);
    return;
  }

  const direction = camera.position.clone().sub(controls.target);
  if (direction.lengthSq() < 0.001) {
    direction.set(1, 0.72, 1);
  }
  direction.normalize();
  controls.target.copy(center);

  if (camera.isPerspectiveCamera) {
    const fov = THREE.MathUtils.degToRad(camera.fov);
    const distance = (boundingRadius * 1.32) / Math.sin(fov / 2);
    camera.position.copy(center).add(direction.multiplyScalar(distance));
  } else {
    orthographicCamera.zoom = 7.5 / Math.max(boundingRadius * 2.35, 1);
    camera.position.copy(center).add(direction.multiplyScalar(boundingRadius * 3.2));
    camera.updateProjectionMatrix();
  }
  camera.lookAt(center);
  controls.update();
}

function pick(event) {
  const rect = canvas.getBoundingClientRect();
  pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);

  if (imagesVisible) {
    const spriteHits = raycaster.intersectObjects(spriteGroup.children, false);
    if (spriteHits.length) {
      selectIndex(spriteHits[0].object.userData.index, false);
      return;
    }
  }

  if (pointsVisible) {
    const pointHits = raycaster.intersectObject(points, false);
    if (pointHits.length && Number.isInteger(pointHits[0].index)) {
      selectIndex(pointHits[0].index, false);
    }
  }
}

function selectIndex(index, focus) {
  updateSelection(index, focus);
}

function updateSelection(index, focus) {
  const item = dataset.items[index];
  if (!item) {
    return;
  }

  selectedIndex = index;
  const coord = activeCoordinates[index];
  const axes = getActiveAxes();
  const displayIndex = formatDisplayIndex(getDisplayIndex(item, index));
  const title = getItemLabel(item, index);
  selectedPosition.set(coord[0], coord[1], coord[2]);
  const halo = scene.getObjectByName("selection-halo");
  halo.position.copy(selectedPosition);
  halo.visible = true;

  statusSelection.textContent = `#${displayIndex} · ${getActiveLabel()} ${coord[0].toFixed(2)}, ${coord[1].toFixed(2)}, ${coord[2].toFixed(2)}`;
  details.innerHTML = `
    <div class="selected-panel">
      ${renderSelectedMedia(item, index)}
      <div class="selected-meta">
        <p class="selected-title">${escapeHtml(title)}</p>
        ${renderSelectedSupplement(item)}
        <div class="selected-coords">
          <span>${axes[0]} ${coord[0].toFixed(2)}</span>
          <span>${axes[1]} ${coord[1].toFixed(2)}</span>
          <span>${axes[2]} ${coord[2].toFixed(2)}</span>
        </div>
      </div>
    </div>
  `;
  const selectedAudio = details.querySelector("audio[data-autoplay]");
  if (selectedAudio) {
    selectedAudio.play().catch(() => {});
  }

  if (focus) {
    controls.target.copy(selectedPosition);
    const direction = camera.position.clone().sub(center).normalize();
    camera.position.copy(selectedPosition).add(direction.multiplyScalar(Math.max(boundingRadius * 1.3, 4.5)));
    camera.lookAt(selectedPosition);
    controls.update();
  }
}

function renderSelectedMedia(item, index) {
  if (dataset?.display?.selectedMedia === "point") {
    return `<div class="selected-dot" style="--dot-color:${escapeAttribute(getItemPointColor(item, index))}"><span></span></div>`;
  }
  const mediaPath = getItemMediaPath(item);
  if (mediaPath) {
    return `<img src="${escapeAttribute(resolveAssetPath(mediaPath))}" alt="${escapeAttribute(getItemLabel(item, index))}">`;
  }
  return `<div class="selected-glyph">${escapeHtml(getMediaKind(item))}</div>`;
}

function renderSelectedSupplement(item) {
  const parts = [];
  const mediaKind = getMediaKind(item);
  parts.push(`<span>${escapeHtml(mediaKind)}</span>`);
  if (item.category) {
    parts.push(`<span>${escapeHtml(item.category)}</span>`);
  }
  if (item.stimulusType) {
    parts.push(`<span>${escapeHtml(item.stimulusType)}</span>`);
  }
  const colorField = dataset?.colorBy?.field;
  if (colorField && item[colorField]) {
    const label = dataset.colorBy.label || colorField;
    const value = String(item[colorField]);
    const tag = value.toLowerCase().startsWith(String(label).toLowerCase()) ? value : `${label} ${value}`;
    parts.push(`<span>${escapeHtml(tag)}</span>`);
  } else if (item.speaker) {
    parts.push(`<span>${escapeHtml(item.speaker)}</span>`);
  }
  if (item.license) {
    parts.push(`<span>${escapeHtml(item.license)}</span>`);
  }

  const audioPath = item.audio || item.audioSrc;
  const audio = audioPath
    ? `<audio data-autoplay controls preload="none" src="${escapeAttribute(resolveAssetPath(audioPath))}"></audio>`
    : "";
  const credit = [item.artist, item.sourceUrl].filter(Boolean).join(" · ");
  const text = item.text || item.transcript || item.caption || "";
  const textHtml = text ? `<p class="selected-text">${escapeHtml(truncateText(text, 260))}</p>` : "";
  const creditHtml = credit ? `<p class="selected-credit">${escapeHtml(truncateText(credit, 180))}</p>` : "";

  return `
    <div class="selected-tags">${parts.join("")}</div>
    ${audio}
    ${textHtml}
    ${creditHtml}
  `;
}

function formatDisplayIndex(value) {
  if (Number.isInteger(value)) {
    return String(value).padStart(4, "0");
  }
  const numeric = Number(value);
  if (Number.isInteger(numeric)) {
    return String(numeric).padStart(4, "0");
  }
  return String(value);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function escapeAttribute(value) {
  return escapeHtml(value);
}

function truncateText(value, maxLength) {
  const text = String(value).replace(/\s+/g, " ").trim();
  if (text.length <= maxLength) {
    return text;
  }
  return `${text.slice(0, Math.max(0, maxLength - 1)).trim()}...`;
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  if (selectedIndex >= 0) {
    const halo = scene.getObjectByName("selection-halo");
    halo.position.copy(selectedPosition);
  }
  renderer.render(scene, camera);
}
