"use strict";

const CAMERA_LABELS = {
  cam_high: "HIGH / REVIEW VIEW",
  cam_left_wrist: "LEFT WRIST / MODEL VIEW",
  cam_right_wrist: "RIGHT WRIST / MODEL VIEW",
  head: "HEAD / MODEL VIEW",
  left_wrist: "LEFT WRIST / MODEL VIEW",
  right_wrist: "RIGHT WRIST / MODEL VIEW",
};

const FINGER_LABELS = ["Thumb", "Index", "Middle", "Ring", "Pinky"];
const CHANNEL_KEYS = ["normal", "tangential", "proximity"];
const CHANNEL_CLASSES = ["normal", "tangent", "proximity"];
const APP_BASE = new URL("./", window.location.href);

const elements = Object.fromEntries(
  [
    "episodeCount", "videoCount", "episodeSearch", "refreshButton", "episodeList",
    "previousButton", "nextButton", "activeKicker", "activeTitle", "healthBadge",
    "metadataButton", "deleteButton", "emptyState", "reviewWorkspace", "cameraGrid",
    "playButton", "currentTime", "durationTime", "timeline", "timelineBuffered",
    "speedSelect", "loopButton", "tactileStatus", "tactileDashboard",
    "metadataDialog", "metadataTitle", "metadataHighlights", "metadataJson",
    "deleteDialog", "deleteForm", "deleteEpisodeName", "deleteConfirmationHint",
    "deleteConfirmation", "cancelDeleteButton", "confirmDeleteButton", "toast",
    "toastMessage", "toastAction", "toastClose",
  ].map((id) => [id, document.getElementById(id)]),
);

const state = {
  dataset: null,
  episodes: [],
  filteredEpisodes: [],
  activeId: null,
  activeSummary: null,
  metadata: null,
  telemetry: null,
  telemetryIndex: -1,
  videos: new Map(),
  master: null,
  generation: 0,
  animationFrame: null,
  seeking: false,
  toastTimer: null,
  restoreToken: null,
  deletedId: null,
};

async function api(path, options = {}) {
  const response = await fetch(appUrl(path), {
    ...options,
    headers: {
      Accept: "application/json",
      ...(options.body ? { "Content-Type": "application/json" } : {}),
      ...(options.headers || {}),
    },
  });
  const type = response.headers.get("content-type") || "";
  const payload = type.includes("application/json") ? await response.json() : null;
  if (!response.ok) {
    const message = payload?.error || payload?.message || `${response.status} ${response.statusText}`;
    throw new Error(message);
  }
  return payload;
}

function appUrl(path) {
  if (/^https?:\/\//i.test(path)) return path;
  return new URL(String(path).replace(/^\/+/, ""), APP_BASE).toString();
}

function episodeId(episode) {
  return Number(episode.id ?? episode.episode_id ?? episode.index);
}

function episodeName(episodeOrId) {
  if (typeof episodeOrId === "object") {
    return episodeOrId.name || `episode_${episodeId(episodeOrId)}`;
  }
  return `episode_${episodeOrId}`;
}

function episodeVideos(episode) {
  if (!episode) return {};
  if (Array.isArray(episode.videos)) {
    return Object.fromEntries(episode.videos.map((camera) => [camera, true]));
  }
  return episode.videos || episode.video_available || {};
}

function frameCount(episode) {
  return Number(episode?.frame_count ?? episode?.frames ?? 0);
}

function episodeFps(episode) {
  return Number(episode?.fps ?? 30);
}

function durationSeconds(episode) {
  const explicit = Number(episode?.duration_s ?? episode?.duration_seconds);
  if (Number.isFinite(explicit) && explicit >= 0) return explicit;
  const frames = frameCount(episode);
  const fps = episodeFps(episode);
  return fps > 0 ? frames / fps : 0;
}

function availableCameraCount(episode) {
  const videos = episodeVideos(episode);
  return Object.values(videos).filter((value) => value === true || typeof value === "string").length;
}

function cameraNames() {
  const declared = state.dataset?.cameras || state.dataset?.camera_names;
  if (Array.isArray(declared) && declared.length) return declared;
  const fromEpisode = Object.keys(episodeVideos(state.activeSummary));
  return fromEpisode.length ? fromEpisode : ["cam_high", "cam_left_wrist", "cam_right_wrist"];
}

function formatDuration(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return "--:--.-";
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds - minutes * 60;
  return `${String(minutes).padStart(2, "0")}:${remainder.toFixed(1).padStart(4, "0")}`;
}

function formatBytes(bytes) {
  const value = Number(bytes);
  if (!Number.isFinite(value) || value < 0) return "—";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let size = value;
  let unit = 0;
  while (size >= 1000 && unit < units.length - 1) {
    size /= 1000;
    unit += 1;
  }
  return `${size >= 10 || unit === 0 ? size.toFixed(0) : size.toFixed(1)} ${units[unit]}`;
}

function setHealth(kind, message) {
  elements.healthBadge.className = `health-badge is-${kind}`;
  elements.healthBadge.lastElementChild.textContent = message;
}

function totalVideoCount(payload) {
  const explicit = Number(payload?.video_count ?? payload?.total_videos);
  if (Number.isFinite(explicit)) return explicit;
  return (payload?.episodes || []).reduce((sum, episode) => sum + availableCameraCount(episode), 0);
}

async function loadDataset(refresh = false, preferredId = null) {
  elements.refreshButton.disabled = true;
  setHealth("loading", refresh ? "重新扫描中" : "读取数据集");
  try {
    const payload = await api(`/api/dataset${refresh ? "?refresh=1" : ""}`);
    state.dataset = payload;
    state.episodes = [...(payload.episodes || [])].sort((a, b) => episodeId(a) - episodeId(b));
    elements.episodeCount.textContent = String(payload.count ?? state.episodes.length);
    elements.videoCount.textContent = String(totalVideoCount(payload));
    const root = payload.root || payload.dataset_root || "";
    elements.episodeCount.closest(".dataset-summary").title = root;
    applyFilter();

    const target = preferredId ?? state.activeId;
    if (target !== null && state.episodes.some((episode) => episodeId(episode) === target)) {
      await selectEpisode(target, { force: refresh });
    } else if (state.activeId !== null) {
      clearSelection();
    } else {
      setHealth("ready", `${state.episodes.length} 条已就绪`);
    }
  } catch (error) {
    setHealth("error", "数据集读取失败");
    showToast(`读取数据集失败：${error.message}`);
  } finally {
    elements.refreshButton.disabled = false;
  }
}

function applyFilter() {
  const query = elements.episodeSearch.value.trim().toLowerCase();
  state.filteredEpisodes = state.episodes.filter((episode) => {
    if (!query) return true;
    const haystack = [
      episodeName(episode),
      String(episodeId(episode)),
      episode.task,
      episode.collect_time_utc,
    ].filter(Boolean).join(" ").toLowerCase();
    return haystack.includes(query);
  });
  renderEpisodeList();
}

function renderEpisodeList() {
  const fragment = document.createDocumentFragment();
  if (!state.filteredEpisodes.length) {
    const empty = document.createElement("p");
    empty.className = "list-empty";
    empty.textContent = state.episodes.length ? "没有匹配的条目" : "数据集中没有可用 episode";
    fragment.append(empty);
  }

  const expectedCameras = cameraNames().length;
  for (const episode of state.filteredEpisodes) {
    const id = episodeId(episode);
    const cameras = availableCameraCount(episode);
    const button = document.createElement("button");
    button.type = "button";
    button.className = "episode-item";
    button.dataset.id = String(id);
    button.setAttribute("role", "option");
    button.setAttribute("aria-selected", String(id === state.activeId));
    button.innerHTML = `
      <span class="episode-number">${String(id).padStart(3, "0")}</span>
      <span class="episode-copy">
        <strong>${escapeHtml(episodeName(episode))}</strong>
        <span>${frameCount(episode).toLocaleString()} frames · ${formatDuration(durationSeconds(episode))}</span>
      </span>
      <span class="episode-trailing">
        <i class="availability-dot ${cameras < expectedCameras ? "is-incomplete" : ""}"></i>
        ${cameras}/${expectedCameras}
      </span>`;
    button.addEventListener("click", () => selectEpisode(id));
    fragment.append(button);
  }
  elements.episodeList.replaceChildren(fragment);
}

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
  })[character]);
}

function renderCameraGrid() {
  stopPlayback();
  for (const video of state.videos.values()) {
    video.removeAttribute("src");
    video.load();
  }
  state.videos.clear();
  state.master = null;
  const fragment = document.createDocumentFragment();
  const videos = episodeVideos(state.activeSummary);
  const cameras = cameraNames().filter((camera) => videos[camera] !== false && videos[camera] !== null);
  cameras.forEach((camera, index) => {
    const card = document.createElement("article");
    card.className = "camera-card";
    const label = document.createElement("span");
    label.className = "camera-label";
    label.textContent = CAMERA_LABELS[camera] || camera.replaceAll("_", " ").toUpperCase();
    const video = document.createElement("video");
    video.preload = index === 0 ? "auto" : "metadata";
    video.muted = true;
    video.playsInline = true;
    video.disablePictureInPicture = false;
    video.src = appUrl(
      typeof videos[camera] === "string"
        ? videos[camera]
        : `media/${state.activeId}/${encodeURIComponent(camera)}`,
    );
    video.dataset.camera = camera;
    const status = document.createElement("span");
    status.className = "camera-state";
    status.textContent = "载入中";
    const error = document.createElement("div");
    error.className = "camera-error";
    error.textContent = "浏览器无法解码该 AV1 视频。请使用新版 Chrome/Edge，或在服务器端生成 H.264 代理。";
    video.addEventListener("loadedmetadata", () => {
      status.textContent = `${video.videoWidth}×${video.videoHeight} · ${episodeFps(state.activeSummary)} FPS`;
      if (video === state.master) setupMasterDuration();
    });
    video.addEventListener("canplay", () => card.classList.remove("has-error"));
    video.addEventListener("error", () => {
      card.classList.add("has-error");
      label.classList.add("is-error");
      status.textContent = "解码失败";
      setHealth("error", "视频解码失败");
    });
    card.append(video, label, status, error);
    fragment.append(card);
    state.videos.set(camera, video);
    if (!state.master || camera === "cam_high" || camera === "head") state.master = video;
  });
  elements.cameraGrid.replaceChildren(fragment);
  if (!state.master && state.videos.size) state.master = state.videos.values().next().value;
  bindMasterEvents();
}

function bindMasterEvents() {
  const master = state.master;
  if (!master) return;
  master.addEventListener("timeupdate", updatePlaybackUi);
  master.addEventListener("progress", updateBufferedUi);
  master.addEventListener("ended", () => {
    if (elements.loopButton.getAttribute("aria-pressed") === "true") {
      seekAll(0);
      playAll();
    } else {
      stopPlayback();
    }
  });
  master.addEventListener("waiting", () => setHealth("loading", "视频缓冲中"));
  master.addEventListener("playing", () => setHealth("ready", "三路同步播放"));
}

function setupMasterDuration() {
  if (!state.master) return;
  const duration = Number.isFinite(state.master.duration)
    ? state.master.duration
    : durationSeconds(state.activeSummary);
  elements.timeline.max = String(Math.max(duration, 0.001));
  elements.durationTime.textContent = formatDuration(duration);
  updatePlaybackUi();
}

async function selectEpisode(id, { force = false } = {}) {
  id = Number(id);
  if (!force && state.activeId === id) return;
  const summary = state.episodes.find((episode) => episodeId(episode) === id);
  if (!summary) return;
  const generation = ++state.generation;
  stopPlayback();
  state.activeId = id;
  state.activeSummary = summary;
  state.metadata = null;
  state.telemetry = null;
  state.telemetryIndex = -1;
  elements.emptyState.hidden = true;
  elements.reviewWorkspace.hidden = false;
  elements.activeKicker.textContent = `${frameCount(summary).toLocaleString()} FRAMES · ${episodeFps(summary)} FPS · ${availableCameraCount(summary)} CAMERAS`;
  elements.activeTitle.textContent = episodeName(summary);
  elements.metadataButton.disabled = true;
  elements.deleteButton.disabled = false;
  elements.previousButton.disabled = false;
  elements.nextButton.disabled = false;
  elements.tactileStatus.textContent = "读取并按 10 Hz 对齐触觉…";
  setHealth("loading", "载入 episode");
  renderEpisodeList();
  const active = elements.episodeList.querySelector(`[data-id="${id}"]`);
  active?.scrollIntoView({ block: "nearest" });
  renderCameraGrid();
  resetTactileDashboard();

  const detailsPromise = api(`/api/episodes/${id}`)
    .then((payload) => {
      if (generation !== state.generation) return;
      state.metadata = payload;
      elements.metadataButton.disabled = false;
      updateHeaderFromMetadata(payload);
    });
  const telemetryPromise = api(`/api/episodes/${id}/telemetry?fps=10`)
    .then((payload) => {
      if (generation !== state.generation) return;
      state.telemetry = normalizeTelemetry(payload);
      state.telemetryIndex = -1;
      const samples = state.telemetry.times.length;
      elements.tactileStatus.textContent = `${samples.toLocaleString()} 个显示采样 · latest-nonfuture 对齐`;
      updateTactile(0, true);
    });
  const results = await Promise.allSettled([detailsPromise, telemetryPromise]);
  if (generation !== state.generation) return;
  const failures = results.filter((result) => result.status === "rejected");
  if (failures.length) {
    const message = failures.map((result) => result.reason.message).join("；");
    elements.tactileStatus.textContent = `部分数据不可用：${message}`;
    showToast(`载入 ${episodeName(id)} 时出现问题：${message}`);
  }
  setHealth(failures.length === results.length ? "error" : "ready", failures.length ? "部分数据可用" : "同步数据已就绪");
}

function updateHeaderFromMetadata(payload) {
  const attrs = payload.attrs || payload.attributes || {};
  const task = attrs.task ?? payload.task ?? state.activeSummary?.task;
  if (task) {
    const cleaned = String(task).replace(/^['"]|['"]$/g, "");
    elements.activeTitle.textContent = `${episodeName(state.activeId)} · ${cleaned}`;
    const root = String(state.dataset?.root || state.dataset?.dataset_root || "").toLowerCase();
    if (root.includes("poker") && /cup/i.test(cleaned)) {
      elements.activeKicker.textContent += " · TASK META ⚠";
      elements.activeKicker.title = "目录名称是 poker_card，但 HDF5 task 字段写的是 pick up the cup";
    } else {
      elements.activeKicker.title = "";
    }
  }
}

function clearSelection() {
  ++state.generation;
  stopPlayback();
  for (const video of state.videos.values()) {
    video.removeAttribute("src");
    video.load();
  }
  state.activeId = null;
  state.activeSummary = null;
  state.metadata = null;
  state.telemetry = null;
  state.videos.clear();
  state.master = null;
  elements.activeKicker.textContent = "SELECT AN EPISODE";
  elements.activeTitle.textContent = "请选择一条数据";
  elements.metadataButton.disabled = true;
  elements.deleteButton.disabled = true;
  elements.previousButton.disabled = true;
  elements.nextButton.disabled = true;
  elements.emptyState.hidden = false;
  elements.reviewWorkspace.hidden = true;
  renderEpisodeList();
  setHealth("ready", `${state.episodes.length} 条已就绪`);
}

function resetTactileDashboard() {
  const fragment = document.createDocumentFragment();
  [["LEFT", "左手"], ["RIGHT", "右手"]].forEach(([hand, zh]) => {
    const handLabel = document.createElement("div");
    handLabel.className = "hand-label";
    handLabel.innerHTML = `<strong>${hand}</strong><span>${zh}</span>`;
    fragment.append(handLabel);
    FINGER_LABELS.forEach((finger, fingerIndex) => {
      const card = document.createElement("div");
      card.className = "finger-card";
      card.dataset.hand = hand.toLowerCase();
      card.dataset.finger = String(fingerIndex);
      card.innerHTML = `
        <span>${finger}</span>
        <div class="finger-bars">
          ${CHANNEL_CLASSES.map((channel) => `<i class="sensor-bar ${channel}" style="--level:0"></i>`).join("")}
        </div>
        <div class="finger-values"><b>0</b><b>0</b><b>0</b></div>`;
      fragment.append(card);
    });
  });
  elements.tactileDashboard.replaceChildren(fragment);
}

function normalizeTelemetry(payload) {
  const samples = payload.samples || {};
  const times = payload.times || payload.time_s || samples.times || [];
  const channels = payload.channels || payload.channel_names || CHANNEL_KEYS;
  const rawScales = payload.scales || payload.channel_scales || {};
  const scales = CHANNEL_KEYS.map((key, index) => {
    if (Array.isArray(rawScales)) return Number(rawScales[index]) || 1;
    return Number(rawScales[key] ?? rawScales[channels[index]]) || 1;
  });
  return {
    times: Array.from(times, Number),
    left: payload.left || samples.left || [],
    right: payload.right || samples.right || [],
    scales,
    channels,
  };
}

function sampleIndexAt(time) {
  const times = state.telemetry?.times || [];
  if (!times.length) return -1;
  let low = 0;
  let high = times.length;
  while (low < high) {
    const middle = (low + high) >> 1;
    if (times[middle] <= time + 1e-7) low = middle + 1;
    else high = middle;
  }
  return Math.max(0, low - 1);
}

function updateTactile(time, force = false) {
  if (!state.telemetry) return;
  const index = sampleIndexAt(time);
  if (index < 0 || (!force && index === state.telemetryIndex)) return;
  state.telemetryIndex = index;
  ["left", "right"].forEach((hand) => {
    const handSample = state.telemetry[hand]?.[index];
    if (!handSample) return;
    for (let finger = 0; finger < 5; finger += 1) {
      const card = elements.tactileDashboard.querySelector(`[data-hand="${hand}"][data-finger="${finger}"]`);
      const values = handSample[finger] || [0, 0, 0];
      card?.querySelectorAll(".sensor-bar").forEach((bar, channel) => {
        const value = Math.max(0, Number(values[channel]) || 0);
        const level = Math.min(1, value / state.telemetry.scales[channel]);
        bar.style.setProperty("--level", level.toFixed(4));
      });
      card?.querySelectorAll(".finger-values b").forEach((label, channel) => {
        label.textContent = compactNumber(values[channel]);
        label.title = String(values[channel] ?? 0);
      });
    }
  });
}

function compactNumber(value) {
  const number = Number(value) || 0;
  const absolute = Math.abs(number);
  if (absolute >= 1e6) return `${(number / 1e6).toFixed(1)}m`;
  if (absolute >= 1e3) return `${(number / 1e3).toFixed(1)}k`;
  if (absolute >= 10) return number.toFixed(0);
  if (absolute >= 1) return number.toFixed(1);
  return number.toFixed(2);
}

async function playAll() {
  if (!state.master) return;
  if (state.master.ended || state.master.currentTime >= state.master.duration - 0.02) seekAll(0);
  const target = state.master.currentTime;
  for (const video of state.videos.values()) {
    if (Math.abs(video.currentTime - target) > 0.03) video.currentTime = target;
    video.playbackRate = Number(elements.speedSelect.value);
  }
  const outcomes = await Promise.allSettled([...state.videos.values()].map((video) => video.play()));
  const rejected = outcomes.filter((outcome) => outcome.status === "rejected");
  if (rejected.length === outcomes.length) {
    showToast(`无法开始播放：${rejected[0].reason?.message || "浏览器拒绝播放"}`);
    return;
  }
  elements.playButton.classList.add("is-playing");
  elements.playButton.setAttribute("aria-label", "暂停");
  syncLoop();
}

function stopPlayback() {
  for (const video of state.videos.values()) video.pause();
  elements.playButton.classList.remove("is-playing");
  elements.playButton.setAttribute("aria-label", "播放");
  if (state.animationFrame !== null) cancelAnimationFrame(state.animationFrame);
  state.animationFrame = null;
}

function syncLoop() {
  if (!state.master || state.master.paused) {
    state.animationFrame = null;
    return;
  }
  const time = state.master.currentTime;
  for (const video of state.videos.values()) {
    if (video === state.master || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) continue;
    if (Math.abs(video.currentTime - time) > 0.09) video.currentTime = time;
  }
  updatePlaybackUi();
  state.animationFrame = requestAnimationFrame(syncLoop);
}

function seekAll(time) {
  const duration = Number(elements.timeline.max) || durationSeconds(state.activeSummary);
  const target = Math.max(0, Math.min(Number(time) || 0, duration));
  for (const video of state.videos.values()) {
    try { video.currentTime = target; } catch (_) { /* metadata is still loading */ }
  }
  elements.timeline.value = String(target);
  updatePlaybackUi(target);
}

function updatePlaybackUi(explicitTime = null) {
  const time = explicitTime ?? state.master?.currentTime ?? 0;
  const duration = Number(elements.timeline.max) || state.master?.duration || 0;
  if (!state.seeking) elements.timeline.value = String(time);
  elements.timeline.style.setProperty("--progress", `${duration ? (time / duration) * 100 : 0}%`);
  elements.currentTime.textContent = formatDuration(time);
  updateTactile(time);
}

function updateBufferedUi() {
  if (!state.master || !state.master.buffered.length) return;
  const duration = state.master.duration || 0;
  const end = state.master.buffered.end(state.master.buffered.length - 1);
  elements.timelineBuffered.style.width = `${duration ? (end / duration) * 100 : 0}%`;
}

function moveEpisode(delta) {
  if (!state.episodes.length) return;
  const index = state.episodes.findIndex((episode) => episodeId(episode) === state.activeId);
  const nextIndex = index < 0 ? 0 : Math.max(0, Math.min(state.episodes.length - 1, index + delta));
  selectEpisode(episodeId(state.episodes[nextIndex]));
}

function openMetadata() {
  if (!state.metadata) return;
  const summary = state.activeSummary;
  const attrs = state.metadata.attrs || state.metadata.attributes || {};
  const videos = episodeVideos(summary);
  const totalBytes = Number(summary?.size_bytes ?? summary?.total_size_bytes);
  const highlights = [
    ["Frames", frameCount(summary).toLocaleString()],
    ["FPS", String(episodeFps(summary))],
    ["Duration", formatDuration(durationSeconds(summary))],
    ["Total size", formatBytes(totalBytes)],
    ["Task (HDF5)", String(attrs.task ?? summary?.task ?? "—").replace(/^['"]|['"]$/g, "")],
    ["Cameras", `${availableCameraCount(summary)} / ${Object.keys(videos).length || cameraNames().length}`],
    ["Collected", String(attrs.collect_time_utc ?? summary?.collect_time_utc ?? "—").replace(/^['"]|['"]$/g, "")],
    ["Schema", String(attrs.schema_version ?? state.metadata.schema_version ?? "—")],
  ];
  elements.metadataTitle.textContent = episodeName(state.activeId);
  elements.metadataHighlights.innerHTML = highlights.map(([label, value]) => `
    <div class="metadata-highlight"><span>${escapeHtml(label)}</span><strong title="${escapeHtml(value)}">${escapeHtml(value)}</strong></div>`).join("");
  elements.metadataJson.textContent = JSON.stringify(state.metadata, null, 2);
  elements.metadataDialog.showModal();
}

function openDeleteDialog() {
  if (state.activeId === null) return;
  const name = episodeName(state.activeId);
  elements.deleteEpisodeName.textContent = name;
  elements.deleteConfirmationHint.textContent = name;
  elements.deleteConfirmation.value = "";
  elements.confirmDeleteButton.disabled = true;
  elements.deleteDialog.showModal();
  requestAnimationFrame(() => elements.deleteConfirmation.focus());
}

async function deleteActiveEpisode(event) {
  event.preventDefault();
  if (state.activeId === null) return;
  const id = state.activeId;
  const name = episodeName(id);
  if (elements.deleteConfirmation.value.trim() !== name) return;
  elements.confirmDeleteButton.disabled = true;
  elements.cancelDeleteButton.disabled = true;
  elements.confirmDeleteButton.textContent = "正在移动…";
  try {
    stopPlayback();
    const payload = await api(`/api/episodes/${id}`, {
      method: "DELETE",
      body: JSON.stringify({ confirm: name }),
    });
    state.restoreToken = payload.trash_token || payload.token || payload.trash_id;
    state.deletedId = id;
    elements.deleteDialog.close();
    const remaining = state.episodes.filter((episode) => episodeId(episode) !== id);
    const next = remaining.find((episode) => episodeId(episode) > id) || remaining.at(-1);
    clearSelection();
    await loadDataset(true, next ? episodeId(next) : null);
    showToast(`${name} 已移到回收站`, state.restoreToken ? "撤销" : null, restoreLastDeletion);
  } catch (error) {
    showToast(`删除失败：${error.message}`);
  } finally {
    elements.confirmDeleteButton.textContent = "移到回收站";
    elements.cancelDeleteButton.disabled = false;
    elements.confirmDeleteButton.disabled = elements.deleteConfirmation.value.trim() !== name;
  }
}

async function restoreLastDeletion() {
  if (!state.restoreToken) return;
  const token = state.restoreToken;
  const id = state.deletedId;
  elements.toastAction.disabled = true;
  try {
    await api(`/api/trash/${encodeURIComponent(token)}/restore`, { method: "POST" });
    state.restoreToken = null;
    state.deletedId = null;
    hideToast();
    await loadDataset(true, id);
    showToast(`${episodeName(id)} 已恢复`);
  } catch (error) {
    elements.toastAction.disabled = false;
    showToast(`恢复失败：${error.message}`, "重试", restoreLastDeletion);
  }
}

function showToast(message, actionLabel = null, action = null) {
  window.clearTimeout(state.toastTimer);
  elements.toastMessage.textContent = message;
  elements.toastAction.hidden = !actionLabel;
  elements.toastAction.textContent = actionLabel || "";
  elements.toastAction.disabled = false;
  elements.toastAction.onclick = action;
  elements.toast.hidden = false;
  state.toastTimer = window.setTimeout(hideToast, actionLabel ? 12000 : 5000);
}

function hideToast() {
  window.clearTimeout(state.toastTimer);
  elements.toast.hidden = true;
}

function isEditingTarget(target) {
  return target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement || target instanceof HTMLSelectElement || target.isContentEditable;
}

elements.episodeSearch.addEventListener("input", applyFilter);
elements.refreshButton.addEventListener("click", () => loadDataset(true));
elements.previousButton.addEventListener("click", () => moveEpisode(-1));
elements.nextButton.addEventListener("click", () => moveEpisode(1));
elements.metadataButton.addEventListener("click", openMetadata);
elements.deleteButton.addEventListener("click", openDeleteDialog);
elements.playButton.addEventListener("click", () => state.master?.paused ? playAll() : stopPlayback());
elements.speedSelect.addEventListener("change", () => {
  for (const video of state.videos.values()) video.playbackRate = Number(elements.speedSelect.value);
});
elements.loopButton.addEventListener("click", () => {
  const enabled = elements.loopButton.getAttribute("aria-pressed") !== "true";
  elements.loopButton.setAttribute("aria-pressed", String(enabled));
});
elements.timeline.addEventListener("pointerdown", () => { state.seeking = true; });
elements.timeline.addEventListener("input", () => {
  const time = Number(elements.timeline.value);
  elements.currentTime.textContent = formatDuration(time);
  elements.timeline.style.setProperty("--progress", `${(time / Number(elements.timeline.max)) * 100}%`);
  updateTactile(time);
});
elements.timeline.addEventListener("change", () => seekAll(Number(elements.timeline.value)));
window.addEventListener("pointerup", () => {
  if (!state.seeking) return;
  state.seeking = false;
  seekAll(Number(elements.timeline.value));
});
elements.deleteConfirmation.addEventListener("input", () => {
  elements.confirmDeleteButton.disabled = elements.deleteConfirmation.value.trim() !== episodeName(state.activeId);
});
elements.deleteForm.addEventListener("submit", deleteActiveEpisode);
elements.cancelDeleteButton.addEventListener("click", () => elements.deleteDialog.close());
elements.toastClose.addEventListener("click", hideToast);

document.addEventListener("keydown", (event) => {
  if (event.key === "/" && !isEditingTarget(event.target)) {
    event.preventDefault();
    elements.episodeSearch.focus();
    return;
  }
  if (isEditingTarget(event.target) || elements.deleteDialog.open || elements.metadataDialog.open) return;
  if (event.code === "Space") {
    event.preventDefault();
    state.master?.paused ? playAll() : stopPlayback();
  } else if (event.key === "ArrowLeft") {
    event.preventDefault();
    seekAll((state.master?.currentTime || 0) - 5);
  } else if (event.key === "ArrowRight") {
    event.preventDefault();
    seekAll((state.master?.currentTime || 0) + 5);
  } else if (event.key === "[") {
    event.preventDefault();
    moveEpisode(-1);
  } else if (event.key === "]") {
    event.preventDefault();
    moveEpisode(1);
  }
});

window.addEventListener("beforeunload", stopPlayback);
resetTactileDashboard();
loadDataset();
