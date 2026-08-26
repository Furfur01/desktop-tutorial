(() => {
  "use strict";

  const DEFAULT_VIEW = Object.freeze({ lon: 250, lat: 35, zoom: 1.08 });

  class AtmosGlobeRenderer {
    constructor(canvas, options = {}) {
      this.canvas = canvas;
      this.mapElement = options.mapElement || canvas.closest(".weather-map");
      this.mediaElement = canvas.closest(".map-media");
      this.fallbackElement = options.fallbackElement || document.querySelector("#globeFallback");
      this.gl = canvas.getContext("webgl2", {
        alpha: false,
        antialias: true,
        depth: false,
        premultipliedAlpha: false,
        powerPreference: "high-performance",
      });
      this.lon = DEFAULT_VIEW.lon;
      this.lat = DEFAULT_VIEW.lat;
      this.zoom = DEFAULT_VIEW.zoom;
      this.velocityLon = 0;
      this.velocityLat = 0;
      this.fieldOpacity = 1;
      this.flowOpacity = 0.78;
      this.fieldSource = null;
      this.flowSource = null;
      this.fieldDirty = false;
      this.flowDirty = false;
      this.fieldTextureReady = false;
      this.flowTextureReady = false;
      this.playing = true;
      this.frameInterval = 1000 / 24;
      this.nextFrameAt = 0;
      this.manifestFrameCount = 0;
      this.timelineFrameCount = 1;
      this.timelineFrame = 0;
      this.sourceDecodeToken = 0;
      this.playbackToken = 0;
      this.decodeRequestSerial = 0;
      this.activeDecodeRequest = 0;
      this.decodePending = false;
      this.pendingFrames = null;
      this.channels = {
        field: this.createChannel("field"),
        flow: this.createChannel("flow"),
      };
      this.canvas.dataset.playing = "true";
      this.updateFrameDataset();
      this.lastTick = performance.now();
      this.textureSizes = new WeakMap();
      this.pointers = new Map();
      this.dragPoint = null;
      this.pinch = null;
      this.viewport = { lonMin: 0, lonSpan: 360, latMin: -90, latSpan: 180 };
      this.updateDataset();

      if (!this.gl) {
        this.enableFallback("当前浏览器不支持 WebGL 2，已切换为二维全球播放。");
        return;
      }

      this.initializeGraphics();
      this.bindInteraction();
      this.canvas.dataset.renderer = "webgl2";
      this.canvas.addEventListener("webglcontextlost", (event) => {
        event.preventDefault();
        this.enableFallback("三维绘图上下文已丢失，已切换为二维全球播放。");
      });
      this.resizeObserver = new ResizeObserver(() => this.resize());
      this.resizeObserver.observe(canvas);
      this.resize();
      requestAnimationFrame((time) => this.tick(time));
    }

    enableFallback(message) {
      this.fallback = true;
      this.canvas.dataset.renderer = "2d-fallback";
      this.canvas.dataset.flowMode = "animated-img-fallback";
      this.canvas.dataset.fieldMode = "animated-img-fallback";
      this.mediaElement?.classList.add("globe-fallback-mode");
      if (this.fallbackElement) {
        this.fallbackElement.textContent = message;
        this.fallbackElement.hidden = false;
      }
    }

    createChannel(name) {
      return {
        name,
        source: null,
        url: "",
        decoder: null,
        frameCount: 1,
        displayedFrame: 0,
      };
    }

    initializeGraphics() {
      const gl = this.gl;
      const vertexSource = `#version 300 es
        in vec2 aPosition;
        void main() { gl_Position = vec4(aPosition, 0.0, 1.0); }
      `;
      const fragmentSource = `#version 300 es
        precision highp float;
        uniform vec2 uResolution;
        uniform vec2 uCenter;
        uniform float uRadius;
        uniform float uLon;
        uniform float uLat;
        uniform vec4 uViewport;
        uniform float uLatSpan;
        uniform float uFieldOpacity;
        uniform float uFlowOpacity;
        uniform float uHasField;
        uniform float uHasFlow;
        uniform sampler2D uField;
        uniform sampler2D uFlow;
        out vec4 outColor;

        void main() {
          vec2 screen = gl_FragCoord.xy / uResolution;
          vec3 sky = mix(vec3(0.018, 0.043, 0.058), vec3(0.035, 0.078, 0.098), screen.y);
          float vignette = 1.0 - 0.26 * dot(screen - 0.5, screen - 0.5);
          sky *= vignette;
          vec2 point = (gl_FragCoord.xy - uCenter) / uRadius;
          float radiusSquared = dot(point, point);
          if (radiusSquared > 1.0) {
            float halo = exp(-max(length(point) - 1.0, 0.0) * 42.0) * 0.13;
            outColor = vec4(sky + vec3(0.08, 0.25, 0.34) * halo, 1.0);
            return;
          }

          float depth = sqrt(max(0.0, 1.0 - radiusSquared));
          float sinLon = sin(uLon);
          float cosLon = cos(uLon);
          float sinLat = sin(uLat);
          float cosLat = cos(uLat);
          vec3 center = vec3(cosLat * cosLon, sinLat, cosLat * sinLon);
          vec3 east = vec3(-sinLon, 0.0, cosLon);
          vec3 north = vec3(-sinLat * cosLon, cosLat, -sinLat * sinLon);
          vec3 world = normalize(point.x * east + point.y * north + depth * center);
          float horizontalRadius = length(world.xz);
          // Longitude is undefined at the exact geographic poles. Use the
          // current central meridian for that sub-pixel limit instead of
          // evaluating atan(0, 0), which is undefined in GLSL.
          float longitude = horizontalRadius < 1.0e-6
            ? degrees(uLon)
            : degrees(atan(world.z, world.x));
          float latitude = degrees(asin(clamp(world.y, -1.0, 1.0)));
          float lonOffset = mod(longitude - uViewport.x + 360.0, 360.0);
          bool valid = lonOffset <= uViewport.y + 0.001
            && latitude >= uViewport.z - 0.001
            && latitude <= uViewport.z + uLatSpan + 0.001;

          vec3 ocean = vec3(0.075, 0.205, 0.275);
          vec3 colour = ocean;
          if (valid && uHasField > 0.5) {
            vec2 uv = vec2(lonOffset / uViewport.y, (latitude - uViewport.z) / uLatSpan);
            vec3 field = texture(uField, uv).rgb;
            colour = mix(ocean, field, uFieldOpacity);
            if (uHasFlow > 0.5) {
              vec3 flowSample = texture(uFlow, uv).rgb;
              float flow = max(dot(flowSample, vec3(0.299, 0.587, 0.114)) - 0.012, 0.0);
              vec3 particles = vec3(clamp(flow * uFlowOpacity, 0.0, 1.0));
              colour = 1.0 - (1.0 - colour) * (1.0 - particles);
            }
          }

          vec3 normal = normalize(vec3(point, depth));
          float diffuse = 0.72 + 0.28 * max(dot(normal, normalize(vec3(-0.42, 0.58, 0.74))), 0.0);
          colour *= diffuse;
          float rim = pow(1.0 - depth, 2.4);
          colour += vec3(0.05, 0.18, 0.24) * rim;
          float edge = 1.0 - smoothstep(0.982, 1.0, radiusSquared);
          outColor = vec4(mix(sky, colour, edge), 1.0);
        }
      `;
      const compile = (type, source) => {
        const shader = gl.createShader(type);
        gl.shaderSource(shader, source);
        gl.compileShader(shader);
        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
        return shader;
      };
      this.program = gl.createProgram();
      gl.attachShader(this.program, compile(gl.VERTEX_SHADER, vertexSource));
      gl.attachShader(this.program, compile(gl.FRAGMENT_SHADER, fragmentSource));
      gl.linkProgram(this.program);
      if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(this.program));
      gl.useProgram(this.program);
      const buffer = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, -1,1, 1,-1, 1,1]), gl.STATIC_DRAW);
      const position = gl.getAttribLocation(this.program, "aPosition");
      gl.enableVertexAttribArray(position);
      gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);
      this.uniforms = Object.fromEntries([
        "uResolution", "uCenter", "uRadius", "uLon", "uLat", "uViewport", "uLatSpan",
        "uFieldOpacity", "uFlowOpacity", "uHasField", "uHasFlow",
      ].map((name) => [name, gl.getUniformLocation(this.program, name)]));
      this.fieldTexture = this.createTexture(0, "uField");
      this.flowTexture = this.createTexture(1, "uFlow");
    }

    createTexture(unit, uniformName) {
      const gl = this.gl;
      const texture = gl.createTexture();
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.uniform1i(gl.getUniformLocation(this.program, uniformName), unit);
      return texture;
    }

    sourceSize(source) {
      return {
        width: source?.naturalWidth || source?.videoWidth || source?.displayWidth || source?.codedWidth || source?.width || 0,
        height: source?.naturalHeight || source?.videoHeight || source?.displayHeight || source?.codedHeight || source?.height || 0,
      };
    }

    upload(texture, unit, source) {
      const size = this.sourceSize(source);
      if (!size.width || !size.height) return false;
      const gl = this.gl;
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
      const previous = this.textureSizes.get(texture);
      if (previous?.width === size.width && previous?.height === size.height) {
        gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, gl.RGBA, gl.UNSIGNED_BYTE, source);
      } else {
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);
        this.textureSizes.set(texture, size);
      }
      return true;
    }

    setManifest(manifest) {
      const viewport = manifest.viewport || {};
      const lonMin = Number(viewport.lon_min ?? viewport.lonMin ?? 0);
      let lonMax = Number(viewport.lon_max ?? viewport.lonMax ?? 360);
      const latMin = Number(viewport.lat_min ?? viewport.latMin ?? -90);
      const latMax = Number(viewport.lat_max ?? viewport.latMax ?? 90);
      if (lonMax <= lonMin) lonMax += 360;
      this.viewport = {
        lonMin: Number.isFinite(lonMin) ? lonMin : 0,
        lonSpan: Number.isFinite(lonMax - lonMin) ? Math.max(0.001, Math.min(360, lonMax - lonMin)) : 360,
        latMin: Number.isFinite(latMin) ? latMin : -90,
        latSpan: Number.isFinite(latMax - latMin) ? Math.max(0.001, Math.min(180, latMax - latMin)) : 180,
      };
      this.frameInterval = 1000 / Math.max(1, Number(manifest.fps) || 24);
      this.manifestFrameCount = Math.max(0, Math.floor(Number(manifest.frames) || 0));
      this.updateTimelineCount();
    }

    async setSources(field, flow, fieldUrl = "", flowUrl = "", options = {}) {
      // Keep the original (field, flow, flowUrl) signature working while callers
      // migrate to the unambiguous four-argument form.
      if (arguments.length === 3) {
        flowUrl = fieldUrl;
        fieldUrl = "";
      }
      const requestedFrame = Math.max(0, Math.floor(Number(options.timelineFrame) || 0));
      this.fieldSource = field;
      this.fieldDirty = true;
      this.flowSource = flow;
      this.flowDirty = true;
      this.channels.field.source = field;
      this.channels.flow.source = flow;
      if (!field) this.fieldTextureReady = false;
      if (!flow) this.flowTextureReady = false;
      this.timelineFrame = requestedFrame;
      this.channels.field.displayedFrame = requestedFrame;
      this.channels.flow.displayedFrame = requestedFrame;
      this.updateFrameDataset();
      if (this.fallback) return;
      const resolvedFieldUrl = fieldUrl || field?.currentSrc || field?.src || "";
      const resolvedFlowUrl = flowUrl || flow?.currentSrc || flow?.src || "";
      await this.prepareSourceDecoders(resolvedFieldUrl, resolvedFlowUrl, requestedFrame);
    }

    async prepareSourceDecoders(fieldUrl, flowUrl, requestedFrame = 0) {
      const token = ++this.sourceDecodeToken;
      this.invalidateFrameRequest();
      this.closePendingFrames();
      this.closeDecoders();
      this.channels.field.url = fieldUrl;
      this.channels.flow.url = flowUrl;
      this.channels.field.frameCount = 1;
      this.channels.flow.frameCount = 1;
      this.canvas.dataset.fieldMode = "static-image";
      this.canvas.dataset.flowMode = "static-image";
      this.updateTimelineCount();
      if (!("ImageDecoder" in window)) {
        this.enableFallback("当前浏览器无法逐帧解码 WebP，已切换为二维全球播放。");
        return;
      }
      try {
        const settled = await Promise.allSettled([
          this.createDecoder(fieldUrl),
          this.createDecoder(flowUrl),
        ]);
        const failed = settled.find((result) => result.status === "rejected");
        if (failed) {
          for (const result of settled) {
            if (result.status === "fulfilled") {
              try { result.value?.decoder.close(); } catch { /* already closed */ }
            }
          }
          throw failed.reason;
        }
        const [fieldResult, flowResult] = settled.map((result) => result.value);
        if (token !== this.sourceDecodeToken) {
          for (const result of [fieldResult, flowResult]) {
            try { result?.decoder.close(); } catch { /* already closed */ }
          }
          return;
        }
        for (const [name, result] of [["field", fieldResult], ["flow", flowResult]]) {
          const channel = this.channels[name];
          channel.decoder = result?.decoder || null;
          channel.frameCount = result?.frameCount || 1;
          this.canvas.dataset[`${name}Mode`] = channel.frameCount > 1 ? "image-decoder" : "static-image";
        }
        this.updateTimelineCount();
        this.timelineFrame = this.timelineFrameCount > 1
          ? requestedFrame % this.timelineFrameCount
          : 0;
        this.nextFrameAt = performance.now() + this.frameInterval;
        const initialFrameReady = await this.requestFramePair(
          this.timelineFrame,
          { force: true, sourceToken: token },
        );
        if (!initialFrameReady && token === this.sourceDecodeToken) {
          throw new Error("The first synchronized frame could not be decoded.");
        }
        if (token === this.sourceDecodeToken) this.commitPendingFrames();
      } catch (error) {
        if (token === this.sourceDecodeToken) {
          this.canvas.dataset.flowMode = "animated-img-fallback";
          this.canvas.dataset.fieldMode = "animated-img-fallback";
          console.warn("WebP frame decoding failed; using the image fallback.", error);
          this.enableFallback("图层逐帧解码失败，已切换为二维全球播放。");
          this.closeDecoders();
        }
      }
    }

    async createDecoder(url) {
      if (!url) return null;
      const response = await fetch(url, { cache: "force-cache" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.arrayBuffer();
      const decoder = new ImageDecoder({ data, type: "image/webp", preferAnimation: true });
      try {
        await decoder.tracks.ready;
      } catch (error) {
        try { decoder.close(); } catch { /* already closed */ }
        throw error;
      }
      return {
        decoder,
        frameCount: Math.max(1, Number(decoder.tracks.selectedTrack?.frameCount) || 1),
      };
    }

    updateTimelineCount() {
      const counts = Object.values(this.channels).map((channel) => Math.max(1, channel.frameCount || 1));
      const animatedCounts = counts.filter((count) => count > 1);
      const decodedCount = animatedCounts.length ? Math.max(...animatedCounts) : 1;
      this.timelineFrameCount = this.manifestFrameCount > 1 && animatedCounts.length
        ? this.manifestFrameCount
        : decodedCount;
      this.timelineFrame = ((this.timelineFrame % this.timelineFrameCount) + this.timelineFrameCount) % this.timelineFrameCount;
      this.updateFrameDataset();
    }

    decoderFrameIndex(channel, timelineFrame) {
      if (channel.frameCount <= 1 || this.timelineFrameCount <= 1) return 0;
      if (channel.frameCount === this.timelineFrameCount) return timelineFrame % channel.frameCount;
      return Math.min(
        channel.frameCount - 1,
        Math.floor((timelineFrame % this.timelineFrameCount) * channel.frameCount / this.timelineFrameCount),
      );
    }

    async requestFramePair(timelineFrame, options = {}) {
      if (this.decodePending || this.pendingFrames) return false;
      const sourceToken = options.sourceToken ?? this.sourceDecodeToken;
      const playbackToken = this.playbackToken;
      const force = Boolean(options.force);
      const requestId = ++this.decodeRequestSerial;
      this.activeDecodeRequest = requestId;
      this.decodePending = true;
      const decodeChannel = async (channel) => {
        if (!channel.decoder || (!force && channel.frameCount <= 1)) {
          return { image: null, frameIndex: 0 };
        }
        const frameIndex = this.decoderFrameIndex(channel, timelineFrame);
        const result = await channel.decoder.decode({ frameIndex, completeFramesOnly: true });
        return { image: result.image, frameIndex };
      };
      let decoded = null;
      try {
        const settled = await Promise.allSettled([
          decodeChannel(this.channels.field),
          decodeChannel(this.channels.flow),
        ]);
        decoded = {
          field: settled[0].status === "fulfilled" ? settled[0].value : null,
          flow: settled[1].status === "fulfilled" ? settled[1].value : null,
        };
        const failed = settled.find((result) => result.status === "rejected");
        if (failed) throw failed.reason;
        const stale = sourceToken !== this.sourceDecodeToken
          || requestId !== this.activeDecodeRequest
          || (!force && (playbackToken !== this.playbackToken || !this.playing));
        if (stale) {
          this.closeDecodedPair(decoded);
          return false;
        }
        this.pendingFrames = { timelineFrame, ...decoded };
        decoded = null;
        return true;
      } catch (error) {
        this.closeDecodedPair(decoded);
        if (sourceToken === this.sourceDecodeToken) {
          console.warn(`Frame ${timelineFrame} failed to decode.`, error);
        }
        return false;
      } finally {
        if (requestId === this.activeDecodeRequest) {
          this.decodePending = false;
          this.activeDecodeRequest = 0;
        }
      }
    }

    invalidateFrameRequest() {
      this.playbackToken += 1;
      this.decodeRequestSerial += 1;
      this.activeDecodeRequest = 0;
      this.decodePending = false;
    }

    closeDecodedPair(pair) {
      if (!pair) return;
      for (const name of ["field", "flow"]) {
        try { pair[name]?.image?.close(); } catch { /* already closed */ }
      }
    }

    closePendingFrames() {
      if (!this.pendingFrames) return;
      this.closeDecodedPair(this.pendingFrames);
      this.pendingFrames = null;
    }

    closeDecoders() {
      for (const channel of Object.values(this.channels)) {
        if (channel.decoder) {
          try { channel.decoder.close(); } catch { /* already closed */ }
          channel.decoder = null;
        }
      }
    }

    updateFrameDataset() {
      this.canvas.dataset.frame = String(this.timelineFrame);
      this.canvas.dataset.count = String(this.timelineFrameCount);
      this.canvas.dataset.frames = String(this.timelineFrameCount);
      this.canvas.dataset.frameCount = String(this.timelineFrameCount);
      for (const name of ["field", "flow"]) {
        const channel = this.channels[name];
        this.canvas.dataset[`${name}Frame`] = String(channel.displayedFrame);
        this.canvas.dataset[`${name}Frames`] = String(channel.frameCount);
      }
    }

    emitFrameEvent() {
      const detail = {
        frame: this.timelineFrame,
        count: this.timelineFrameCount,
        fieldFrame: this.channels.field.displayedFrame,
        fieldCount: this.channels.field.frameCount,
        flowFrame: this.channels.flow.displayedFrame,
        flowCount: this.channels.flow.frameCount,
        playing: this.playing,
      };
      this.canvas.dispatchEvent(new CustomEvent("atmos-frame", { detail, bubbles: true }));
    }

    commitPendingFrames() {
      const pending = this.pendingFrames;
      if (!pending) return;
      let uploaded = true;
      for (const [name, texture, unit] of [
        ["field", this.fieldTexture, 0],
        ["flow", this.flowTexture, 1],
      ]) {
        const decoded = pending[name];
        if (decoded?.image) {
          const ok = this.upload(texture, unit, decoded.image);
          uploaded = uploaded && ok;
          if (ok) {
            if (name === "field") this.fieldTextureReady = true;
            else this.flowTextureReady = true;
          }
        }
      }
      if (uploaded) {
        this.timelineFrame = pending.timelineFrame;
        this.channels.field.displayedFrame = pending.field.frameIndex;
        this.channels.flow.displayedFrame = pending.flow.frameIndex;
        this.updateFrameDataset();
        this.emitFrameEvent();
      }
      this.closePendingFrames();
    }

    setOpacity(fieldOpacity, flowOpacity) {
      this.fieldOpacity = fieldOpacity;
      this.flowOpacity = flowOpacity;
      this.mediaElement?.style.setProperty("--field-opacity", String(fieldOpacity));
      this.mediaElement?.style.setProperty("--flow-opacity", String(flowOpacity));
    }

    setPlaying(playing) {
      const nextPlaying = Boolean(playing);
      this.mediaElement?.classList.toggle("flow-paused", !nextPlaying);
      this.canvas.dataset.playing = String(nextPlaying);
      if (nextPlaying === this.playing) return;
      this.playing = nextPlaying;
      this.playbackToken += 1;
      this.closePendingFrames();
      if (nextPlaying) this.nextFrameAt = 0;
    }

    resize() {
      if (!this.gl || this.fallback) return;
      const rect = this.canvas.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const width = Math.max(1, Math.round(rect.width * dpr));
      const height = Math.max(1, Math.round(rect.height * dpr));
      if (this.canvas.width !== width || this.canvas.height !== height) {
        this.canvas.width = width;
        this.canvas.height = height;
        this.gl.viewport(0, 0, width, height);
      }
      const desktop = rect.width > 760;
      const left = desktop && this.mapElement?.classList.contains("drawer-open") ? 360 : 0;
      const right = desktop ? 76 : 0;
      const top = desktop ? 54 : 46;
      const bottom = desktop ? 96 : 76;
      this.centerCss = {
        x: left + (rect.width - left - right) / 2,
        y: top + (rect.height - top - bottom) / 2,
      };
      this.baseRadiusCss = Math.max(72, Math.min((rect.width - left - right) * 0.44, (rect.height - top - bottom) * 0.47));
      this.dpr = dpr;
    }

    bindInteraction() {
      const canvas = this.canvas;
      const pointerPosition = (event) => ({ x: event.clientX, y: event.clientY, time: performance.now() });
      canvas.addEventListener("pointerdown", (event) => {
        const point = pointerPosition(event);
        const distance = Math.hypot(point.x - this.centerCss.x, point.y - this.centerCss.y);
        if (!this.pointers.size && distance > this.baseRadiusCss * this.zoom * 1.06) return;
        canvas.setPointerCapture(event.pointerId);
        this.pointers.set(event.pointerId, point);
        this.velocityLon = 0;
        this.velocityLat = 0;
        canvas.classList.add("dragging");
        if (this.pointers.size === 1) this.dragPoint = point;
        if (this.pointers.size === 2) {
          const [a, b] = [...this.pointers.values()];
          this.pinch = { distance: Math.hypot(a.x - b.x, a.y - b.y), zoom: this.zoom };
        }
      });
      canvas.addEventListener("pointermove", (event) => {
        if (!this.pointers.has(event.pointerId)) return;
        const point = pointerPosition(event);
        this.pointers.set(event.pointerId, point);
        if (this.pointers.size >= 2 && this.pinch) {
          const [a, b] = [...this.pointers.values()];
          const distance = Math.hypot(a.x - b.x, a.y - b.y);
          this.zoom = Math.max(0.68, Math.min(1.7, this.pinch.zoom * distance / Math.max(this.pinch.distance, 1)));
          return;
        }
        if (!this.dragPoint) this.dragPoint = point;
        const dx = point.x - this.dragPoint.x;
        const dy = point.y - this.dragPoint.y;
        const dt = Math.max(8, point.time - this.dragPoint.time);
        const radius = Math.max(1, this.baseRadiusCss * this.zoom);
        const deltaLon = -dx / radius * 57.2958;
        const deltaLat = dy / radius * 57.2958;
        this.lon += deltaLon;
        this.lat = Math.max(-85, Math.min(85, this.lat + deltaLat));
        this.velocityLon = Math.max(-0.09, Math.min(0.09, deltaLon / dt));
        this.velocityLat = Math.max(-0.09, Math.min(0.09, deltaLat / dt));
        this.dragPoint = point;
      });
      const endPointer = (event) => {
        this.pointers.delete(event.pointerId);
        this.pinch = null;
        this.dragPoint = this.pointers.size === 1 ? [...this.pointers.values()][0] : null;
        if (!this.pointers.size) canvas.classList.remove("dragging");
      };
      canvas.addEventListener("pointerup", endPointer);
      canvas.addEventListener("pointercancel", endPointer);
      canvas.addEventListener("wheel", (event) => {
        event.preventDefault();
        this.velocityLon = 0;
        this.velocityLat = 0;
        this.zoom = Math.max(0.68, Math.min(1.7, this.zoom * Math.exp(-event.deltaY * 0.0012)));
      }, { passive: false });
      canvas.addEventListener("dblclick", () => this.reset());
      canvas.addEventListener("keydown", (event) => {
        const keyMoves = { ArrowLeft: -8, ArrowRight: 8, ArrowUp: 8, ArrowDown: -8 };
        if (event.key in keyMoves) {
          event.preventDefault();
          if (event.key === "ArrowLeft" || event.key === "ArrowRight") this.lon += keyMoves[event.key];
          else this.lat = Math.max(-85, Math.min(85, this.lat + keyMoves[event.key]));
        } else if (["Home", "r", "R"].includes(event.key)) {
          event.preventDefault();
          this.reset();
        } else if (["+", "="].includes(event.key)) this.zoom = Math.min(1.7, this.zoom * 1.1);
        else if (event.key === "-") this.zoom = Math.max(0.68, this.zoom / 1.1);
      });
    }

    reset() {
      Object.assign(this, DEFAULT_VIEW);
      this.velocityLon = 0;
      this.velocityLat = 0;
      this.updateDataset();
    }

    updateDataset() {
      this.lon = ((this.lon + 540) % 360) - 180;
      this.canvas.dataset.lon = this.lon.toFixed(2);
      this.canvas.dataset.lat = this.lat.toFixed(2);
      this.canvas.dataset.zoom = this.zoom.toFixed(3);
    }

    tick(time) {
      if (!this.gl || this.fallback) return;
      const dt = Math.min(34, Math.max(0, time - this.lastTick));
      this.lastTick = time;
      if (!document.hidden) {
        if (!this.pointers.size) {
          this.lon += this.velocityLon * dt;
          this.lat = Math.max(-85, Math.min(85, this.lat + this.velocityLat * dt));
          const damping = Math.exp(-dt / 220);
          this.velocityLon *= damping;
          this.velocityLat *= damping;
        }
        this.draw(time);
      }
      requestAnimationFrame((next) => this.tick(next));
    }

    draw(time) {
      const gl = this.gl;
      if (gl.isContextLost()) return;
      gl.useProgram(this.program);
      if (this.fieldDirty && this.upload(this.fieldTexture, 0, this.fieldSource)) {
        this.fieldDirty = false;
        this.fieldTextureReady = true;
      }
      if (this.flowDirty && this.upload(this.flowTexture, 1, this.flowSource)) {
        this.flowDirty = false;
        this.flowTextureReady = true;
      }
      this.commitPendingFrames();
      const hasAnimatedDecoder = Object.values(this.channels).some(
        (channel) => channel.decoder && channel.frameCount > 1,
      );
      if (this.playing
        && hasAnimatedDecoder
        && this.timelineFrameCount > 1
        && !this.decodePending
        && !this.pendingFrames
        && time >= this.nextFrameAt) {
        const targetFrame = (this.timelineFrame + 1) % this.timelineFrameCount;
        this.nextFrameAt = time + this.frameInterval;
        this.requestFramePair(targetFrame);
      }
      const radius = this.baseRadiusCss * this.zoom * this.dpr;
      gl.uniform2f(this.uniforms.uResolution, this.canvas.width, this.canvas.height);
      gl.uniform2f(this.uniforms.uCenter, this.centerCss.x * this.dpr, (this.canvas.clientHeight - this.centerCss.y) * this.dpr);
      gl.uniform1f(this.uniforms.uRadius, radius);
      gl.uniform1f(this.uniforms.uLon, this.lon * Math.PI / 180);
      gl.uniform1f(this.uniforms.uLat, this.lat * Math.PI / 180);
      gl.uniform4f(this.uniforms.uViewport, this.viewport.lonMin, this.viewport.lonSpan, this.viewport.latMin, 0);
      gl.uniform1f(this.uniforms.uLatSpan, this.viewport.latSpan);
      gl.uniform1f(this.uniforms.uFieldOpacity, this.fieldOpacity);
      gl.uniform1f(this.uniforms.uFlowOpacity, this.flowOpacity);
      gl.uniform1f(this.uniforms.uHasField, this.fieldTextureReady ? 1 : 0);
      gl.uniform1f(this.uniforms.uHasFlow, this.flowTextureReady ? 1 : 0);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
      this.updateDataset();
    }
  }

  window.AtmosGlobeRenderer = AtmosGlobeRenderer;
})();
