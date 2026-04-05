/**
 * IsaacLab RL Controller 메인 앱
 */

class RLApp {
    constructor() {
        this.baseUrl = window.location.origin;
        this.ws = null;
        this.wsReconnectInterval = null;
        this.statusPollInterval = null;
        this.autoScroll = true;
        this._streamingActive = false;
    }

    init() {
        this.setupTabNavigation();
        this.setupTrainControls();
        this.setupInferenceControls();
        this.setupLogControls();
        this.setupCameraControls();
        this.connectWebSocket();
        this.startStatusPolling();
    }

    // ===== 탭 네비게이션 =====

    setupTabNavigation() {
        const tabBtns = document.querySelectorAll('.tab-btn');
        tabBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                const tabId = btn.dataset.tab;

                tabBtns.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');

                document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                document.getElementById(`tab-${tabId}`).classList.add('active');
            });
        });
    }

    // ===== Train 제어 =====

    setupTrainControls() {
        const btnStart = document.getElementById('btnStartTrain');
        const btnStop = document.getElementById('btnStopTrain');
        const numEnvsInput = document.getElementById('numEnvs');

        btnStart.addEventListener('click', async () => {
            const numEnvs = parseInt(numEnvsInput.value) || 256;
            btnStart.disabled = true;

            const result = await this.post('/api/rl/train/start', { num_envs: numEnvs });
            if (result.success) {
                btnStart.style.display = 'none';
                btnStop.style.display = 'block';
            } else {
                alert('Train 시작 실패: ' + (result.error || '알 수 없는 오류'));
                btnStart.disabled = false;
            }
        });

        btnStop.addEventListener('click', async () => {
            btnStop.disabled = true;
            const result = await this.post('/api/rl/train/stop');
            btnStop.style.display = 'none';
            btnStart.style.display = 'block';
            btnStart.disabled = false;
            btnStop.disabled = false;
        });

    }

    // ===== Inference 제어 =====

    setupInferenceControls() {
        const btnStart = document.getElementById('btnStartInference');
        const btnStop = document.getElementById('btnStopInference');

        btnStart.addEventListener('click', async () => {
            btnStart.disabled = true;

            const result = await this.post('/api/rl/inference/start');
            if (result.success) {
                btnStart.style.display = 'none';
                btnStop.style.display = 'block';
            } else {
                alert('Inference 시작 실패: ' + (result.error || '알 수 없는 오류'));
                btnStart.disabled = false;
            }
        });

        btnStop.addEventListener('click', async () => {
            btnStop.disabled = true;
            const result = await this.post('/api/rl/inference/stop');
            btnStop.style.display = 'none';
            btnStart.style.display = 'block';
            btnStart.disabled = false;
            btnStop.disabled = false;
        });
    }

    // ===== 카메라 제어 =====

    setupCameraControls() {
        const azimuth = document.getElementById('azimuthSlider');
        const elevation = document.getElementById('elevationSlider');
        const distance = document.getElementById('distanceSlider');
        const btnReset = document.getElementById('btnResetCamera');

        const updateCamera = () => {
            document.getElementById('azimuthValue').textContent = `${azimuth.value}°`;
            document.getElementById('elevationValue').textContent = `${elevation.value}°`;
            document.getElementById('distanceValue').textContent = `${parseFloat(distance.value).toFixed(1)}m`;
            this.sendCameraParams();
        };

        azimuth.addEventListener('input', updateCamera);
        elevation.addEventListener('input', updateCamera);
        distance.addEventListener('input', updateCamera);

        btnReset.addEventListener('click', () => {
            azimuth.value = 45;
            elevation.value = 30;
            distance.value = 5;
            updateCamera();
        });
    }

    async sendCameraParams() {
        const params = {
            azimuth: parseFloat(document.getElementById('azimuthSlider').value),
            elevation: parseFloat(document.getElementById('elevationSlider').value),
            distance: parseFloat(document.getElementById('distanceSlider').value),
        };
        await this.post('/api/rl/camera', params).catch(() => {});
    }

    // ===== 로그 제어 =====

    setupLogControls() {
        const btnClear = document.getElementById('btnClearLogs');
        btnClear.addEventListener('click', () => this.clearLogs());

        // 로그 영역 스크롤 감지: 사용자가 위로 스크롤하면 자동 스크롤 비활성화
        const logContainer = document.getElementById('logContainer');
        logContainer.addEventListener('scroll', () => {
            const { scrollTop, scrollHeight, clientHeight } = logContainer;
            this.autoScroll = (scrollHeight - scrollTop - clientHeight) < 50;
        });
    }

    appendLog(line) {
        if (!line) return;

        const container = document.getElementById('logContainer');

        // 플레이스홀더 제거
        const placeholder = container.querySelector('.log-placeholder');
        if (placeholder) placeholder.remove();

        const lineEl = document.createElement('div');
        lineEl.className = 'log-line';

        // 로그 라인 색상 분류
        if (line.startsWith('[시스템]')) {
            lineEl.classList.add('log-system');
        } else if (line.includes('[ERROR]') || line.includes('Error') || line.includes('error')) {
            lineEl.classList.add('log-error');
        } else if (line.includes('[WARNING]') || line.includes('Warning')) {
            lineEl.classList.add('log-warning');
        } else if (line.includes('[INFO]')) {
            lineEl.classList.add('log-info');
        }

        lineEl.textContent = line;
        container.appendChild(lineEl);

        // 로그 라인 수 제한 (DOM 성능 보호)
        while (container.children.length > 5000) {
            container.removeChild(container.firstChild);
        }

        // 자동 스크롤
        if (this.autoScroll) {
            container.scrollTop = container.scrollHeight;
        }
    }

    clearLogs() {
        const container = document.getElementById('logContainer');
        container.innerHTML = '<div class="log-placeholder">프로세스를 시작하면 로그가 표시됩니다.</div>';
        this.autoScroll = true;
    }

    // ===== WebSocket 연결 =====

    connectWebSocket() {
        if (this.ws) {
            this.ws.close();
        }

        const wsUrl = `ws://${window.location.host}/ws/logs`;
        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
            console.log('WebSocket 연결됨');
            this.updateConnectionStatus(true);

            // 재연결 타이머 해제
            if (this.wsReconnectInterval) {
                clearInterval(this.wsReconnectInterval);
                this.wsReconnectInterval = null;
            }
        };

        this.ws.onmessage = (event) => {
            this.appendLog(event.data);
        };

        this.ws.onerror = () => {
            console.error('WebSocket 오류');
        };

        this.ws.onclose = () => {
            console.log('WebSocket 연결 종료');
            this.updateConnectionStatus(false);

            // 자동 재연결 (3초 간격)
            if (!this.wsReconnectInterval) {
                this.wsReconnectInterval = setInterval(() => {
                    console.log('WebSocket 재연결 시도...');
                    this.connectWebSocket();
                }, 3000);
            }
        };
    }

    // ===== 프레임 스트리밍 =====

    startFrameStreaming() {
        if (this._streamingActive) return;
        this._streamingActive = true;

        const fetchNextFrame = async () => {
            if (!this._streamingActive) return;

            try {
                const response = await fetch(`${this.baseUrl}/api/rl/frame?t=${Date.now()}`);
                if (response.ok && response.status === 200) {
                    const blob = await response.blob();
                    if (blob.size > 0) {
                        const url = URL.createObjectURL(blob);
                        const preview = document.getElementById('previewImage');
                        const overlay = document.getElementById('previewOverlay');
                        if (preview) {
                            if (preview._blobUrl) URL.revokeObjectURL(preview._blobUrl);
                            preview._blobUrl = url;
                            preview.src = url;
                        }
                        if (overlay) overlay.classList.add('hidden');
                    }
                }
            } catch (e) {
                // 프레임 페칭 오류 무시
            }

            if (this._streamingActive) {
                setTimeout(fetchNextFrame, 200);  // ~5fps
            }
        };

        fetchNextFrame();
    }

    stopFrameStreaming() {
        this._streamingActive = false;
        const overlay = document.getElementById('previewOverlay');
        if (overlay) overlay.classList.remove('hidden');
    }

    // ===== 상태 폴링 =====

    startStatusPolling() {
        this.pollStatus();
        this.statusPollInterval = setInterval(() => this.pollStatus(), 2000);
    }

    async pollStatus() {
        try {
            const result = await this.get('/api/rl/status');
            if (result.success) {
                this.updateStatusPanel(result.data);
                this.updateButtonStates(result.data);
                this.updateProgressBar(result.data);

                // 실행 중이면 프레임 스트리밍 시작, 아니면 중지
                if (result.data.running && !this._streamingActive) {
                    this.startFrameStreaming();
                } else if (!result.data.running && this._streamingActive) {
                    this.stopFrameStreaming();
                }
            }
        } catch (e) {
            // 폴링 오류 무시
        }
    }

    updateStatusPanel(status) {
        const modeEl = document.getElementById('statusMode');
        const runningEl = document.getElementById('statusRunning');
        const numEnvsEl = document.getElementById('statusNumEnvs');
        const logLinesEl = document.getElementById('statusLogLines');

        if (status.mode) {
            modeEl.textContent = status.mode === 'train' ? 'Train (학습)' : 'Inference (추론)';
            modeEl.className = 'status-value active';
        } else {
            modeEl.textContent = '-';
            modeEl.className = 'status-value';
        }

        if (status.running) {
            runningEl.textContent = '실행 중';
            runningEl.className = 'status-value running';
        } else {
            runningEl.textContent = '대기 중';
            runningEl.className = 'status-value';
        }

        numEnvsEl.textContent = status.num_envs > 0 ? status.num_envs : '-';
        logLinesEl.textContent = `${status.log_lines}줄`;
    }

    updateButtonStates(status) {
        const btnStartTrain = document.getElementById('btnStartTrain');
        const btnStopTrain = document.getElementById('btnStopTrain');
        const btnStartInference = document.getElementById('btnStartInference');
        const btnStopInference = document.getElementById('btnStopInference');

        if (status.running) {
            if (status.mode === 'train') {
                btnStartTrain.style.display = 'none';
                btnStopTrain.style.display = 'block';
                btnStopTrain.disabled = false;
                btnStartInference.disabled = true;
            } else if (status.mode === 'inference') {
                btnStartInference.style.display = 'none';
                btnStopInference.style.display = 'block';
                btnStopInference.disabled = false;
                btnStartTrain.disabled = true;
            }
        } else {
            btnStartTrain.style.display = 'block';
            btnStartTrain.disabled = false;
            btnStopTrain.style.display = 'none';
            btnStartInference.style.display = 'block';
            btnStartInference.disabled = false;
            btnStopInference.style.display = 'none';
        }
    }

    // ===== 학습 진행률 =====

    updateProgressBar(status) {
        const panel = document.getElementById('progressPanel');
        const progress = status.progress;

        // Train 모드이고 진행률 데이터가 있을 때만 표시
        if (status.mode === 'train' && progress && progress.total > 0) {
            panel.classList.remove('hidden');

            const percent = Math.min(100, (progress.current / progress.total) * 100);
            const fill = document.getElementById('progressBarFill');
            fill.style.width = `${percent}%`;

            document.getElementById('progressStats').textContent =
                `${progress.current} / ${progress.total}`;
            document.getElementById('progressPercent').textContent =
                `${percent.toFixed(1)}%`;

            const rewardEl = document.getElementById('progressReward');
            if (progress.mean_reward !== null && progress.mean_reward !== undefined) {
                rewardEl.textContent = `보상: ${progress.mean_reward.toFixed(2)}`;
            } else {
                rewardEl.textContent = '';
            }

            const etaEl = document.getElementById('progressEta');
            if (progress.eta) {
                etaEl.textContent = `ETA: ${progress.eta}`;
            } else {
                etaEl.textContent = '';
            }
        } else if (!status.running) {
            panel.classList.add('hidden');
        }
    }

    // ===== 연결 상태 =====

    updateConnectionStatus(connected) {
        const statusEl = document.getElementById('connectionStatus');
        const dot = statusEl.querySelector('.status-dot');
        const text = statusEl.querySelector('span:last-child');

        if (connected) {
            dot.classList.remove('disconnected');
            dot.classList.add('connected');
            text.textContent = '연결됨';
        } else {
            dot.classList.remove('connected');
            dot.classList.add('disconnected');
            text.textContent = '연결 끊김';
        }
    }

    // ===== HTTP 유틸리티 =====

    async get(endpoint) {
        const response = await fetch(`${this.baseUrl}${endpoint}`);
        return response.json();
    }

    async post(endpoint, data = {}) {
        const response = await fetch(`${this.baseUrl}${endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data),
        });
        return response.json();
    }
}

// 앱 초기화
document.addEventListener('DOMContentLoaded', () => {
    const app = new RLApp();
    app.init();
});
