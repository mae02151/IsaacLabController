/**
 * IsaacLab Controller 메인 앱
 * VLA / Reinforcement 탭 지원
 */


class App {
    constructor() {
        this.api = window.api;
        this.currentMainTab = 'vla';  // 현재 활성 메인 탭
        this.isStreaming = true;
        this.streamingInterval = null;
        this.objects = [];
        this.materials = [];
        this.cameras = [];

        // 로봇 관련 상태
        this.robotAvailable = false;
        this.teleopRunning = false;
        this.jointPollingInterval = null;
    }

    async init() {
        this.setupMainTabNavigation();
        this.setupTabNavigation();
        this.setupCameraControls();
        this.setupObjectControls();
        this.setupMaterialControls();
        this.setupPreviewControls();
        this.setupRLControls();
        this.setupRobotControls();

        await this.loadInitialData();
        await this.loadCameras();
        await this.loadRobotInfo();
        this.startStreaming();
        this.updateConnectionStatus(true);
    }

    // ===== 메인 탭 네비게이션 (VLA / Reinforcement) =====

    setupMainTabNavigation() {
        const mainTabBtns = document.querySelectorAll('.main-tab-btn');
        const mainTabContents = document.querySelectorAll('.main-tab-content');

        mainTabBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                const tabId = btn.dataset.mainTab;
                this.currentMainTab = tabId;

                // 버튼 상태 변경
                mainTabBtns.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');

                // 컨텐츠 표시
                mainTabContents.forEach(c => c.classList.remove('active'));
                document.getElementById(`main-tab-${tabId}`).classList.add('active');

                console.log('메인 탭 변경:', tabId);
            });
        });
    }

    // ===== 탭 네비게이션 (Camera / Objects / Materials) =====

    setupTabNavigation() {
        // VLA 탭용 (카메라/물체/재질)
        this.setupTabsForSection('vla');
        // RL 탭용 (Train/Inference)
        this.setupTabsForSection('rl');
    }

    setupTabsForSection(suffix) {
        const tabBtns = document.querySelectorAll(`[data-tab$="-${suffix}"]`);

        tabBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                const tabId = btn.dataset.tab;
                const parentPanel = btn.closest('.control-panel');

                // 해당 패널 내 탭들만 처리
                parentPanel.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
                parentPanel.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

                btn.classList.add('active');
                document.getElementById(`tab-${tabId}`).classList.add('active');
            });
        });
    }

    // ===== 카메라 제어 =====

    setupCameraControls() {
        // VLA만 카메라 제어 설정
        ['vla'].forEach(suffix => {
            this.setupCameraControlsForSection(suffix);
        });
    }

    setupCameraControlsForSection(suffix) {
        // 카메라 선택 드롭다운
        const cameraSelect = document.getElementById(`cameraSelect-${suffix}`);
        if (cameraSelect) {
            cameraSelect.addEventListener('change', async () => {
                if (cameraSelect.value) {
                    const result = await this.api.selectCamera(cameraSelect.value);
                    if (result.success) {
                        console.log('카메라 변경됨:', cameraSelect.value);
                    }
                }
            });
        }

        // 카메라 새로고침 버튼
        const btnRefreshCameras = document.getElementById(`btnRefreshCameras-${suffix}`);
        if (btnRefreshCameras) {
            btnRefreshCameras.addEventListener('click', () => this.loadCameras());
        }

        // Orbit 슬라이더
        const azimuth = document.getElementById(`azimuth-${suffix}`);
        const elevation = document.getElementById(`elevation-${suffix}`);
        const distance = document.getElementById(`distance-${suffix}`);

        if (azimuth && elevation && distance) {
            const updateOrbit = () => {
                document.getElementById(`azimuthValue-${suffix}`).textContent = azimuth.value;
                document.getElementById(`elevationValue-${suffix}`).textContent = elevation.value;
                document.getElementById(`distanceValue-${suffix}`).textContent = distance.value;

                this.api.setCameraOrbit(
                    parseFloat(azimuth.value),
                    parseFloat(elevation.value),
                    parseFloat(distance.value)
                );
            };

            azimuth.addEventListener('input', updateOrbit);
            elevation.addEventListener('input', updateOrbit);
            distance.addEventListener('input', updateOrbit);
        }

        // 위치 직접 입력
        const btnSetPosition = document.getElementById(`btnSetPosition-${suffix}`);
        if (btnSetPosition) {
            btnSetPosition.addEventListener('click', async () => {
                const x = parseFloat(document.getElementById(`camX-${suffix}`).value);
                const y = parseFloat(document.getElementById(`camY-${suffix}`).value);
                const z = parseFloat(document.getElementById(`camZ-${suffix}`).value);
                const tx = parseFloat(document.getElementById(`targetX-${suffix}`).value);
                const ty = parseFloat(document.getElementById(`targetY-${suffix}`).value);
                const tz = parseFloat(document.getElementById(`targetZ-${suffix}`).value);

                await this.api.setCameraLookat([x, y, z], [tx, ty, tz]);
            });
        }

        // 타겟 설정
        const btnSetTarget = document.getElementById(`btnSetTarget-${suffix}`);
        if (btnSetTarget) {
            btnSetTarget.addEventListener('click', async () => {
                const pose = await this.api.getCameraPose();
                if (pose.success) {
                    const eye = pose.data.eye || pose.data.position;
                    const tx = parseFloat(document.getElementById(`targetX-${suffix}`).value);
                    const ty = parseFloat(document.getElementById(`targetY-${suffix}`).value);
                    const tz = parseFloat(document.getElementById(`targetZ-${suffix}`).value);

                    await this.api.setCameraLookat(eye, [tx, ty, tz]);
                }
            });
        }
    }

    async loadCameras() {
        try {
            const result = await this.api.listCameras();
            if (!result.success) return;

            this.cameras = result.data;

            // VLA만 카메라 드롭다운 업데이트
            ['vla'].forEach(suffix => {
                const select = document.getElementById(`cameraSelect-${suffix}`);
                if (select) {
                    select.innerHTML = this.cameras.map(cam => `
                        <option value="${cam.id}" ${cam.active ? 'selected' : ''}>
                            📷 ${cam.name} (${cam.resolution || 'N/A'})
                        </option>
                    `).join('');
                }
            });

            console.log('카메라 목록 로드:', this.cameras.length, '개');
        } catch (e) {
            console.error('카메라 목록 로드 실패:', e);
        }
    }

    // ===== 물체 제어 =====

    setupObjectControls() {
        ['vla'].forEach(suffix => {
            this.setupObjectControlsForSection(suffix);
        });
    }

    setupObjectControlsForSection(suffix) {
        const btnSpawnObject = document.getElementById(`btnSpawnObject-${suffix}`);
        if (btnSpawnObject) {
            btnSpawnObject.addEventListener('click', async () => {
                const objType = document.getElementById(`objectType-${suffix}`).value;
                const x = parseFloat(document.getElementById(`objX-${suffix}`).value);
                const y = parseFloat(document.getElementById(`objY-${suffix}`).value);
                const z = parseFloat(document.getElementById(`objZ-${suffix}`).value);

                const result = await this.api.spawnObject(
                    objType,
                    [x, y, z],
                    [1, 0, 0, 0]
                );

                if (result.success) {
                    await this.refreshObjects();
                } else {
                    alert('물체 생성 실패: ' + (result.error || '알 수 없는 오류'));
                }
            });
        }

        const btnRefreshObjects = document.getElementById(`btnRefreshObjects-${suffix}`);
        if (btnRefreshObjects) {
            btnRefreshObjects.addEventListener('click', () => {
                this.refreshObjects();
            });
        }
    }

    async refreshObjects() {
        const result = await this.api.listObjects();
        if (!result.success) return;

        this.objects = result.data;

        // VLA만 업데이트
        ['vla'].forEach(suffix => {
            const listEl = document.getElementById(`objectList-${suffix}`);
            const targetSelect = document.getElementById(`targetObject-${suffix}`);

            if (listEl) {
                if (this.objects.length === 0) {
                    listEl.innerHTML = '<p class="empty-message">물체가 없습니다.</p>';
                } else {
                    listEl.innerHTML = this.objects.map(obj => `
                        <div class="object-item">
                            <span>📦 ${obj.name || obj.id}</span>
                            <div class="object-actions">
                                <button class="btn btn-secondary btn-sm" onclick="app.transformObject('${obj.id}')" title="택배 박스로 변환">🔄</button>
                                <button class="btn btn-danger btn-sm" onclick="app.deleteObject('${obj.id}')" title="삭제">🗑️</button>
                            </div>
                        </div>
                    `).join('');
                }
            }

            if (targetSelect) {
                targetSelect.innerHTML = '<option value="">물체 선택...</option>' +
                    this.objects.map(obj => `<option value="${obj.id}">${obj.name || obj.id}</option>`).join('');
            }
        });
    }

    async deleteObject(objectId) {
        const result = await this.api.deleteObject(objectId);
        if (result.success) {
            await this.refreshObjects();
        }
    }

    async transformObject(objectId) {
        const result = await this.api.transformObject(objectId);
        if (result.success) {
            console.log('물체 변환 완료:', objectId);
            await this.refreshObjects();
        } else {
            alert('물체 변환 실패');
        }
    }

    // ===== 재질 제어 =====

    setupMaterialControls() {
        ['vla'].forEach(suffix => {
            this.setupMaterialControlsForSection(suffix);
        });
    }

    setupMaterialControlsForSection(suffix) {
        const roughness = document.getElementById(`roughness-${suffix}`);
        const metallic = document.getElementById(`metallic-${suffix}`);

        if (roughness) {
            roughness.addEventListener('input', () => {
                document.getElementById(`roughnessValue-${suffix}`).textContent = roughness.value;
            });
        }

        if (metallic) {
            metallic.addEventListener('input', () => {
                document.getElementById(`metallicValue-${suffix}`).textContent = metallic.value;
            });
        }

        // 재질 생성
        const btnCreateMaterial = document.getElementById(`btnCreateMaterial-${suffix}`);
        if (btnCreateMaterial) {
            btnCreateMaterial.addEventListener('click', async () => {
                const name = document.getElementById(`materialName-${suffix}`).value || 'custom_material';
                const colorHex = document.getElementById(`materialColor-${suffix}`).value;
                const color = this.hexToRgb(colorHex);
                const rough = parseFloat(document.getElementById(`roughness-${suffix}`).value);
                const metal = parseFloat(document.getElementById(`metallic-${suffix}`).value);

                const result = await this.api.createMaterial(name, color, rough, metal);
                if (result.success) {
                    await this.refreshMaterials();
                }
            });
        }

        // 재질 바인딩
        const btnBindMaterial = document.getElementById(`btnBindMaterial-${suffix}`);
        if (btnBindMaterial) {
            btnBindMaterial.addEventListener('click', async () => {
                const objectId = document.getElementById(`targetObject-${suffix}`).value;
                const materialId = document.getElementById(`selectedMaterial-${suffix}`).value;

                if (!objectId || !materialId) {
                    alert('물체와 재질을 선택하세요.');
                    return;
                }

                const result = await this.api.bindMaterial(objectId, materialId);
                if (!result.success) {
                    alert('재질 적용 실패');
                }
            });
        }
    }

    async loadPresetMaterials() {
        const result = await this.api.getPresetMaterials();
        if (!result.success) return;

        ['vla'].forEach(suffix => {
            const container = document.getElementById(`presetMaterials-${suffix}`);
            if (!container) return;

            container.innerHTML = result.data.map(preset => {
                const [r, g, b] = preset.color.map(v => Math.round(v * 255));
                return `<button class="preset-btn" 
                    style="background: rgb(${r}, ${g}, ${b})"
                    data-color="${JSON.stringify(preset.color)}"
                    data-roughness="${preset.roughness}"
                    data-metallic="${preset.metallic}"
                    title="${preset.name}"></button>`;
            }).join('');

            // 프리셋 클릭 이벤트
            container.querySelectorAll('.preset-btn').forEach(btn => {
                btn.addEventListener('click', async () => {
                    const color = JSON.parse(btn.dataset.color);
                    const roughness = parseFloat(btn.dataset.roughness);
                    const metallic = parseFloat(btn.dataset.metallic);

                    document.getElementById(`materialColor-${suffix}`).value = this.rgbToHex(color);
                    document.getElementById(`roughness-${suffix}`).value = roughness;
                    document.getElementById(`metallic-${suffix}`).value = metallic;
                    document.getElementById(`roughnessValue-${suffix}`).textContent = roughness;
                    document.getElementById(`metallicValue-${suffix}`).textContent = metallic;
                });
            });
        });
    }

    async refreshMaterials() {
        const result = await this.api.listMaterials();
        if (!result.success) return;

        this.materials = result.data;

        ['vla'].forEach(suffix => {
            const select = document.getElementById(`selectedMaterial-${suffix}`);
            if (select) {
                select.innerHTML = '<option value="">재질 선택...</option>' +
                    this.materials.map(mat => `<option value="${mat.id}">${mat.name}</option>`).join('');
            }
        });
    }

    // ===== 로봇 제어 =====

    setupRobotControls() {
        // 텔레오프 시작
        const btnStartTeleop = document.getElementById('btnStartTeleop');
        const btnStopTeleop = document.getElementById('btnStopTeleop');

        if (btnStartTeleop) {
            btnStartTeleop.addEventListener('click', async () => {
                const mode = document.getElementById('teleopMode').value;
                btnStartTeleop.disabled = true;

                const result = await this.api.startTeleop(mode);
                if (result.success) {
                    this.teleopRunning = true;
                    btnStartTeleop.style.display = 'none';
                    btnStopTeleop.style.display = 'block';
                    document.getElementById('teleopStatus').textContent = mode === 'ros2' ? 'ROS2 실행 중' : 'Demo 실행 중';
                    document.getElementById('teleopStatus').className = 'teleop-status-value running';

                    if (mode === 'ros2') {
                        document.getElementById('ros2StatusBox').style.display = '';
                    }

                    // 관절 상태 폴링 시작
                    this.startJointPolling();
                } else {
                    btnStartTeleop.disabled = false;
                    alert('텔레오프 시작 실패');
                }
            });
        }

        if (btnStopTeleop) {
            btnStopTeleop.addEventListener('click', async () => {
                const result = await this.api.stopTeleop();
                if (result.success) {
                    this.teleopRunning = false;
                    btnStopTeleop.style.display = 'none';
                    btnStartTeleop.style.display = 'block';
                    btnStartTeleop.disabled = false;
                    document.getElementById('teleopStatus').textContent = '대기 중';
                    document.getElementById('teleopStatus').className = 'teleop-status-value';
                    document.getElementById('ros2StatusBox').style.display = 'none';

                    // 관절 상태 폴링 중지
                    this.stopJointPolling();
                }
            });
        }

    }

    async loadRobotInfo() {
        try {
            const result = await this.api.getRobotInfo();
            if (result.success && result.data) {
                this.robotAvailable = true;
                const info = result.data;
                const infoBox = document.getElementById('robotInfoBox');
                if (infoBox) {
                    infoBox.innerHTML = `
                        <div class="robot-info-item"><strong>이름:</strong> ${info.name}</div>
                        <div class="robot-info-item"><strong>관절 수:</strong> ${info.num_joints}</div>
                        <div class="robot-info-item"><strong>관절:</strong> ${info.joint_names.join(', ')}</div>
                    `;
                }
                console.log('로봇 정보 로드 완료:', info.name);
            } else {
                const infoBox = document.getElementById('robotInfoBox');
                if (infoBox) {
                    infoBox.innerHTML = '<p class="empty-message">로봇이 없습니다.</p>';
                }
            }
        } catch (e) {
            console.error('로봇 정보 로드 실패:', e);
        }
    }

    startJointPolling() {
        this.stopJointPolling();  // 기존 폴링 중지
        this.jointPollingInterval = setInterval(async () => {
            try {
                // 관절 상태 + 텔레오프 상태를 병렬 요청 (ZMQ 경합 시간 감소)
                const [result, status] = await Promise.all([
                    this.api.getRobotJoints(),
                    this.api.getTeleopStatus()
                ]);

                if (result.success && result.data) {
                    this.updateJointDisplay(result.data.positions);
                }

                if (status.success && status.data) {
                    if (status.data.mode === 'ros2') {
                        const ros2El = document.getElementById('ros2Status');
                        if (ros2El) {
                            ros2El.textContent = status.data.connected ? '연결됨' : '연결 대기 중...';
                            ros2El.className = 'teleop-status-value ' + (status.data.connected ? 'connected' : 'waiting');
                        }
                    }
                }
            } catch (e) {
                // 폴링 오류 무시
            }
        }, 250);  // 4Hz 폴링 (ZMQ 경합 감소)
    }

    stopJointPolling() {
        if (this.jointPollingInterval) {
            clearInterval(this.jointPollingInterval);
            this.jointPollingInterval = null;
        }
    }

    updateJointDisplay(positions) {
        if (!positions) return;

        // 각 관절의 리미트 정의
        const limits = [
            [-3.14159, 3.14159],  // joint1
            [-1.5, 1.5],          // joint2
            [-1.5, 1.4],          // joint3
            [-1.7, 1.97],         // joint4
            [-0.01, 0.019],       // gripper
        ];

        for (let i = 0; i < Math.min(positions.length, 5); i++) {
            const val = positions[i];
            const valEl = document.getElementById(`jointVal-${i}`);
            const barEl = document.getElementById(`jointBar-${i}`);

            if (valEl) {
                valEl.textContent = val.toFixed(3);
            }
            if (barEl) {
                const [min, max] = limits[i];
                const pct = ((val - min) / (max - min)) * 100;
                barEl.style.width = Math.max(0, Math.min(100, pct)) + '%';
            }

        }
    }

    // ===== RL (Train / Inference) 제어 =====

    setupRLControls() {
        // Train 시작
        const btnStartTrain = document.getElementById('btnStartTrain');
        const btnStopTrain = document.getElementById('btnStopTrain');
        if (btnStartTrain) {
            btnStartTrain.addEventListener('click', async () => {
                const numEnvs = parseInt(document.getElementById('numEnvs').value) || 64;
                btnStartTrain.disabled = true;
                this.updateRLStatus('train', '시작 중...', 'running');

                const result = await this.api.startTrain(numEnvs);
                if (result.success) {
                    this.updateRLStatus('train', '학습 중', 'running');
                    btnStartTrain.style.display = 'none';
                    btnStopTrain.style.display = 'block';
                } else {
                    this.updateRLStatus('train', '시작 실패', 'error');
                    btnStartTrain.disabled = false;
                }
            });
        }

        if (btnStopTrain) {
            btnStopTrain.addEventListener('click', async () => {
                const result = await this.api.stopTrain();
                this.updateRLStatus('train', '대기 중', '');
                btnStopTrain.style.display = 'none';
                const btnStart = document.getElementById('btnStartTrain');
                btnStart.style.display = 'block';
                btnStart.disabled = false;
            });
        }

        // Inference 시작
        const btnStartInference = document.getElementById('btnStartInference');
        const btnStopInference = document.getElementById('btnStopInference');
        if (btnStartInference) {
            btnStartInference.addEventListener('click', async () => {
                btnStartInference.disabled = true;
                this.updateRLStatus('inference', '시작 중...', 'running');

                const result = await this.api.startInference();
                if (result.success) {
                    this.updateRLStatus('inference', '추론 중', 'running');
                    btnStartInference.style.display = 'none';
                    btnStopInference.style.display = 'block';
                } else {
                    this.updateRLStatus('inference', '시작 실패', 'error');
                    btnStartInference.disabled = false;
                }
            });
        }

        if (btnStopInference) {
            btnStopInference.addEventListener('click', async () => {
                const result = await this.api.stopInference();
                this.updateRLStatus('inference', '대기 중', '');
                btnStopInference.style.display = 'none';
                const btnStart = document.getElementById('btnStartInference');
                btnStart.style.display = 'block';
                btnStart.disabled = false;
            });
        }
    }

    updateRLStatus(mode, text, statusClass) {
        const statusEl = document.getElementById(`${mode}Status`);
        if (statusEl) {
            statusEl.textContent = text;
            statusEl.className = 'rl-status-value';
            if (statusClass) {
                statusEl.classList.add(statusClass);
            }
        }
    }

    // ===== 미리보기 제어 =====

    setupPreviewControls() {
        ['vla', 'rl'].forEach(suffix => {
            this.setupPreviewControlsForSection(suffix);
        });
    }

    setupPreviewControlsForSection(suffix) {
        const btnSnapshot = document.getElementById(`btnSnapshot-${suffix}`);
        if (btnSnapshot) {
            btnSnapshot.addEventListener('click', async () => {
                const frameUrl = await this.api.getCameraFrame();
                const a = document.createElement('a');
                a.href = frameUrl;
                a.download = `snapshot_${suffix}_${Date.now()}.jpg`;
                a.click();
            });
        }

        const btnToggleStream = document.getElementById(`btnToggleStream-${suffix}`);
        if (btnToggleStream) {
            btnToggleStream.addEventListener('click', () => {
                this.isStreaming = !this.isStreaming;

                // 모든 탭의 버튼 상태 동기화
                ['vla', 'rl'].forEach(s => {
                    const btn = document.getElementById(`btnToggleStream-${s}`);
                    if (btn) {
                        btn.textContent = this.isStreaming ? '⏸️ 스트림 일시정지' : '▶️ 스트림 재생';
                    }
                });

                if (this.isStreaming) {
                    this.startStreaming();
                } else {
                    this.stopStreaming();
                }
            });
        }
    }

    startStreaming() {
        // fetch 체이닝 방식: 이전 프레임 로드 완료 후 다음 프레임 요청
        // setInterval + img.src 방식은 응답이 느릴 때 브라우저가 이전 로드를 취소하여 프레임이 표시되지 않음
        this._streamingActive = true;

        const fetchNextFrame = async () => {
            if (!this._streamingActive) return;

            try {
                const response = await fetch(`${this.api.baseUrl}/api/camera/frame?t=${Date.now()}`);
                if (!response.ok) throw new Error('Frame fetch failed');

                const blob = await response.blob();
                const url = URL.createObjectURL(blob);

                ['vla', 'rl'].forEach(suffix => {
                    const preview = document.getElementById(`previewImage-${suffix}`);
                    const overlay = document.getElementById(`previewOverlay-${suffix}`);
                    if (preview) {
                        // 이전 Blob URL 해제
                        if (preview._blobUrl) URL.revokeObjectURL(preview._blobUrl);
                        preview._blobUrl = url;
                        preview.src = url;
                    }
                    if (overlay) overlay.classList.add('hidden');
                });
            } catch (e) {
                ['vla', 'rl'].forEach(suffix => {
                    const overlay = document.getElementById(`previewOverlay-${suffix}`);
                    if (overlay) overlay.classList.remove('hidden');
                });
                // 에러 시 짧은 대기 후 재시도
                await new Promise(r => setTimeout(r, 100));
            }

            // 다음 프레임 즉시 요청 (rAF는 ~16ms 대기하므로 setTimeout(0) 사용)
            if (this._streamingActive) {
                setTimeout(fetchNextFrame, 0);
            }
        };

        fetchNextFrame();
    }

    stopStreaming() {
        this._streamingActive = false;
        if (this.streamingInterval) {
            clearInterval(this.streamingInterval);
            this.streamingInterval = null;
        }
    }

    // ===== 초기 데이터 로드 =====

    async loadInitialData() {
        try {
            await this.loadPresetMaterials();
            await this.refreshObjects();
            await this.refreshMaterials();
        } catch (e) {
            console.error('초기 데이터 로드 실패:', e);
        }
    }

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

    // ===== 유틸리티 =====

    hexToRgb(hex) {
        const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
        return result ? [
            parseInt(result[1], 16) / 255,
            parseInt(result[2], 16) / 255,
            parseInt(result[3], 16) / 255
        ] : [1, 0, 0];
    }

    rgbToHex(rgb) {
        const [r, g, b] = rgb.map(v => Math.round(v * 255));
        return `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`;
    }
}

// 전역 앱 접근용
window.app = null;
document.addEventListener('DOMContentLoaded', () => {
    window.app = new App();
    window.app.init();
});
