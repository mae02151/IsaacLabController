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
    }

    async init() {
        this.setupMainTabNavigation();
        this.setupTabNavigation();
        this.setupCameraControls();
        this.setupObjectControls();
        this.setupMaterialControls();
        this.setupPreviewControls();

        await this.loadInitialData();
        await this.loadCameras();
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
        // VLA 탭용
        this.setupTabsForSection('vla');
        // Reinforcement 탭용
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
        // VLA와 RL 모두 설정
        ['vla', 'rl'].forEach(suffix => {
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

            // VLA와 RL 모두 업데이트
            ['vla', 'rl'].forEach(suffix => {
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
        ['vla', 'rl'].forEach(suffix => {
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

        // VLA와 RL 모두 업데이트
        ['vla', 'rl'].forEach(suffix => {
            const listEl = document.getElementById(`objectList-${suffix}`);
            const targetSelect = document.getElementById(`targetObject-${suffix}`);

            if (listEl) {
                if (this.objects.length === 0) {
                    listEl.innerHTML = '<p class="empty-message">물체가 없습니다.</p>';
                } else {
                    listEl.innerHTML = this.objects.map(obj => `
                        <div class="object-item">
                            <span>📦 ${obj.name || obj.id}</span>
                            <button class="btn btn-danger" onclick="app.deleteObject('${obj.id}')">🗑️</button>
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

    // ===== 재질 제어 =====

    setupMaterialControls() {
        ['vla', 'rl'].forEach(suffix => {
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

        ['vla', 'rl'].forEach(suffix => {
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

        ['vla', 'rl'].forEach(suffix => {
            const select = document.getElementById(`selectedMaterial-${suffix}`);
            if (select) {
                select.innerHTML = '<option value="">재질 선택...</option>' +
                    this.materials.map(mat => `<option value="${mat.id}">${mat.name}</option>`).join('');
            }
        });
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
        // 폴링 방식 스트리밍 (WebSocket 대안)
        this.streamingInterval = setInterval(async () => {
            if (!this.isStreaming) return;

            try {
                const frameUrl = await this.api.getCameraFrame();

                // 모든 탭의 프리뷰 업데이트
                ['vla', 'rl'].forEach(suffix => {
                    const preview = document.getElementById(`previewImage-${suffix}`);
                    const overlay = document.getElementById(`previewOverlay-${suffix}`);
                    if (preview) preview.src = frameUrl;
                    if (overlay) overlay.classList.add('hidden');
                });
            } catch (e) {
                ['vla', 'rl'].forEach(suffix => {
                    const overlay = document.getElementById(`previewOverlay-${suffix}`);
                    if (overlay) overlay.classList.remove('hidden');
                });
            }
        }, 30);  // ~33 FPS (테스트용 고속)
    }

    stopStreaming() {
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
