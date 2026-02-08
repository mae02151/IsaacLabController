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

        // X/Y 패닝 버튼
        const panStep = 0.2;
        const panBtns = {
            [`btnPanLeft-${suffix}`]: [-panStep, 0],
            [`btnPanRight-${suffix}`]: [panStep, 0],
            [`btnPanUp-${suffix}`]: [0, panStep],
            [`btnPanDown-${suffix}`]: [0, -panStep],
        };

        Object.entries(panBtns).forEach(([btnId, [dx, dy]]) => {
            const btn = document.getElementById(btnId);
            if (btn) {
                btn.addEventListener('click', () => this.panCamera(dx, dy));
            }
        });
    }

    async panCamera(dx, dy) {
        try {
            const poseResult = await this.api.getCameraPose();
            if (!poseResult.success) return;

            const pose = poseResult.data;
            const eye = pose.eye || pose.position;
            const target = pose.target || [0, 0, 0];

            // 카메라 로컬 축 계산 (top-down 호환)
            const forward = this.normalize(this.sub(target, eye));
            const { right, camUp } = this.getCameraAxes(forward);

            // eye와 target을 동일하게 이동 (패닝)
            const newEye = [
                eye[0] + dx * right[0] + dy * camUp[0],
                eye[1] + dx * right[1] + dy * camUp[1],
                eye[2] + dx * right[2] + dy * camUp[2]
            ];
            const newTarget = [
                target[0] + dx * right[0] + dy * camUp[0],
                target[1] + dx * right[1] + dy * camUp[1],
                target[2] + dx * right[2] + dy * camUp[2]
            ];

            await this.api.setCameraLookat(newEye, newTarget);

            // UI 입력 필드 동기화
            ['vla', 'rl'].forEach(suffix => {
                const camX = document.getElementById(`camX-${suffix}`);
                const camY = document.getElementById(`camY-${suffix}`);
                const camZ = document.getElementById(`camZ-${suffix}`);
                const targetX = document.getElementById(`targetX-${suffix}`);
                const targetY = document.getElementById(`targetY-${suffix}`);
                const targetZ = document.getElementById(`targetZ-${suffix}`);

                if (camX) camX.value = newEye[0].toFixed(1);
                if (camY) camY.value = newEye[1].toFixed(1);
                if (camZ) camZ.value = newEye[2].toFixed(1);
                if (targetX) targetX.value = newTarget[0].toFixed(1);
                if (targetY) targetY.value = newTarget[1].toFixed(1);
                if (targetZ) targetZ.value = newTarget[2].toFixed(1);
            });
        } catch (e) {
            console.error('카메라 패닝 오류:', e);
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

        // 클릭-투-스폰: 프리뷰 이미지 클릭 시 해당 3D 위치에 물체 생성
        const previewImage = document.getElementById(`previewImage-${suffix}`);
        if (previewImage) {
            previewImage.style.cursor = 'crosshair';
            previewImage.addEventListener('click', (e) => this.onPreviewClick(e, suffix));
        }
    }

    async onPreviewClick(event, suffix) {
        const img = event.target;
        const rect = img.getBoundingClientRect();

        // 이미지 상의 클릭 좌표 (0~1 정규화)
        const clickX = (event.clientX - rect.left) / rect.width;
        const clickY = (event.clientY - rect.top) / rect.height;

        try {
            // 카메라 intrinsics + pose 동시 조회
            const [intrResult, poseResult] = await Promise.all([
                this.api.getCameraIntrinsics(),
                this.api.getCameraPose()
            ]);

            if (!intrResult.success || !poseResult.success) {
                console.error('카메라 정보 조회 실패');
                return;
            }

            const intr = intrResult.data;
            const pose = poseResult.data;
            const eye = pose.eye || pose.position;
            const target = pose.target || [0, 0, 0];

            // 3D 위치 계산 (ray-ground intersection)
            const worldPos = this.screenToWorld(clickX, clickY, eye, target, intr);

            if (!worldPos) {
                console.warn('바닥면과 교차점 없음 (카메라가 위를 보고 있음)');
                return;
            }

            // 물체 스폰
            const objType = document.getElementById(`objectType-${suffix}`)?.value || 'box';
            const result = await this.api.spawnObject(
                objType,
                [worldPos[0], worldPos[1], worldPos[2]],
                [1, 0, 0, 0]
            );

            if (result.success) {
                console.log(`클릭 스폰: (${worldPos[0].toFixed(2)}, ${worldPos[1].toFixed(2)}, ${worldPos[2].toFixed(2)})`);
                await this.refreshObjects();
            }
        } catch (e) {
            console.error('클릭 스폰 오류:', e);
        }
    }

    screenToWorld(u, v, eye, target, intrinsics) {
        // NDC 좌표 계산 (-1 ~ 1)
        // CSS scaleY(-1) 플립 반영: 화면 아래 = +Y
        const ndcX = (u - 0.5) * 2;
        const ndcY = (v - 0.5) * 2;  // CSS 플립 보정 (down=+Y)

        // 카메라 벡터 계산 (top-down 호환)
        const forward = this.normalize(this.sub(target, eye));
        const { right, camUp } = this.getCameraAxes(forward);

        // FOV 계산 (intrinsics 기반)
        const focalLength = intrinsics.focal_length || 24.0;
        const hAperture = intrinsics.horizontal_aperture || 20.955;
        const width = intrinsics.width || 640;
        const height = intrinsics.height || 480;
        const vAperture = hAperture * height / width;

        const halfFovX = Math.atan(hAperture / (2 * focalLength));
        const halfFovY = Math.atan(vAperture / (2 * focalLength));

        // 레이 방향 계산
        const tanX = ndcX * Math.tan(halfFovX);
        const tanY = ndcY * Math.tan(halfFovY);

        const dir = this.normalize([
            forward[0] + tanX * right[0] + tanY * camUp[0],
            forward[1] + tanX * right[1] + tanY * camUp[1],
            forward[2] + tanX * right[2] + tanY * camUp[2]
        ]);

        // 바닥면(z = spawnHeight)과 교차
        const spawnHeight = 0.05;
        if (Math.abs(dir[2]) < 1e-6) return null;  // 수평 레이

        const t = (spawnHeight - eye[2]) / dir[2];
        if (t < 0) return null;  // 뒤쪽 교차

        return [
            eye[0] + t * dir[0],
            eye[1] + t * dir[1],
            spawnHeight
        ];
    }

    // 카메라 로컬 축 계산 (top-down 뷰 호환)
    getCameraAxes(forward) {
        let worldUp = [0, 0, 1];
        // top-down: forward가 worldUp과 거의 평행 → 대체 up 벡터 사용
        if (Math.abs(forward[2]) > 0.99) {
            worldUp = [0, 1, 0];
        }
        const right = this.normalize(this.cross(forward, worldUp));
        const camUp = this.cross(right, forward);
        return { right, camUp };
    }

    // 벡터 유틸리티
    sub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
    cross(a, b) {
        return [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]
        ];
    }
    normalize(v) {
        const len = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        return len > 0 ? [v[0] / len, v[1] / len, v[2] / len] : v;
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
