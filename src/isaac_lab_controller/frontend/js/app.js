/**
 * IsaacLab Controller 메인 앱
 */


class App {
    constructor() {
        this.api = window.api;
        this.isStreaming = true;
        this.streamingInterval = null;
        this.objects = [];
        this.materials = [];
        this.cameras = [];
    }

    async init() {
        this.setupTabNavigation();
        this.setupCameraControls();
        this.setupObjectControls();
        this.setupMaterialControls();
        this.setupPreviewControls();

        await this.loadInitialData();
        await this.loadCameras();  // 카메라 목록 로드
        this.startStreaming();
        this.updateConnectionStatus(true);
    }

    // ===== 탭 네비게이션 =====

    setupTabNavigation() {
        const tabBtns = document.querySelectorAll('.tab-btn');
        const tabContents = document.querySelectorAll('.tab-content');

        tabBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                const tabId = btn.dataset.tab;

                tabBtns.forEach(b => b.classList.remove('active'));
                tabContents.forEach(c => c.classList.remove('active'));

                btn.classList.add('active');
                document.getElementById(`tab-${tabId}`).classList.add('active');
            });
        });
    }

    // ===== 카메라 제어 =====

    setupCameraControls() {
        // 카메라 선택 드롭다운
        const cameraSelect = document.getElementById('cameraSelect');
        cameraSelect.addEventListener('change', async () => {
            if (cameraSelect.value) {
                const result = await this.api.selectCamera(cameraSelect.value);
                if (result.success) {
                    console.log('카메라 변경됨:', cameraSelect.value);
                }
            }
        });

        // 카메라 새로고침 버튼
        const btnRefreshCameras = document.getElementById('btnRefreshCameras');
        if (btnRefreshCameras) {
            btnRefreshCameras.addEventListener('click', () => this.loadCameras());
        }

        // Orbit 슬라이더
        const azimuth = document.getElementById('azimuth');
        const elevation = document.getElementById('elevation');
        const distance = document.getElementById('distance');

        const updateOrbit = () => {
            document.getElementById('azimuthValue').textContent = azimuth.value;
            document.getElementById('elevationValue').textContent = elevation.value;
            document.getElementById('distanceValue').textContent = distance.value;

            this.api.setCameraOrbit(
                parseFloat(azimuth.value),
                parseFloat(elevation.value),
                parseFloat(distance.value)
            );
        };

        azimuth.addEventListener('input', updateOrbit);
        elevation.addEventListener('input', updateOrbit);
        distance.addEventListener('input', updateOrbit);

        // 위치 직접 입력
        document.getElementById('btnSetPosition').addEventListener('click', async () => {
            const x = parseFloat(document.getElementById('camX').value);
            const y = parseFloat(document.getElementById('camY').value);
            const z = parseFloat(document.getElementById('camZ').value);
            const tx = parseFloat(document.getElementById('targetX').value);
            const ty = parseFloat(document.getElementById('targetY').value);
            const tz = parseFloat(document.getElementById('targetZ').value);

            await this.api.setCameraLookat([x, y, z], [tx, ty, tz]);
        });

        // 타겟 설정
        document.getElementById('btnSetTarget').addEventListener('click', async () => {
            const pose = await this.api.getCameraPose();
            if (pose.success) {
                const eye = pose.data.eye || pose.data.position;
                const tx = parseFloat(document.getElementById('targetX').value);
                const ty = parseFloat(document.getElementById('targetY').value);
                const tz = parseFloat(document.getElementById('targetZ').value);

                await this.api.setCameraLookat(eye, [tx, ty, tz]);
            }
        });
    }

    async loadCameras() {
        try {
            const result = await this.api.listCameras();
            if (!result.success) return;

            this.cameras = result.data;
            const select = document.getElementById('cameraSelect');

            select.innerHTML = this.cameras.map(cam => `
                <option value="${cam.id}" ${cam.active ? 'selected' : ''}>
                    📷 ${cam.name} (${cam.resolution || 'N/A'})
                </option>
            `).join('');

            console.log('카메라 목록 로드:', this.cameras.length, '개');
        } catch (e) {
            console.error('카메라 목록 로드 실패:', e);
        }
    }

    // ===== 물체 제어 =====

    setupObjectControls() {
        document.getElementById('btnSpawnObject').addEventListener('click', async () => {
            const objType = document.getElementById('objectType').value;
            const x = parseFloat(document.getElementById('objX').value);
            const y = parseFloat(document.getElementById('objY').value);
            const z = parseFloat(document.getElementById('objZ').value);

            const result = await this.api.spawnObject(
                objType,
                [x, y, z],
                [1, 0, 0, 0]  // 기본 회전
            );

            if (result.success) {
                await this.refreshObjects();
            } else {
                alert('물체 생성 실패: ' + (result.error || '알 수 없는 오류'));
            }
        });

        document.getElementById('btnRefreshObjects').addEventListener('click', () => {
            this.refreshObjects();
        });
    }

    async refreshObjects() {
        const result = await this.api.listObjects();
        if (!result.success) return;

        this.objects = result.data;
        const listEl = document.getElementById('objectList');
        const targetSelect = document.getElementById('targetObject');

        // 물체 목록 업데이트
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

        // 드롭다운 업데이트
        targetSelect.innerHTML = '<option value="">물체 선택...</option>' +
            this.objects.map(obj => `<option value="${obj.id}">${obj.name || obj.id}</option>`).join('');
    }

    async deleteObject(objectId) {
        const result = await this.api.deleteObject(objectId);
        if (result.success) {
            await this.refreshObjects();
        }
    }

    // ===== 재질 제어 =====

    setupMaterialControls() {
        // 슬라이더 값 표시
        const roughness = document.getElementById('roughness');
        const metallic = document.getElementById('metallic');

        roughness.addEventListener('input', () => {
            document.getElementById('roughnessValue').textContent = roughness.value;
        });

        metallic.addEventListener('input', () => {
            document.getElementById('metallicValue').textContent = metallic.value;
        });

        // 재질 생성
        document.getElementById('btnCreateMaterial').addEventListener('click', async () => {
            const name = document.getElementById('materialName').value || 'custom_material';
            const colorHex = document.getElementById('materialColor').value;
            const color = this.hexToRgb(colorHex);
            const rough = parseFloat(roughness.value);
            const metal = parseFloat(metallic.value);

            const result = await this.api.createMaterial(name, color, rough, metal);
            if (result.success) {
                await this.refreshMaterials();
            }
        });

        // 재질 바인딩
        document.getElementById('btnBindMaterial').addEventListener('click', async () => {
            const objectId = document.getElementById('targetObject').value;
            const materialId = document.getElementById('selectedMaterial').value;

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

    async loadPresetMaterials() {
        const result = await this.api.getPresetMaterials();
        if (!result.success) return;

        const container = document.getElementById('presetMaterials');
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

                // 폼 업데이트
                document.getElementById('materialColor').value = this.rgbToHex(color);
                document.getElementById('roughness').value = roughness;
                document.getElementById('metallic').value = metallic;
                document.getElementById('roughnessValue').textContent = roughness;
                document.getElementById('metallicValue').textContent = metallic;
            });
        });
    }

    async refreshMaterials() {
        const result = await this.api.listMaterials();
        if (!result.success) return;

        this.materials = result.data;
        const select = document.getElementById('selectedMaterial');

        select.innerHTML = '<option value="">재질 선택...</option>' +
            this.materials.map(mat => `<option value="${mat.id}">${mat.name}</option>`).join('');
    }

    // ===== 미리보기 제어 =====

    setupPreviewControls() {
        document.getElementById('btnSnapshot').addEventListener('click', async () => {
            const frameUrl = await this.api.getCameraFrame();
            const a = document.createElement('a');
            a.href = frameUrl;
            a.download = `snapshot_${Date.now()}.jpg`;
            a.click();
        });

        document.getElementById('btnToggleStream').addEventListener('click', () => {
            this.isStreaming = !this.isStreaming;
            const btn = document.getElementById('btnToggleStream');

            if (this.isStreaming) {
                btn.textContent = '⏸️ 스트림 일시정지';
                this.startStreaming();
            } else {
                btn.textContent = '▶️ 스트림 재생';
                this.stopStreaming();
            }
        });
    }

    startStreaming() {
        const preview = document.getElementById('previewImage');
        const overlay = document.getElementById('previewOverlay');

        // 폴링 방식 스트리밍 (WebSocket 대안)
        this.streamingInterval = setInterval(async () => {
            if (!this.isStreaming) return;

            try {
                const frameUrl = await this.api.getCameraFrame();
                preview.src = frameUrl;
                overlay.classList.add('hidden');
            } catch (e) {
                overlay.classList.remove('hidden');
            }
        }, 100);  // 10 FPS
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
