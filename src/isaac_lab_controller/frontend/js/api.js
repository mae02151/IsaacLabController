/**
 * API 통신 모듈
 */

class IsaacLabAPI {
    constructor(baseUrl = '') {
        this.baseUrl = baseUrl || window.location.origin;
        this.wsStream = null;
        this.wsControl = null;
    }

    // ===== HTTP API =====

    async get(endpoint) {
        const response = await fetch(`${this.baseUrl}${endpoint}`);
        return response.json();
    }

    async post(endpoint, data) {
        const response = await fetch(`${this.baseUrl}${endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        return response.json();
    }

    async delete(endpoint) {
        const response = await fetch(`${this.baseUrl}${endpoint}`, {
            method: 'DELETE'
        });
        return response.json();
    }

    async patch(endpoint, data) {
        const response = await fetch(`${this.baseUrl}${endpoint}`, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        return response.json();
    }

    // ===== Camera API =====

    async listCameras() {
        return this.get('/api/camera/list');
    }

    async selectCamera(cameraId) {
        return this.post('/api/camera/select', { camera_id: cameraId });
    }

    async getActiveCamera() {
        return this.get('/api/camera/active');
    }

    async getCameraPose() {
        return this.get('/api/camera/pose');
    }

    async setCameraPose(position, orientation, convention = 'ros') {
        return this.post('/api/camera/pose', { position, orientation, convention });
    }

    async setCameraLookat(eye, target) {
        return this.post('/api/camera/lookat', { eye, target });
    }

    async setCameraOrbit(azimuth, elevation, distance, target = null) {
        return this.post('/api/camera/orbit', { azimuth, elevation, distance, target });
    }

    async getCameraFrame() {
        return `${this.baseUrl}/api/camera/frame?t=${Date.now()}`;
    }

    // ===== Objects API =====

    async listObjects() {
        return this.get('/api/objects/');
    }

    async getObjectTypes() {
        return this.get('/api/objects/types');
    }

    async spawnObject(objType, position, rotation, name = null, options = {}) {
        return this.post('/api/objects/spawn', {
            obj_type: objType,
            position,
            rotation,
            name,
            ...options
        });
    }

    async deleteObject(objectId) {
        return this.delete(`/api/objects/${objectId}`);
    }

    async updateObjectPose(objectId, position = null, rotation = null) {
        return this.patch(`/api/objects/${objectId}/pose`, { position, rotation });
    }

    async transformObject(objectId) {
        return this.post(`/api/objects/${objectId}/transform`);
    }

    // ===== Materials API =====

    async listMaterials() {
        return this.get('/api/materials/');
    }

    async getPresetMaterials() {
        return this.get('/api/materials/presets');
    }

    async createMaterial(name, color, roughness = 0.5, metallic = 0.0) {
        return this.post('/api/materials/', { name, color, roughness, metallic });
    }

    async bindMaterial(objectId, materialId) {
        return this.post('/api/materials/bind', { object_id: objectId, material_id: materialId });
    }

    async unbindMaterial(objectId) {
        return this.post(`/api/materials/unbind/${objectId}`);
    }

    // ===== WebSocket =====

    connectStream(onFrame, onError) {
        const wsUrl = `ws://${window.location.host}/ws/stream`;
        this.wsStream = new WebSocket(wsUrl);

        this.wsStream.onopen = () => {
            console.log('스트림 WebSocket 연결됨');
        };

        this.wsStream.onmessage = (event) => {
            if (event.data instanceof Blob) {
                const url = URL.createObjectURL(event.data);
                onFrame(url);
            }
        };

        this.wsStream.onerror = (error) => {
            console.error('스트림 WebSocket 오류:', error);
            if (onError) onError(error);
        };

        this.wsStream.onclose = () => {
            console.log('스트림 WebSocket 연결 종료');
        };
    }

    connectControl(onMessage, onError) {
        const wsUrl = `ws://${window.location.host}/ws/control`;
        this.wsControl = new WebSocket(wsUrl);

        this.wsControl.onopen = () => {
            console.log('제어 WebSocket 연결됨');
        };

        this.wsControl.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (onMessage) onMessage(data);
        };

        this.wsControl.onerror = (error) => {
            console.error('제어 WebSocket 오류:', error);
            if (onError) onError(error);
        };
    }

    sendControl(action, data) {
        if (this.wsControl && this.wsControl.readyState === WebSocket.OPEN) {
            this.wsControl.send(JSON.stringify({ action, data }));
        }
    }

    disconnect() {
        if (this.wsStream) this.wsStream.close();
        if (this.wsControl) this.wsControl.close();
    }
}

// 전역 API 인스턴스
window.api = new IsaacLabAPI();
