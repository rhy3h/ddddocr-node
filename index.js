const ort = require('onnxruntime-node');

class DdddOcr {
    constructor(path) {
        this._loadOrtSession(path);
    }

    _loadOrtSession(path) {
        if (!this._ortSessionPending) {
            this._ortSessionPending = ort.InferenceSession.create(path);
        }

        return this._ortSessionPending;
    }
}

module.exports = {
    DdddOcr
};