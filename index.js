const ort = require('onnxruntime-node');
const { Jimp } = require('jimp');

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

    async run(inputTensor) {
        const ortSession = await this._ortSessionPending;
        const result = await ortSession.run({ input1: inputTensor });

        return result['387'];
    }

    async classification(img) {
        const image = await Jimp.read(img);

        const { width, height } = image.bitmap;
        const targetHeight = 64;
        const targetWidth = Math.floor(width * (targetHeight / height));
        image.resize({
                w: targetWidth, 
                h: targetHeight
            })
        image.greyscale();

        const { data } = image.bitmap;
        const floatData = new Float32Array(targetWidth * targetHeight);
        for (let i = 0, j = 0; i < data.length; i += 4, j += 3) {
            floatData[j] = (data[i] / 255.0 - 0.5) / 0.5;
        }
        const inputTensor = new ort.Tensor('float32', floatData, [1, 1, targetHeight, targetWidth]);

        const result = await this.run(inputTensor);

        return result;
    }
}

module.exports = {
    DdddOcr
};