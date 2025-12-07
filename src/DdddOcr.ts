import { existsSync, readFile } from './file-ops/index.js';

import { ort } from './ort/index.js';

import { LogSeverityLevel } from './type.js';

import { ONNX_DIR } from './onnx/index.js';

import { to } from 'await-to-js';

class DdddOcr {
    private path: string = undefined;
    private logSeverityLevel: LogSeverityLevel = 4;

    /**
     * Flag indicating whether debugging is enabled.
     */
    protected isDebug = false;

    constructor() {
        if (ONNX_DIR !== undefined) {
            this.path = `${ONNX_DIR}/`;
        }
    }

    /**
     * Enables the debug mode and prepares the debug folder.
     */
    public enableDebug(): this {
        this.isDebug = true;

        return this;
    }

    /**
     * Sets the log severity level.
     * 
     * @param logSeverityLevel Log severity level. See https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/common/logging/severity.h
     */
    public setLogSeverityLevel(logSeverityLevel: LogSeverityLevel) {
        this.logSeverityLevel = logSeverityLevel;

        return this;
    }

    /**
     * Sets the root path for ONNX models.
     */
    public setPath(path: string) {
        this.path = path;

        return this;
    }

    private async readModelAsync(url: string) {
        if (existsSync(url)) {
            const result = await readFile(url);
            return result.buffer;
        }

        const [fetchErr, response] = await to(fetch(url));

        if (fetchErr) {
            throw new Error(`Could not load Buffer from URL: ${url}`);
        }

        if (!response.ok) {
            throw new Error(`HTTP Status ${response.status} for url ${url}`);
        }

        const [arrayBufferErr, data] = await to(response.arrayBuffer());

        return data;
    }

    protected async loadModelAsync(url: string) {
        if (this.path !== undefined) {
            url = this.path + url;
        }

        const result = await this.readModelAsync(url);

        const onnxPromise = ort.InferenceSession.create(result, {
            logSeverityLevel: this.logSeverityLevel
        });

        return onnxPromise;
    }

    private async readJsonAsync(url: string) {
        if (existsSync(url)) {
            const result = await readFile(url, { encoding: 'utf-8' });
            return result;
        }

        const [fetchErr, response] = await to(fetch(url));

        if (fetchErr) {
            throw new Error(`Could not load Buffer from URL: ${url}`);
        }

        if (!response.ok) {
            throw new Error(`HTTP Status ${response.status} for url ${url}`);
        }

        const [arrayBufferErr, data] = await to(response.text());

        return data;
    }

    protected async loadJsonAsync(url: string) {
        if (this.path !== undefined) {
            url = this.path + url;
        }

        const result = await this.readJsonAsync(url);
        return JSON.parse(result);
    }
}

export { DdddOcr };
