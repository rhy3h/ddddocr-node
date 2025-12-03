import { readFile } from './file-ops';

import { ort } from './ort/index';

import { LogSeverityLevel } from './type';

class DdddOcr {
    private path: string = '';
    private logSeverityLevel: LogSeverityLevel = 4;

    /**
     * Flag indicating whether debugging is enabled.
     */
    protected isDebug = false;

    constructor() {}

    /**
     * Enables the debug mode and prepares the debug folder.
     */
    public enableDebug(): this {
        this.isDebug = true;

        return this;
    }

    public setLogSeverityLevel(logSeverityLevel: LogSeverityLevel) {
        this.logSeverityLevel = logSeverityLevel;

        return this;
    }

    public setPath(path: string) {
        this.path = path;
        return this;
    }

    protected async loadModelAsync(url: string) {
        if (this.path !== undefined) {
            url = this.path + url;
        }

        const result = await readFile(url);

        const onnxPromise = ort.InferenceSession.create(result.buffer, {
            logSeverityLevel: this.logSeverityLevel
        });

        return onnxPromise;
    }

    protected async loadJsonAsync(url: string) {
        if (this.path !== undefined) {
            url = this.path + url;
        }

        const result = await readFile(url, { encoding: 'utf-8' });
        return JSON.parse(result);
    }
}

export { DdddOcr };
