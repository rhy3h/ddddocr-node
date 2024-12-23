const { DdddOcr, CHARSET_RANGE, MODEL_TYPE } = require('../index');
const assert = require('assert');

(async () => {
    const ddddOcr = new DdddOcr();

    const ocrResult = await ddddOcr.classification('./test/example-en.jpg');
    assert.strictEqual(ocrResult, '8A62N1', 'OCR Error');

    ddddOcr.setRanges(CHARSET_RANGE.NUM_CASE);
    const ocrNumCaseResult = await ddddOcr.classification('./test/example-en.jpg');
    assert.strictEqual(ocrNumCaseResult, '8621', 'OCR num case Error');

    ddddOcr.setRanges(CHARSET_RANGE.LOWWER_CASE);
    const ocrLowerCaseResult = await ddddOcr.classification('./test/example-en.jpg');
    assert.strictEqual(ocrLowerCaseResult, '', 'OCR lower case Error');
    
    ddddOcr.setRanges(CHARSET_RANGE.UPPER_CASE);
    const ocrUpperCaseResult = await ddddOcr.classification('./test/example-en.jpg');
    assert.strictEqual(ocrUpperCaseResult, 'AN', 'OCR upper case Error');

    ddddOcr.setRanges('N');
    const ocrStrResult = await ddddOcr.classification('./test/example-en.jpg');
    assert.strictEqual(ocrStrResult, 'N', 'OCR string case Error');

    ddddOcr.setRanges('');
    ddddOcr.setOcrMode(MODEL_TYPE.OCR_BETA);
    const ocrBetaResult = await ddddOcr.classification('./test/example-ch.jpg');
    assert.strictEqual(ocrBetaResult, '九乘六等于？', 'OCR beta case Error');

    const detectionResult = await ddddOcr.detection('./test/example-ch.jpg');
    assert.deepEqual(
        detectionResult,
        [
            [ 80, 3, 99, 21 ],
            [ 100, 0, 127, 19 ],
            [ 2, 2, 22, 22 ],
            [ 31, 7, 51, 26 ],
            [ 56, 6, 77, 25 ]
        ],
        'Detection Error'
    );

    console.log('[TEST] SUCESS');
})();
