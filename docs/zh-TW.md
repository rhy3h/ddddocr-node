# DdddOcr JS

[![npm](https://img.shields.io/npm/v/ddddocr-node.svg)](https://www.npmjs.com/package/ddddocr-node)

[English](/README.md)

這個專案是 Python 專案 [DdddOcr](https://github.com/sml2h3/ddddocr) 的移植。

目標是讓在 JavaScript 中使用這個訓練好的模型進行文字偵測變得容易。

[文件](https://rhy3h.github.io/ddddocr-node/)

## 安裝

```sh
npm install ddddocr-node
```

## 功能

 - 基礎 OCR 識別能力
 - OCR 機率輸出
 - 目標偵測能力

### 基礎 OCR 識別能力

主要用於識別單行文字，文字佔據圖片主要部分的情況，例如常見的英數驗證碼。本專案可以識別中文字元、英文（隨機大小寫或指定範圍的大小寫）、數字和某些特殊字元。

```js
const { DdddOcr } = require('ddddocr-node');

const ddddOcr = new DdddOcr();

const result = await ddddOcr.classification('example.jpg');
console.log(result);
```

此函式庫包含兩個內建的 OCR 模型，預設不會自動切換。您需要使用帶參數的 `setOcrMode()` 來切換它們。

```js
const { DdddOcr, MODEL_TYPE } = require('ddddocr-node');

// 方法 1：建立實例後啟用 Beta OCR 模式
const ddddOcr = new DdddOcr();
ddddOcr.setOcrMode(MODEL_TYPE.OCR_BETA);

// 方法 2：在建立實例時直接啟用 Beta OCR 模式
const ddddOcr = new DdddOcr().setOcrMode(MODEL_TYPE.OCR_BETA);

const result = await ddddOcr.classification('example.jpg');
console.log(result);
```

### OCR 機率輸出

為了提供更靈活的控制和 OCR 結果的範圍限制，本專案支援設定 OCR 結果的範圍限制。

`setRanges()` 方法限制返回的字元。

此方法接受一個參數。如果輸入是 `int` 類型，則指的是預定義的字元集限制。如果輸入是 `string` 類型，則代表自定義字元集。

對於 int 輸入，請參考下表。

| 參數值 | 意義                                                                |
|-----------------------|------------------------------------------------------------------------|
| 0                     | 純整數 0-9                                                      |
| 1                     | 純小寫字母 a-z                                             |
| 2                     | 純大寫字母 A-Z                                             |
| 3                     | 小寫字母 a-z + 大寫字母 A-Z                          |
| 4                     | 小寫字母 a-z + 整數 0-9                                   |
| 5                     | 大寫字母 A-Z + 整數 0-9                                   |
| 6                     | 小寫字母 a-z + 大寫字母 A-Z + 整數 0-9           |
| 7                     | 預設字元集 - 小寫 a-z，大寫 A-Z 和整數 0-9 |

對於 `string` 輸入，請提供一個字串，其中每個字元都被視為候選字元，例如 `"0123456789+-x/="`。

```js
const { DdddOcr, CHARSET_RANGE } = require('ddddocr-node');

const ddddOcr = new DdddOcr();
ddddOcr.setRanges(CHARSET_RANGE.NUM_CASE);

const result = await ddddOcr.classification('example.jpg');
console.log(result);
```

## 目標偵測能力

主要目的是快速偵測目標物件在圖片中的可能位置。由於偵測到的目標不一定是文字，此函式僅提供目標的邊界框 (bbox) 位置。在目標偵測中，我們通常使用 bbox (bounding box) 來描述目標位置。bbox 是一個矩形框，可以由左上角的 x 和 y 座標以及右下角的 x 和 y 座標確定。

```js
const { DdddOcr } = require('ddddocr-node');

const ddddOcr = new DdddOcr();

const result = await ddddOcr.detection('example.jpg');
console.log(result);
```

如果您想將偵測到的邊界框新增到原始圖片中，這裡有一個範例。

```js
const { Jimp, cssColorToHex } = require('jimp');

const { DdddOcr } = require('ddddocr-node');
const { drawRectangle } = require('ddddocr-core');

const ddddOcr = new DdddOcr();

const result = await ddddOcr.detection('example.jpg');

const image = await Jimp.read('example.jpg');
const color = cssColorToHex('#ff0000');
for (let i = 0; i < result.length; i++) {
    const [x1, y1, x2, y2] = result[i];

    const points = [
        { x: x1, y: y1 },
        { x: x2, y: y1 },
        { x: x2, y: y2 },
        { x: x1, y: y2 }
    ];
    drawRectangle(image, points, color);
}

image.write('output.jpg');
```

## Star 歷史

[![Star History Chart](https://api.star-history.com/svg?repos=rhy3h/ddddocr-node&type=date&legend=top-left)](https://www.star-history.com/#rhy3h/ddddocr-node&type=date&legend=top-left)

## 未來計畫

 - 滑塊偵測
 - 匯入自定義 OCR 訓練模型
