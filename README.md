# DdddOcr Node

[![npm](https://img.shields.io/npm/v/ddddocr-node.svg)](https://www.npmjs.com/package/ddddocr-node)

This project is a port of the Python project [DdddOcr](https://github.com/sml2h3/ddddocr). 

The goal is to make it easy to use this trained model for text detection in JavaScript.

## Installation

```sh
npm install ddddocr-node
```

## Features

 - Basic OCR recognition capability
 - OCR probability output
 - Object detection capability

### Basic OCR recognition capability

Primarily used for recognizing single-line text, where the text occupies the main portion of the image, such as common alphanumeric captchas. This project can recognize Chinese characters, English (with random case sensitivity or case constrained by specified ranges), numbers, and certain special characters.

```js
const { DdddOcr } = require('ddddocr-node');

const ddddOcr = new DdddOcr();

const result = await ddddOcr.classification('example.jpg');
console.log(result);
```

This library includes two built-in OCR models, which do not switch automatically by default. You need to use `enableBetaOcr()` with parameters to switch between them.


```js
const { DdddOcr, MODEL_TYPE } = require('ddddocr-node');

// Method 1: Enable the beta OCR after creating the instance
const ddddOcr = new DdddOcr();
ddddOcr.enableBetaOcr(true);

// Method 2: Enable the beta OCR during instance creation
const ddddOcr = new DdddOcr().enableBetaOcr(true);

const result = await ddddOcr.classification('example.jpg');
console.log(result);
```

### OCR probability output

To provide more flexible control and range restriction for OCR results, the project supports setting range limitations on OCR results.

The `setRanges()` method restricts the returned characters.

This method accepts one parameter. If the input is of type `int`, it refers to a predefined character set restriction. If the input is of type `string`, it represents a custom character set.

For int input, please refer to the table below.

| Parameter <br/> Value | Meaning                                                                |
|-----------------------|------------------------------------------------------------------------|
| 0                     | Pure integers 0-9                                                      |
| 1                     | Pure lowercase letters a-z                                             |
| 2                     | Pure uppercase letters A-Z                                             |
| 3                     | Lowercase letters a-z + uppercase letters A-Z                          |
| 4                     | Lowercase letters a-z + integers 0-9                                   |
| 5                     | Uppercase letters A-Z + integers 0-9                                   |
| 6                     | Lowercase letters a-z + uppercase letters A-Z + integers 0-9           |
| 7                     | Default character set - lowercase a-z, uppercase A-Z, and integers 0-9 |

For `string` input, provide a string where each character is treated as a candidate character, e.g., `"0123456789+-x/="`.

```js
const { DdddOcr, CHARSET_RANGE } = require('ddddocr-node');

const ddddOcr = new DdddOcr();
ddddOcr.setRanges(CHARSET_RANGE.NUM_CASE);

const result = await ddddOcr.classification('example.jpg');
console.log(result);
```

## Object detection capability

The main purpose is to quickly detect the possible location of the target object in the image. Since the detected target may not necessarily be text, this function only provides the bounding box (bbox) location of the target. In object detection, we typically use a bbox (bounding box) to describe the target location. A bbox is a rectangular frame, which can be determined by the x and y coordinates of the top-left corner and the x and y coordinates of the bottom-right corner.

```js
const { DdddOcr } = require('ddddocr-node');

const ddddOcr = new DdddOcr();

const result = await ddddOcr.detection('example.jpg');
console.log(result);
```

If you want to add the detected bounding box to the original image, here is an example.

```js
const { Jimp, cssColorToHex } = require('jimp');

const { DdddOcr } = require('ddddocr-node');
const { drawRectangle } = require('ddddocr-node/image-utils');

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

## Futures

 - Slider detection
 - Import custom OCR training model
