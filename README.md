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

### Basic OCR recognition capability

```js
const { DdddOcr } = require('ddddocr-node');

const ddddOcr = new DdddOcr();

const result = await ddddOcr.classification('example.jpg');
console.log(result);
```

### OCR probability output

To provide more flexible control and range restriction for OCR results, the project supports setting range limitations on OCR results.

The `setRanges()` method restricts the returned characters.

This method accepts one parameter. If the input is of type `int`, it refers to a predefined character set restriction. If the input is of type `string`, it represents a custom character set.

For int input, please refer to the table below.

| Parameter <br/> Value | Meaning                                                          |
|-----------------|------------------------------------------------------------------------|
| 0               | Pure integers 0-9                                                      |
| 1               | Pure lowercase letters a-z                                             |
| 2               | Pure uppercase letters A-Z                                             |
| 3               | Lowercase letters a-z + uppercase letters A-Z                          |
| 4               | Lowercase letters a-z + integers 0-9                                   |
| 5               | Uppercase letters A-Z + integers 0-9                                   |
| 6               | Lowercase letters a-z + uppercase letters A-Z + integers 0-9           |
| 7               | Default character set - lowercase a-z, uppercase A-Z, and integers 0-9 |

For `string` input, provide a string where each character is treated as a candidate character, e.g., `"0123456789+-x/="`.

```js
const { DdddOcr, CHARSET_RANGE } = require('ddddocr-node');

const ddddOcr = new DdddOcr();
ddddOcr.setRanges(CHARSET_RANGE.NUM_CASE);

const result = await ddddOcr.classification('example.jpg');
console.log(result);
```

## Futures

 - Object detection capability
 - Slider detection
 - Import custom OCR training model
