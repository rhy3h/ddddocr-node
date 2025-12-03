import { promises as fs } from 'fs';

export { existsSync, rmSync, mkdirSync } from 'fs';

export const readFile = fs.readFile;
export const writeFile = fs.writeFile;
