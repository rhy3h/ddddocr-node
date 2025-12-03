import { promises as fs } from 'fs';

export const isSupportDebug = () => true;

export { existsSync, rmSync, mkdirSync } from 'fs';

export const readFile = fs.readFile;
export const writeFile = fs.writeFile;
