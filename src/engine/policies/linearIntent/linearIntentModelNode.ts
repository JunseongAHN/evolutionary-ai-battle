import fs from 'node:fs';
import path from 'node:path';
import { parseLinearIntentModelJsonString } from './linearIntentModel';
import { LinearIntentModelJson } from './linearIntentTypes';

export function loadLinearIntentModelFromFile(filePath: string): LinearIntentModelJson {
    const contents = fs.readFileSync(filePath, 'utf8');
    return parseLinearIntentModelJsonString(contents);
}

export function getDefaultLinearIntentModelPath(): string {
    return path.resolve(process.cwd(), 'public/models/linear-intent-model-v0.2.json');
}
