import { Trajectory } from '../../engine/traces/trace';
import { parseTrajectoryJsonString, trajectoryToJsonString } from '../../engine/traces/trajectorySerialization';

function getTrajectoryDownloadFilename(trajectory: Trajectory): string {
    if (trajectory.trajectoryId) {
        return `${trajectory.trajectoryId}.json`;
    }

    return `trajectory-${new Date().toISOString().replace(/[:.]/g, '-')}.json`;
}

export function downloadTrajectoryJson(trajectory: Trajectory, filename?: string): void {
    if (!trajectory) return;

    const json = trajectoryToJsonString(trajectory);
    const blob = new Blob([json], { type: 'application/json;charset=utf-8' });
    const objectUrl = window.URL.createObjectURL(blob);
    const anchor = document.createElement('a');

    anchor.href = objectUrl;
    anchor.download = filename || getTrajectoryDownloadFilename(trajectory);
    anchor.rel = 'noopener';
    anchor.style.display = 'none';

    document.body.appendChild(anchor);
    anchor.click();
    document.body.removeChild(anchor);
    window.URL.revokeObjectURL(objectUrl);
}

export async function readTrajectoryFile(file: File): Promise<Trajectory> {
    const jsonString = await file.text();
    return parseTrajectoryJsonString(jsonString);
}
