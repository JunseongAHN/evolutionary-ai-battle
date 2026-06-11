import assert from 'node:assert/strict';
import { test } from 'node:test';
import { scenarioCatalog } from '../scenarioCatalog';
import { getScenarioPlayer, loadScenarioDefinition, validateScenarioDefinition } from '../scenarioGt';

test('validateScenarioDefinition accepts a valid scenario', () => {
    assert.deepEqual(validateScenarioDefinition(scenarioCatalog[0]), []);
});

test('validateScenarioDefinition rejects missing gt', () => {
    const invalid = JSON.parse(JSON.stringify(scenarioCatalog[0]));
    delete invalid.gt;

    const errors = validateScenarioDefinition(invalid);
    assert.ok(errors.length > 0);
    assert.match(errors.join('; '), /gt is required/);
});

test('loadScenarioDefinition preserves scenario content', () => {
    const loaded = loadScenarioDefinition(scenarioCatalog[1]);
    assert.equal(loaded.scenarioId, 'teammate_under_pressure');
    assert.equal(loaded.gt.intent, 'support_teammate_under_pressure');
});

test('getScenarioPlayer returns the requested player', () => {
    const player = getScenarioPlayer(scenarioCatalog[2], 'team-a-1');
    assert.equal(player?.id, 'team-a-1');
    assert.equal(player?.teamId, 'team-a');
});
