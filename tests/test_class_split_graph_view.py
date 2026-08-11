import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "ybat-master" / "class_split_graph_view.js"


def test_graph_view_controls_are_restyle_only_stable_and_non_compounding():
    script = r"""
const assert = require('assert');
const calls = [];
global.Plotly = {
  restyle: (_graph, update, indexes) => {
    calls.push({ update, indexes });
    return Promise.resolve();
  },
  react: () => { throw new Error('react must not run'); },
  relayout: () => { throw new Error('relayout must not run'); },
  newPlot: () => { throw new Error('newPlot must not run'); },
};
const view = require(process.argv[1]);
const graph = { data: [
  { mode: 'lines', name: 'hull', x: [0], y: [0] },
  {
    mode: 'markers',
    name: 'TransitObject',
    showlegend: true,
    customdata: ['a', 'b', 'c', 'd'],
    marker: { size: [10, 20, 12, 8], opacity: [0.8, 0.6, 1, 0.4] },
  },
  {
    mode: 'markers',
    name: 'Objects',
    showlegend: true,
    customdata: ['x'],
    marker: { size: [7], opacity: [0.5] },
  },
] };
(async () => {
  const settings = {
    sizePercent: 150,
    opacityPercent: 50,
    labelDensityPercent: 50,
  };
  await view.captureAndApply(graph, settings);
  await view.applyToGraph(graph, settings);
  assert.strictEqual(calls.length, 2);
  assert.deepStrictEqual(calls[0].indexes, [1, 2]);
  assert.deepStrictEqual(
    calls[0].update['marker.size'][0],
    [15, 30, 18, 12],
  );
  assert.deepStrictEqual(
    calls[1].update['marker.size'][0],
    [15, 30, 18, 12],
  );
  assert.deepStrictEqual(calls[0].update['marker.opacity'][0], [0.4, 0.3, 0.5, 0.2]);
  assert.strictEqual(
    calls[0].update.texttemplate[0].filter(Boolean).length,
    2,
  );
  assert.deepStrictEqual(calls[0].update.texttemplate[1], null);
  graph.data[1].customdata = ['b', 'd'];
  graph.data[1].marker.size = [30, 12];
  await view.syncAfterExternalRestyle(graph, settings);
  assert.deepStrictEqual(
    calls[2].update['marker.size'][0],
    [30, 12],
  );
  assert.deepStrictEqual(
    calls[2].update['marker.opacity'][0],
    [0.3, 0.2],
  );
  process.stdout.write(JSON.stringify({ calls: calls.length }));
})().catch((error) => {
  console.error(error);
  process.exit(1);
});
"""
    completed = subprocess.run(
        ["node", "-e", script, str(MODULE)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == {"calls": 3}


def test_graph_view_label_sampling_is_deterministic_and_class_only():
    script = r"""
const assert = require('assert');
const view = require(process.argv[1]);
const classTrace = {
  mode: 'markers',
  name: 'Elevated fixture',
  showlegend: true,
  customdata: ['p4', 'p1', 'p3', 'p2'],
  marker: {},
};
const numericTrace = {
  mode: 'markers',
  name: 'Objects',
  showlegend: true,
  customdata: ['p4', 'p1'],
  marker: {},
};
const first = view.sampledLabels(classTrace, 50);
const second = view.sampledLabels(classTrace, 50);
assert.deepStrictEqual(first, second);
assert.strictEqual(first.filter(Boolean).length, 2);
assert.deepStrictEqual(
  view.sampledLabels(numericTrace, 100),
  ['', ''],
);
"""
    subprocess.run(
        ["node", "-e", script, str(MODULE)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_graph_view_disables_density_outside_class_color_mode():
    script = r"""
const assert = require('assert');
const view = require(process.argv[1]);
const elements = {
  classSplitMarkerSize: { value: '100' },
  classSplitMarkerOpacity: { value: '100' },
  classSplitLabelDensity: { value: '80', disabled: false },
  classSplitColorMode: { value: 'area' },
  classSplitMarkerSizeValue: { textContent: '' },
  classSplitMarkerOpacityValue: { textContent: '' },
  classSplitLabelDensityValue: { textContent: '' },
};
const doc = { getElementById: (id) => elements[id] || null };
const settings = view.settingsFromDocument(doc);
assert.strictEqual(settings.labelDensityPercent, 0);
"""
    subprocess.run(
        ["node", "-e", script, str(MODULE)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_graph_view_coalesces_rapid_restyles_in_latest_settings_order():
    script = r"""
const assert = require('assert');
const calls = [];
const resolvers = [];
global.Plotly = {
  restyle: (_graph, update) => {
    calls.push(update);
    return new Promise((resolve) => resolvers.push(resolve));
  },
};
const view = require(process.argv[1]);
const graph = { data: [{
  mode: 'markers', name: 'TransitObject', showlegend: true,
  customdata: ['a'], marker: { size: [10], opacity: [0.8] },
}] };
(async () => {
  const first = view.captureAndApply(graph, {
    sizePercent: 100, opacityPercent: 100, labelDensityPercent: 0,
  });
  const latest = view.applyToGraph(graph, {
    sizePercent: 200, opacityPercent: 50, labelDensityPercent: 0,
  });
  assert.strictEqual(calls.length, 1);
  resolvers.shift()();
  await new Promise((resolve) => setImmediate(resolve));
  assert.strictEqual(calls.length, 2);
  assert.deepStrictEqual(calls[1]['marker.size'][0], [20]);
  assert.deepStrictEqual(calls[1]['marker.opacity'][0], [0.4]);
  resolvers.shift()();
  await Promise.all([first, latest]);
})().catch((error) => { console.error(error); process.exit(1); });
"""
    subprocess.run(
        ["node", "-e", script, str(MODULE)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
