"""Generate tutorials/DIMER_Sound_Event_Classification_Workshop.ipynb, the standalone sound event workshop.

This file is the notebook's source of truth (NOTEBOOK_SPEC SRC13). Edit it, regenerate, and commit both; CI runs
``python tools/build_sound_workshop.py --check`` and fails on any byte difference.

Inputs:
- tools/sound_workshop_source.py: the carried isolated worker (``workshop.py`` in the notebook);
- tools/sound-workshop-requirements.lock: the hash-locked worker environment (compiled from the ``.in`` file);
- tools/sound_workshop_sample.json, tools/sound_workshop_attribution.txt, tools/sound_workshop_metadata.csv: the
  frozen ESC-10 sample manifest and the pinned ESC-50 attribution and metadata (checked against the manifest pins);
- src/ast_audio_classification_pipeline/*.py, weights/ast-audioset/dimer-base-manifest.json, LICENSE and
  docs/WEIGHTS.md: carried unchanged (the release validator checks their parity);
- the cell texts and notebook metadata below. CARRIED_FILES uses tools/split_workshop_carrier.py's layout.

History: first generator revision on the backup branch commit b19719c (``build_sound_workshop.py/1``, sha256
``337c129c...``); the notebook then received scripted in-place revisions (logged in ``post_generation_revisions``)
until this generator was brought up to date and imported into tools/ (SEC2-m8, option A). The import was made to
reproduce the committed notebook byte for byte, so ``generated_from.generator``/``generator_sha256`` keep recording
that first revision (``RECORDED_GENERATOR``) rather than a digest of this file: a self-digest would change the
notebook on every edit here. The CI ``--check`` byte parity, not that digest, is what ties this file to the notebook.
When the next content revision is generated, update ``RECORDED_GENERATOR`` (and drop or extend the revision log)
in the same commit.

Usage: python tools/build_sound_workshop.py [--check]
"""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "DIMER_Sound_Event_Classification_Workshop.ipynb"
TARGET = ROOT / "tutorials" / NAME
# Provenance stamped into generated_from and the carried source.json; see "History" above.
RECORDED_GENERATOR = ("build_sound_workshop.py/1", "337c129c65acdd0305a279f1db3f1168c1f28f5496389d16bd55c40f0b51181d")
REPOSITORY = "kurtvalcorza/ast-audio-classification-pipeline"
BASE_REVISION = "1988bd6cdda5fb3d1d8619f866f53f7ed9c56f27"
CARRIED_SOURCES = {
    "workshop.py": "tools/sound_workshop_source.py",
    "requirements.txt": "tools/sound-workshop-requirements.lock",
    "sample.json": "tools/sound_workshop_sample.json",
    "licenses/esc-attribution.txt": "tools/sound_workshop_attribution.txt",
    "licenses/esc-metadata.csv": "tools/sound_workshop_metadata.csv",
    "weights/ast/dimer-base-manifest.json": "weights/ast-audioset/dimer-base-manifest.json",
    "licenses/ast-code.txt": "LICENSE",
    "licenses/weight-provenance.md": "docs/WEIGHTS.md",
}


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read(relative: str) -> str:
    return (ROOT / relative).read_bytes().decode("utf-8").replace("\r\n", "\n")


def carried_files() -> dict[str, str]:
    sample = json.loads(_read("tools/sound_workshop_sample.json"))
    for name, key in (("sound_workshop_attribution.txt", "license"), ("sound_workshop_metadata.csv", "csv")):
        raw = (ROOT / "tools" / name).read_bytes()
        pin = sample["metadata_pins"][key]
        if len(raw) != pin["bytes"] or hashlib.sha256(raw).hexdigest() != pin["sha256"]:
            raise ValueError("Pinned dataset attribution/metadata changed: " + name)
    files = {
        "ast_reference/" + p.name: _read(str(p.relative_to(ROOT)))
        for p in sorted((ROOT / "src/ast_audio_classification_pipeline").glob("*.py"))
    }
    for target, source in CARRIED_SOURCES.items():
        files[target] = _read(source)
    return files


def _carried_literal():
    spec = importlib.util.spec_from_file_location("split_workshop_carrier", ROOT / "tools" / "split_workshop_carrier.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.carried_literal


def generator_identity() -> tuple[str, str]:
    return RECORDED_GENERATOR


def build(identity: tuple[str, str] | None = None) -> str:
    """The notebook as text. ``identity`` overrides (generator, generator_sha256), for parity proofs only."""
    files = carried_files()
    generator, generator_sha256 = identity or generator_identity()
    source = {
        "repository": REPOSITORY,
        "base_revision": BASE_REVISION,
        "generator": generator,
        "generator_sha256": generator_sha256,
        "files": {name: sha(text) for name, text in files.items()},
    }
    files["source.json"] = json.dumps(source, indent=2)
    hashes = {name: sha(text) for name, text in files.items()}
    carrier = "CARRIED_FILES = " + _carried_literal()(files) + "\nCARRIED_HASHES = " + repr(hashes) + CARRIER_TAIL
    cells = []
    for kind, text, metadata in CELLS:
        text = carrier if text is CARRIER else text
        if kind == "code":
            ast.parse(text)
            cells.append({"cell_type": "code", "execution_count": None, "id": f"code-{len(cells):02d}",
                          "metadata": copy.deepcopy(metadata), "outputs": [], "source": text.splitlines(True)})
        else:
            cells.append({"cell_type": "markdown", "id": f"md-{len(cells):02d}", "metadata": copy.deepcopy(metadata),
                          "source": text.splitlines(True)})
    metadata = copy.deepcopy(METADATA)
    metadata["dimer"]["generated_from"] = {**source, "post_generation_revisions": copy.deepcopy(REVISIONS)}
    notebook = {"cells": cells, "metadata": metadata, "nbformat": 4, "nbformat_minor": 5}
    return json.dumps(_ordered(notebook), indent=1, ensure_ascii=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="exit 1 if the committed notebook differs")
    args = parser.parse_args()
    content = build().encode("utf-8")
    if args.check:
        if not TARGET.exists() or TARGET.read_bytes() != content:
            print(f"FAIL: tutorials/{NAME} differs from tools/build_sound_workshop.py; regenerate it")
            return 1
        print(f"Sound workshop parity: PASS (tutorials/{NAME})")
        return 0
    TARGET.write_bytes(content)
    print(f"wrote tutorials/{NAME}")
    return 0


CARRIER = object()  # placeholder for the generated CARRIED_FILES cell


def _ordered(notebook: dict) -> dict:
    """Key order of the committed notebook JSON."""
    code_keys, md_keys = ['cell_type', 'id', 'metadata', 'execution_count', 'outputs', 'source'], ['cell_type', 'id', 'metadata', 'source']
    cells = [{k: c[k] for k in (code_keys if c['cell_type'] == 'code' else md_keys)} for c in notebook['cells']]
    return {k: (cells if k == 'cells' else notebook[k]) for k in ['nbformat', 'nbformat_minor', 'metadata', 'cells']}


CARRIER_TAIL = r"""
for name, text in CARRIED_FILES.items():
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8', newline='\n')
    if hashlib.sha256(path.read_bytes()).hexdigest() != CARRIED_HASHES[name]:
        raise RuntimeError('Carried file integrity failure: ' + name)
print('Embedded reference code and immutable manifests verified.')"""

# Notebook metadata; generated_from is filled in by build().
METADATA = {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
 'language_info': {'name': 'python', 'version': '3.12'},
 'accelerator': 'GPU',
 'colab': {'name': 'DIMER_Sound_Event_Classification_Workshop.ipynb', 'gpuType': 'T4', 'provenance': []},
 'dimer': {'notebook_spec': '2.2',
           'notebook_profile': 'E2E',
           'notebook_mode': 'WORKSHOP',
           'notebook_kind': 'WORKSHOP',
           'standalone': True,
           'requires_dimer_worker': False,
           'release_status': 'Candidate',
           'generated_from': None}}

# Scripted revisions applied before this generator was brought up to date (kept as history).
REVISIONS = [{'date': '2026-10-02',
  'reason': 'Notebook Review Framework v1 fixes SEC-M1..M3 and SEC-m1..m5 (docs/reviews/2026-10-02-notebook-review)',
  'cells': ['code-03', 'code-04', 'code-10', 'md-11', 'md-15', 'code-16', 'md-17', 'code-18'],
  'carried_files_changed': ['workshop.py', 'source.json'],
  'method': 'anchor-asserting replacements on the generated notebook; generator not in this repository'},
 {'date': '2026-10-03',
  'reason': 'Source layout: no cell line over 2000 characters (the CARRIED_FILES line was 959,724 characters)',
  'cells': ['code-03'],
  'carried_files_changed': [],
  'method': 'tools/split_workshop_carrier.py: CARRIED_FILES re-serialised as parenthesised runs of string pieces (at '
            'most 1000 characters each); ast.literal_eval of the new literal equals the old one, so the carried text '
            'and its hashes are unchanged; generator not in this repository'},
 {'date': '2026-10-13',
  'reason': 'Notebook Review Framework v1 fixes SEC2-m1, SEC2-m2, SEC2-m3, SEC2-m5, SEC2-m6 and SEC2-m7 (review of '
            '4e9256f, round 2026-10-12)',
  'cells': ['md-00',
            'code-03',
            'code-04',
            'md-05',
            'md-07',
            'md-11',
            'md-17',
            'code-18',
            'md-19',
            'code-20',
            'md-21'],
  'carried_files_changed': ['workshop.py', 'source.json'],
  'method': 'fix_sec2.py: anchor-asserting replacements (each anchor matched exactly once) on blob 586a465; '
            'CARRIED_FILES re-serialised with tools/split_workshop_carrier.py carried_literal; carried hashes, the '
            'carried source.json and generated_from.files updated together; generator not in this repository'},
 {'date': '2026-10-13',
  'reason': 'Peer-review follow-up CX1-CX9 to the SEC2 fixes (PR #10)',
  'cells': ['code-03', 'code-04', 'md-07', 'md-11', 'code-14', 'md-15', 'md-17', 'code-18', 'md-21'],
  'carried_files_changed': ['workshop.py', 'source.json'],
  'method': 'fix_cx.py: anchor-asserting replacements (each anchor matched exactly once) on blob 76c4ac7; '
            'prepare/train/reload/activity asserted AST-identical; CARRIED_FILES re-serialised with '
            'tools/split_workshop_carrier.py carried_literal; carried hashes, the carried source.json and '
            'generated_from.files updated together; generator not in this repository'}]

CELLS = [
    # md-00
    ('markdown', """# DIMER Notebook: Sound Event Classification — Learning to Recognize What We Hear

**What sound is present, and how much should we trust the model when the recording changes?**

> **⚠️ Notice.** For research, training and evaluation only. Not validated for production,
> clinical, diagnostic or safety-critical use. Running this notebook downloads third-party
> model weights and datasets under their own licenses (see the model card) and uses compute
> on your own Colab account. Data you bring stays in your runtime, and you are
> responsible for having the right to use it. Results here are tutorial evidence, not
> benchmarks. Everything is provided "as is", without warranty of any kind. To the maximum
> extent permitted by law, the maintainers and the DIMER platform accept no liability for
> its use.

**Profile:** `E2E` · **Mode:** `WORKSHOP` · **Notebook specification:** `2.2` · **Status:** Candidate; fresh hosted GPU and real labelled BYOD qualification pending.

This modular, self-paced **Level 2 — Applied tasks** unit complements Whisper. Whisper transcribes words; this unit classifies environmental sounds. Basic Python and Colab familiarity are sufficient; no prior ML experience is required. Completion records are optional personal notes, **not required submissions**.

**Input → Model → Output:** a short audio recording → a frozen Audio Spectrogram Transformer (AST [1]; `MIT/ast-finetuned-audioset-10-10-0.4593` [2] at immutable revision `f826b80d28226b62986cc218e5cec390b1096902`) with a new trainable classifier head → one of ten named sound categories and its scores. A confident score does not establish correctness or the absence of an unknown sound.

**What the backbone already knows:** the checkpoint was fine-tuned on AudioSet [3], and its 527 labels already include all ten ESC-10 categories (for example Dog, Rain, Sneeze and Chainsaw). This notebook therefore shows how a new head reads features from a backbone already supervised on these sounds; it does not show a model learning to hear them from 240 clips.

You will inspect waveforms and spectrograms, compare simple baselines, train only the task head, select with validation recordings, evaluate held-out recordings, verify a saved artifact in a fresh process, and change background noise while holding other conditions fixed.

**How to use this notebook — fresh Run all:** choose Runtime → Change runtime type → T4 GPU, then Runtime → Run all. Section 1 is infrastructure; learner-facing work starts in Section 2, and the optional sections at the end are form-controlled and off by default. No token, upload, restart, runtime Git clone, DIMER service or inference API is required. Public data and weights download anonymously; an isolated Python 3.12.12 worker preserves the notebook kernel. Optional own-data routes are disabled by default. The 400 pinned WAV files total **176,417,600 bytes**, plus the model and environment. Allow several GiB; full runtime and peak memory are measured during qualification, not promised here.

**Roadmap:** inspect real sounds → understand output semantics and metrics → train and select → evaluate and reload → add controlled noise → conclude → optional own recordings.""", {}),
    # md-01
    ('markdown', """## 1. Prepare the isolated runtime

**Input:** a fresh Linux x86_64 Colab GPU runtime. **System:** embedded reference code and a verified, hash-locked environment. **Output:** an isolated worker with pinned model/data identities.

**Infrastructure:** the next three cells set up the runtime. You may run them without studying their implementation, and they may be collapsed. Public defaults need no Hugging Face token. Integrity failures stop execution; do not bypass them. Stage receipts invalidate downstream results when prerequisites change.""", {}),
    # code-02
    ('code', """import hashlib
import json
import os
import platform
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request
import uuid
import zipfile

SESSION_START = time.perf_counter()
if platform.system() != 'Linux' or platform.machine() != 'x86_64':
    raise RuntimeError('The canonical environment is Linux x86_64 Google Colab with a T4 GPU.')
try:
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], capture_output=True, text=True)
except FileNotFoundError as exc:
    raise RuntimeError('Select Runtime > Change runtime type > T4 GPU before Run all.') from exc
if gpu.returncode:
    raise RuntimeError('Select a fresh T4 GPU runtime; CUDA was not detected.')
print(gpu.stdout.strip())
ROOT = Path.cwd() / 'outputs' / 'sound_events' / uuid.uuid4().hex[:12]
ROOT.mkdir(parents=True)""", {}),
    # code-03
    ('code', CARRIER, {'cellView': 'form', 'dimer': {'embedded_sources': True}}),
    # code-04
    ('code', r"""UV_URL = 'https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl'
UV_SHA256 = 'aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60'
for attempt in range(3):
    try:
        with urllib.request.urlopen(UV_URL, timeout=90) as response:
            wheel = response.read(20081405)
        break
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        if attempt == 2:
            raise
        time.sleep(2 ** attempt)
if len(wheel) != 20081404 or hashlib.sha256(wheel).hexdigest() != UV_SHA256:
    raise RuntimeError('uv wheel size/hash mismatch')
import io
with zipfile.ZipFile(io.BytesIO(wheel)) as archive:
    member = next(n for n in archive.namelist() if n.endswith('.data/scripts/uv'))
    UV = ROOT / 'uv'
    UV.write_bytes(archive.read(member))
UV.chmod(0o700)
ENV = dict(os.environ, HF_HUB_DISABLE_IMPLICIT_TOKEN='1', HF_HUB_DISABLE_TELEMETRY='1', DO_NOT_TRACK='1')
ENV.pop('HF_TOKEN', None)
ENV.pop('HUGGING_FACE_HUB_TOKEN', None)
# Colab's kernel exports MPLBACKEND=module://matplotlib_inline.backend_inline; the isolated worker does not
# install matplotlib_inline, so importing matplotlib there would fail. The worker only writes figures to files.
ENV['MPLBACKEND'] = 'Agg'
print('Creating the isolated Python 3.12.12 environment and installing the hash-locked packages;'
      ' this can take a minute or more on a fresh runtime.', flush=True)
subprocess.run([str(UV), 'venv', '--managed-python', '--python', '3.12.12', str(ROOT / 'env')], env=ENV, check=True)
PYTHON = ROOT / 'env/bin/python'
subprocess.run([str(UV), 'pip', 'install', '--python', str(PYTHON), '--require-hashes', '--only-binary', ':all:',
                '--index-url', 'https://pypi.org/simple', '-r', str(ROOT / 'requirements.txt')], env=ENV, check=True)
VERSION_CHECK = ('import sys, torch, torchaudio, transformers; print("Python", sys.version.split()[0]); '
                 'print("torch", torch.__version__); print("torchaudio", torchaudio.__version__); '
                 'print("transformers", transformers.__version__); assert torch.cuda.is_available(); '
                 'print("CUDA device", torch.cuda.get_device_name(0))')
versions = subprocess.run([str(PYTHON), '-c', VERSION_CHECK], env=ENV, capture_output=True, text=True)
print(versions.stdout.strip())
if versions.returncode:
    print(versions.stderr)
    raise RuntimeError('The isolated environment failed its version/CUDA check. Read the output above.')
BOOTSTRAP_SECONDS = time.perf_counter() - SESSION_START

def run_stage(stage, *options, root=None):
    selected_root = ROOT if root is None else root
    log = selected_root / (stage + '.log')
    print('Running', stage, '— one model per process; log:', log, flush=True)
    with log.open('w', encoding='utf-8') as output:
        result = subprocess.Popen([str(PYTHON), '-u', str(ROOT / 'workshop.py'), '--root', str(selected_root), '--stage', stage, *options],
                                  env=ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in result.stdout:
            output.write(line)
            output.flush()
            if line.strip() and len(line) < 2000:
                print(line.rstrip(), flush=True)
        result.wait()
    if result.returncode == 65:
        # The worker refused the learner's input and printed the reason once; nothing to repeat or raise.
        return False
    if result.returncode:
        # Lines were streamed above; on failure, repeat the end of the log next to the error.
        print(log.read_text(encoding='utf-8')[-7000:])
        raise RuntimeError(f'{stage} failed with exit {result.returncode}. Read the diagnostics above.')
    return True

from IPython.display import Audio, FileLink, Image, Markdown, display

def read_output(name, root=None):
    return json.loads(((ROOT if root is None else root) / 'outputs' / name).read_text())

def table(rows, columns):
    def cell(value):
        if value is None:
            return 'undefined'
        if isinstance(value, float):
            return f'{value:.4f}'
        return str(value).replace('|', '/').replace('\n', ' ')
    lines = ['| ' + ' | '.join(title for key, title in columns) + ' |',
             '| ' + ' | '.join('---' for _ in columns) + ' |']
    lines += ['| ' + ' | '.join(cell(row.get(key)) for key, title in columns) + ' |' for row in rows]
    display(Markdown('\n'.join(lines)))

def show_metrics(root=None):
    rows = []
    for model, roles in read_output('metrics.json', root).items():
        for role, values in roles.items():
            rows.append(dict(model=model, role=role, accuracy=values['accuracy'], macro_f1=values['macro_f1']))
    table(rows, [('model', 'System'), ('role', 'Role'), ('accuracy', 'Accuracy'), ('macro_f1', 'Macro-F1')])

def show_selection(root=None):
    history = read_output('training_history.json', root)
    selection = read_output('selection.json', root)
    best = selection['validation_macro_f1']
    tied = [row['epoch'] for row in history if row['validation_macro_f1'] == best]
    chosen = next(row for row in history if row['epoch'] == selection['selected_epoch'])
    print('Selected epoch:', selection['selected_epoch'], '| validation macro-F1:', round(best, 4))
    print('Updates contained in the selected head:', chosen['optimizer_steps'], 'of',
          selection['optimizer_steps'], 'performed in total')
    print('Epochs tied at the best validation macro-F1:', tied)
    if len(tied) > 1:
        print(f'{len(tied)} epochs tie, so validation cannot tell them apart; the earliest-tie rule kept epoch '
              f"{selection['selected_epoch']}. This does not show that later epochs were worse.")

def show_paired_errors(root=None):
    paired = read_output('paired_errors.json', root)
    names = read_output('configuration.json', root)['class_names']
    table([{'outcome': key.replace('_', ' '), 'clips': paired[key]}
           for key in ('corrected', 'new_errors', 'unchanged_correct', 'unchanged_incorrect')],
          [('outcome', 'Initial head to selected head, same test clips'), ('clips', 'Test clips')])
    mistakes = paired['highest_confidence_mistakes']
    if mistakes:
        table([{'id': r['id'], 'true': names[r['label']], 'predicted': names[r['predicted']],
                'score': max(r['probabilities'])} for r in mistakes],
              [('id', 'Test clip'), ('true', 'Reference'), ('predicted', 'Predicted'),
               ('score', 'Softmax score of the prediction')])
    else:
        print('The selected head made no test errors in this run, so there is no mistake to inspect.')
    pair = paired['representative_confusion_pair']
    print('Most frequent confusion (reference -> predicted):',
          f'{names[pair[0]]} -> {names[pair[1]]}' if pair else 'none')

def show_noise(root=None):
    table(read_output('noise_summary.json', root), [('condition', 'Condition'), ('accuracy', 'Accuracy'),
        ('macro_f1', 'Macro-F1'), ('changed_predictions', 'Changed predictions vs matched clean')])
    rows = read_output('noise_records.json', root)
    clean = {r['id']: r['predicted'] == r['label'] for r in rows if r['condition'] == 'clean'}
    controls, paired = [], []
    for condition in ('clean', '20', '10', '0'):
        subset = [r for r in rows if r['condition'] == condition]
        snr = [r['actual_snr_db'] for r in subset if r['actual_snr_db'] is not None]
        gains = [r['gain'] for r in subset]
        controls.append({'condition': condition, 'clips': len(subset),
                         'snr': f'{min(snr):.2f} to {max(snr):.2f}' if snr else 'none added (matched clean)',
                         'gain': f'{min(gains):.4f} to {max(gains):.4f}', 'peak': max(r['peak'] for r in subset),
                         'clipped': sum(r['clipped_samples'] for r in subset)})
        if condition != 'clean':
            counts = {'corrected': 0, 'new_errors': 0, 'unchanged_correct': 0, 'unchanged_incorrect': 0}
            for r in subset:
                was, now = clean[r['id']], r['predicted'] == r['label']
                counts[('unchanged_correct' if was else 'corrected') if now
                       else ('new_errors' if was else 'unchanged_incorrect')] += 1
            paired.append({'condition': condition, **counts})
    table(controls, [('condition', 'Condition'), ('clips', 'Clips'), ('snr', 'Measured SNR (dB), min to max'),
                     ('gain', 'Common gain, min to max'), ('peak', 'Largest peak'), ('clipped', 'Clipped samples')])
    table(paired, [('condition', 'Noisy condition vs matched clean, same clips'), ('corrected', 'Corrected errors'),
                   ('new_errors', 'New errors'), ('unchanged_correct', 'Unchanged correct'),
                   ('unchanged_incorrect', 'Unchanged incorrect')])
    print(f'These are {len(clean)} reused validation clips (two per class), not new held-out evidence.')

def show_figures(pattern, limit=4):
    paths = sorted((ROOT / 'outputs').glob(pattern), key=lambda p: ({'clean': 0, '20': 1, '10': 2, '0': 3}.get(p.stem.rsplit('_', 1)[-1], 4), p.name))[:limit]
    if not paths:
        raise RuntimeError('Expected learning figures were not produced: ' + pattern)
    for path in paths:
        print(path.name)
        display(Image(filename=str(path)))

def play_examples(pattern, limit=2):
    paths = sorted((ROOT / 'outputs').glob(pattern), key=lambda p: ({'clean': 0, '20': 1, '10': 2, '0': 3}.get(p.stem.rsplit('_', 1)[-1], 4), p.name))[:limit]
    if not paths:
        raise RuntimeError('Expected listening examples were not produced: ' + pattern)
    print('Optional listening: start at low volume. Captions and figures convey the same comparisons.')
    for path in paths:
        print(path.stem)
        display(Audio(filename=str(path), normalize=False, autoplay=False))""", {'cellView': 'form'}),
    # md-05
    ('markdown', """## 2. Inspect real recordings and their roles

**Input:** all **400 real ESC-10 clips** [4], [5], selected from pinned ESC metadata. **System:** verified WAV decoding, deterministic mono float32 resampling from 44.1 kHz to 16 kHz, duplicate and original-recording checks. **Output:** an immutable split and listening examples.

| Role | Official folds | Clips | Per class | Purpose |
|---|---|---:|---:|---|
| Train | 1–3 | 240 | 24 | Fit the head and training-only baseline |
| Validation | 4 | 80 | 8 | Choose the epoch; later descriptive noise activity |
| Test | 5 | 80 | 8 | Evaluate the locked choice |

An original recording can supply several clips. Related `src_file` groups must stay in one role; 400 clips are not 400 independent recording sessions. This fixed split is a teaching protocol, not the official five-fold benchmark average. The fixed label order is **chainsaw, clock_tick, crackling_fire, crying_baby, dog, helicopter, rain, rooster, sea_waves, sneezing**.

A **waveform** shows amplitude over time. **Sample rate** counts observations per second: 16 kHz means 16,000 samples/second. A **spectrogram** shows how frequency content changes over time. Resampling changes the sampling grid; it does not make a low-quality recording contain more information. We do not peak-normalize, trim quiet regions or stretch time silently. A **fixed gain of 0.5** is applied to every default and BYOD waveform before resampling, including 16 kHz input. This provides consistent headroom for sinc-resampling overshoot without per-clip normalization or clipping. Original amplitudes and this gain remain recorded. The later noise experiment applies its separately recorded common gain after this preprocessing.

**Predict:** which categories might share similar rhythms or frequency patterns? Why could two excerpts of the same original recording make a random split misleading?""", {}),
    # code-06
    ('code', """run_stage('prepare')
prepared = read_output('prepare.json')
print('Class vocabulary:', prepared['class_names'])
print('Split counts:', prepared.get('roles', prepared.get('counts')))
print('Original-recording groups:', prepared['groups'])
show_figures('examples/*.png', limit=2)
play_examples('examples/*.wav', limit=2)""", {}),
    # md-07
    ('markdown', """**Notice:** real clip labels, seconds on the waveform and frequency/time axes on the spectrogram. Playback is optional, starts only when clicked and uses `normalize=False`; amplitude differences remain visible and audible. Begin at low volume. Plots and captions support participation without listening.

A log-mel spectrogram is a frequency summary, not the model's answer. AST receives a normalized filterbank tensor padded to **1,024 frames**. The model-input plot later marks the valid-time/padding boundary: padding is not newly observed silence.

## 3. Distinguish the original model from our new task

The pinned AST checkpoint was trained for **527 AudioSet labels**, using independent sigmoid scores: several events may be present together. This notebook attaches a new ten-class head to the frozen backbone and uses **softmax** for a single-label task. Its ten scores sum to one; this does not make them calibrated probabilities of correctness.

Those 527 AudioSet labels already include all ten ESC-10 categories: Chainsaw, Tick, Crackle and Fire, "Baby cry, infant cry", Dog, Helicopter, Rain, "Chicken, rooster", "Waves, surf" and Sneeze. The new head does not reuse those AudioSet outputs, but the frozen features it reads were trained to separate these same sounds, so expect a strong result from a small head. The original AudioSet classifier's own performance on these clips is not measured here, so beating the random head is not an improvement over pretrained AST.

The randomly initialized head is an **untrained task head**, not pretrained ESC-10 classification. We never match AudioSet names to ESC-10 using fuzzy strings or compare sigmoid scores directly with softmax scores. A closed vocabulary forces a choice even for an unfamiliar sound.

Three directly comparable systems use the same ten labels: a training-majority predictor, the seeded initial head, and the validation-selected trained head. Majority ties use fixed class order; the balanced training set gives no informed reason to favor the winning class.

## 4. Learn the metric before evaluating the model

Accuracy is the fraction of correct clips. For each class, **precision** is `TP/(TP+FP)`, **recall** is `TP/(TP+FN)`, and **F1** is `2TP/(2TP+FP+FN)`. Macro-F1 averages class F1 over the fixed vocabulary, giving each category equal weight. Undefined precision/recall are null with a reason, not perfect scores.

**Predict:** in a hypothetical sample with 90 dogs and 10 sneezes, what happens if a system always predicts dog? Is 90% accuracy enough?""", {}),
    # code-08
    ('code', """accuracy = 90 / 100
recall_sneezing = 0 / 10
f1_dog = 2 * 90 / (2 * 90 + 10 + 0)
f1_sneezing = 0.0
print({'accuracy': accuracy, 'sneezing_recall': recall_sneezing,
       'macro_f1': (f1_dog + f1_sneezing) / 2})""", {}),
    # md-09
    ('markdown', """<details><summary>Worked interpretation</summary>Accuracy is 0.90, sneezing recall is 0, and macro-F1 is about 0.474. High accuracy can conceal complete failure on a minority class. This is toy arithmetic, not an ESC-10 result; our default sample is balanced.</details>

## 5. Adapt the head and lock the checkpoint

**Input:** training clips and validation clips in disjoint original-recording groups. **Model:** frozen AST backbone; trainable classifier LayerNorm and linear layer. **Output:** initial and selected predictions, training history, measured updates and an exported head.

The frozen recipe is seed 42, ten epochs, batch four, AdamW learning rate `1e-3`, weight decay `0.01`, float32 cross-entropy, no augmentation or scheduler. Features are extracted in evaluation mode in batches of at most four. Detached feature caches bind inputs, processor, model, source and dtype. One optimizer continues across all ten epochs: the default expects **600 actual updates**.

Evaluate epoch zero, then each epoch. Select the highest **unrounded validation macro-F1**; earliest ties win, including epoch zero. Restore the chosen head before final test evaluation. Test labels never choose a checkpoint, learning rate or noise recipe.

**Predict:** must ten training epochs improve held-out accuracy? If epoch zero wins, how can we distinguish actual training from a skipped training stage?""", {}),
    # code-10
    ('code', """run_stage('train')
history = read_output('training_history.json')
table(history, [('epoch', 'Epoch'), ('train_loss', 'Training loss'), ('validation_macro_f1', 'Validation macro-F1')])
selection = read_output('selection.json')
show_selection()
show_metrics()
per_class = read_output('metrics.json')['selected']['test']['per_class']
table([dict(category=name, **values) for name, values in per_class.items()],
      [('category', 'Class'), ('precision', 'Precision'), ('recall', 'Recall'), ('f1', 'F1'), ('support', 'Test clips')])
print('Undefined precision means no predicted positives; see exported reasons.')
print('Actual updates (all epochs):', selection['optimizer_steps'], 'Maximum head change during training:', selection['max_head_delta_during_training'])
show_figures('model_input.png', limit=1)""", {}),
    # md-11
    ('markdown', """**Notice:** validation changes, per-class errors and initial-versus-selected test outcomes. The model-input figure separates measured sound from padding. The confusion matrix and the paired comparison of initial and selected predictions appear in Section 8, after the report stage.

**If several epochs tie:** on this sample, validation macro-F1 can reach 1.0 after the first epoch and stay there. Every tied epoch then looks equally good to validation, and the earliest-tie rule keeps the first one, so the exported head contains only that epoch's updates, not all of them. A saturated validation score means validation can no longer tell the epochs apart; it does not show that later epochs were worse. Read the test comparison with the untrained head conditionally. When a trained checkpoint is selected and improves the held-out comparison, that supports an improved mapping from these frozen features to the ten labels. If epoch zero is selected, the exported head is the unchanged initial head. If test performance worsens, report that outcome. Updates performed during training are not the same as updates contained in the selected head. Because the backbone was already fine-tuned on AudioSet labels covering all ten categories (Section 3), even an improvement is not evidence that the head's updates on 240 training clips taught the model to recognise these sounds.

Training evidence records exact trainable names/counts, actual updates, finite losses, pre-selection head changes and unchanged backbone hashes. Epoch-zero selection can have zero final delta even when training genuinely updated parameters. Improvement is not required for an honest result.

<details><summary>Interpretation hint</summary>Optimization fits training data; validation selects; test evaluates the fixed decision. Correlated clips from a source are not independent observations. Report corrected errors and new errors together, and never tune after seeing final test performance.</details>

## 6. Reload the selected artifact in a fresh process

**Input:** selected classifier `.pt`, SHA-256 manifest, exact base identity and ordered vocabulary. **System:** a new process using `weights_only=True`, schema/finiteness checks and fresh waveform-to-feature inference. **Output:** prediction and metric parity checks.

This is the existing classifier-only PyTorch artifact format, **not safetensors**. Base weights are not included. Wrong label order, base, activation, tensor schema or digest must be refused. No unrestricted pickle fallback is allowed.

Fixed tolerances: logits `atol=1e-5, rtol=1e-4`; probabilities `atol=1e-6, rtol=1e-4`; identical top-one labels/confusion counts; accuracy and macro-F1 drift at most `1e-12`.

**Predict:** does successful reload establish accurate sound recognition, or reproducibility of the saved behavior?""", {}),
    # code-12
    ('code', """run_stage('reload')
parity = read_output('reload_parity.json')
print({key: value for key, value in parity.items() if isinstance(value, (bool, int, float))})
print('Per-probe deviations and identities are retained in reload_parity.json.')""", {}),
    # md-13
    ('markdown', """**Notice:** raw-logit and probability differences as well as metric parity. A saved model can reproduce wrong answers perfectly. Preserve fixed tolerances and investigate a mismatch before qualification.

## 7. Change one thing: synthetic additive noise

**Input:** 20 validation clips, two per class selected by fixed filename order with distinct source groups where available. **Model:** the locked selected head. **Change:** clean / 20 / 10 / 0 dB signal-to-noise ratio (SNR).

A stable digest-derived seed creates one zero-mean Gaussian noise realization per clip, reused at all levels. With whole-clip signal RMS `r`, noise RMS is `r * 10**(-SNR/20)`. At 20 dB, noise RMS is one tenth of signal RMS; at 0 dB they are equal. This includes quiet regions and is not speech-active SNR.

One common gain per clip prevents clipping across all four conditions. The control is **gain-matched clean**, not original-amplitude clean. Noise level is the only intended changing factor: labels, duration, preprocessing, model and noise realization stay fixed. Each waveform gets newly computed features.

**Predict:** will macro-F1 necessarily decline at every noise level? Which confusion pairs might change? Write your prediction before running.""", {}),
    # code-14
    ('code', """run_stage('activity')
show_noise()
show_figures('noise_examples/*.png', limit=4)
play_examples('noise_examples/*.wav', limit=4)""", {}),
    # md-15
    ('markdown', """**Notice:** the control table shows the measured SNR range, the common gain, the largest peak and the clipped-sample count (zero is expected). The paired table splits each noisy condition's outcomes against matched clean into corrected errors, new errors, unchanged correct and unchanged incorrect. "Changed predictions" counts every change, including corrections and switches between two wrong classes, so it is not a count of new errors. Every waveform uses the same full-scale amplitude axis and every spectrogram the same labelled colour scale, so the conditions of one clip compare directly. Playback uses the actual scored signal and does not normalize each condition independently. The exported records preserve these controls.

<details><summary>Interpretation hint</summary>A noisy condition may improve some predictions. Keep that observation instead of choosing a different seed. This is one synthetic additive-noise experiment on reused validation clips; it supplies neither new held-out evidence nor proof of street, microphone or room-noise robustness. Selection stays locked.</details>

## 8. Export evidence and draw a bounded conclusion

The report contains full class logits/probabilities, per-class metrics, confusion counts, paired errors, deterministic examples, selected artifact, reload evidence, data attribution and measured time/memory. Tables are reloaded and archive checksums verified. Hashes and detailed traces live in files, keeping the learning flow readable.

The next cell then shows the confusion matrix (rows are reference labels and columns are predictions; normalized rows show where each true class goes, while raw counts retain support) and a paired comparison of the initial and selected heads on the same test clips: corrected errors, new errors, the highest-confidence mistakes and the most frequent confusion. These are the values the conclusion template below asks for.

A successful execution remains **Candidate** until the exact fresh hosted run and real labelled BYOD evidence are reviewed. It does not automatically approve a release.""", {}),
    # code-16
    ('code', """run_stage('report')
display(Markdown((ROOT / 'outputs/summary.md').read_text()))
show_figures('confusion.png', limit=1)
show_paired_errors()
display(FileLink(str(ROOT / 'outputs/sound_results.zip')))
session = {'bootstrap_seconds': BOOTSTRAP_SECONDS, 'elapsed_seconds': time.perf_counter() - SESSION_START,
           'source_files': CARRIED_HASHES, 'qualification': 'Candidate pending exact hosted and real BYOD review'}
(ROOT / 'session_summary.json').write_text(json.dumps(session, indent=2))
print('Setup seconds:', round(BOOTSTRAP_SECONDS, 1), 'Total elapsed seconds:', round(session['elapsed_seconds'], 1))""", {}),
    # md-17
    ('markdown', """**How precise are these numbers?** They come from one training seed, one test fold with 8 clips per class, and a noise activity with 2 clips per class and one noise realization each. Clips from one source recording are correlated, so the effective evidence is smaller than the clip count. One clip moves a class's test recall by 0.125 and an activity accuracy by 0.05; treat differences of that size as coarse, not as trends.

Use your measured results in this optional personal note:

> On [clips and independent source groups], the selected head obtained [accuracy and macro-F1] versus [baseline]. It corrected [count] initial errors and introduced [count] new errors. At [noise level], [paired result] changed. An error worth inspecting is [actual example]. These results do not establish [broader claim].

**Self-check:** explain a waveform versus a spectrogram; sigmoid versus softmax; training versus checkpoint selection; clip count versus independent groups; actual updates versus selected state; matched-clean control versus original amplitude; and reload parity versus predictive quality.

## 9. Optional: your own labelled recordings

Use only recordings you may process in third-party Colab. **Do not use confidential, restricted, personal or otherwise sensitive or regulated recordings** (for example identifiable voices) unless you are authorised to process them in this environment. **Data movement:** this notebook reads your files from the Colab runtime's disk and processes them there; it does not send them to any external service. Colab is itself Google-hosted, so the files sit on Google's infrastructure while the runtime exists. Provide a directory with `manifest.json`, ordered `class_names` and records with `id`, relative WAV `audio`, class `label`, `group_id`, and explicit `train`/`validation`/`test` role. Use **2–10 classes**, each with at least **4/2/2 clips** and **two independent groups per role**. Related excerpts share a group and stay in one role.

**Classes outside AudioSet:** the frozen backbone was supervised on AudioSet's 527 labels, which cover all ten ESC-10 categories. Classes outside AudioSet, or poorly represented in it, may score well below the ESC-10 results above, so do not expect those results to carry over to your classes.

Limits: at most 400 clips; WAV PCM16 or IEEE float32; 8–48 kHz; one or two channels; 0.5–10 seconds; 8 MiB/file; 1 MiB manifest; 4,000 seconds total. Stereo is downmixed by arithmetic mean. Longer-than-window audio is refused, not silently truncated. Headers, allocation bounds, finite normalized amplitudes, vocabulary, group isolation and decoded/resampled duplicates are checked before feature extraction. All-silent clips are refused.

This repeats **validation → baselines → training → selection → test → reload → noise → export** in a separate directory, preserving the canonical run. Private waveforms and spectrograms are excluded from shareable ZIPs by default; IDs, labels and metrics may remain. Review exports before sharing. Default Run all never opens an upload dialog.

**Placing your data:** open the Files pane (folder icon, left), create a folder such as `my_sounds`, and upload `manifest.json` and the WAV files into it, recreating any subfolders the manifest's `audio` paths name (or mount Google Drive and point at a folder there). Then set `BYOD_PATH = 'my_sounds'` and `RUN_BYOD = True` and run the next cell. If a recording or the manifest breaks one of these rules, the cell prints one `Refused:` line naming the record and the rule, then stops without an error; fix it and run the cell again. When it finishes, it shows your run's metrics, selected epoch, reload parity, noise summary and paired errors, then links the shareable ZIP.""", {}),
    # code-18
    ('code', """RUN_BYOD = False  # @param {type:"boolean"}
BYOD_PATH = ''  # @param {type:"string"}
if RUN_BYOD:
    if not BYOD_PATH.strip():
        raise ValueError('Set BYOD_PATH to the labelled directory containing manifest.json.')
    if run_stage('byod', '--byod', str(Path(BYOD_PATH).resolve())):
        byod_root = Path(read_output('byod_output.json')['root'])
        print('Results of your labelled run, read from', byod_root / 'outputs')
        show_metrics(byod_root)
        show_selection(byod_root)
        parity = read_output('reload_parity.json', byod_root)
        print('Reload parity:', {key: value for key, value in parity.items() if isinstance(value, (bool, int, float))})
        show_noise(byod_root)
        show_paired_errors(byod_root)
        display(FileLink(str(byod_root / 'outputs/sound_results.zip')))
else:
    print('Optional labelled BYOD disabled.')""", {}),
    # md-19
    ('markdown', """## 10. Optional: one recording without a label

The **ESC-10 head selected in Section 5** of this session scores one bounded WAV without a reference label. It always uses that head and its ten ESC-10 classes, even after a Section 9 run on your own recordings; the head trained on your recordings is not used here. Section 5 must be complete and current, otherwise the cell stops and names the cell to re-run. A WAV outside the Section 9 limits gets one `Refused:` line and no error. Its scores use the trained closed vocabulary. Unknown-sound detection is not provided; accuracy and F1 are **null / not measurable**. This inference option supplements the full labelled workflow and performs no new adaptation.""", {}),
    # code-20
    ('code', """RUN_UNLABELLED = False  # @param {type:"boolean"}
AUDIO_PATH = ''  # @param {type:"string"}
if RUN_UNLABELLED:
    if not AUDIO_PATH.strip():
        raise ValueError('Set AUDIO_PATH to a WAV within the documented limits.')
    if run_stage('infer', '--audio', str(Path(AUDIO_PATH).resolve())):
        print('Scored with the ESC-10 head selected in Section 5 of this session, not a Section 9 head.')
        result = read_output('inference.json')
        table([{'class': name, 'probability': value} for name, value in zip(result['class_names'], result['probabilities'], strict=True)],
              [('class', 'Known class'), ('probability', 'Softmax score')])
        print('Predicted:', result['predicted'], 'Accuracy/F1:', result['accuracy'], result['macro_f1'])
        print(result['reason'])
else:
    print('Optional unlabelled inference disabled.')""", {}),
    # md-21
    ('markdown', """## Troubleshooting, limitations and attribution

| Observation | Action |
|---|---|
| GPU missing or out of memory | Select a fresh T4 runtime. Do not silently skip feature extraction or training. |
| Download or package failure | Inspect the stage log; retry transient downloads while preserving pins. Never bypass integrity checks. |
| Group or duplicate refusal | Correct recording ownership/splits; do not silently discard records to retain target counts. |
| Epoch zero wins or test worsens | Preserve the result and investigate; do not tune on test labels. |
| Reload mismatch | Keep fixed tolerances and inspect raw outputs/schema before qualification. |
| Noise activity looks better | Report the counterexample; do not reroll the noise seed. |
| Own recordings refused | Fix WAV format, duration, amplitudes, grouping or class support before model execution. |

ESC-10 is a small, curated balanced dataset with correlated excerpts and known class-related preprocessing artifacts. The backbone's AudioSet labels include all ten ESC-10 categories, so these results measure a small head on closely matched pretrained features; upstream pretraining clip overlap has not been ruled out either. The fixed split does not reproduce the official five-fold benchmark. These results do not establish ecological monitoring, calibrated confidence, unknown-event detection or deployment robustness. Recordings can contain overlapping events despite one dataset label.

**Attribution:** ESC-10 (Karol J. Piczak, 2015) [4], [5], derived from Freesound recordings; pinned metadata and per-clip original-source attribution accompany the sample manifest. Upstream describes ESC-10 as CC BY 3.0, but its original per-clip attribution tags give **233 clips tagged CC0, 165 tagged CC BY and two tagged CC Sampling+**. Those 400 clips come from 304 unique source recordings: 180 tagged CC0, 122 CC BY and 2 CC Sampling+. These original terms remain visible; their relationship to the subset statement is unresolved and requires maintainer attribution review before release. Do not assume all recordings have identical terms or extend the subset statement to all ESC-50. AST uses the pinned MIT AudioSet checkpoint with its recorded BSD-3-Clause weight license; carried source retains its own license. Base weights are downloaded and are not bundled in the classifier export.

**AI Assistance Disclosure:** This notebook's code and explanations were developed with generative AI assistance under maintainer direction. The maintainer remains responsible for reviewing implementation, validating results and making release decisions. AI assistance does not constitute independent verification, provider endorsement or release approval.

## References

[1] Y. Gong, Y.-A. Chung, and J. Glass, "AST: Audio Spectrogram Transformer," in *Proc. Interspeech 2021*, 2021, pp. 571–575, doi: 10.21437/Interspeech.2021-698. [Online]. Available: https://doi.org/10.21437/Interspeech.2021-698

[2] MIT, "MIT/ast-finetuned-audioset-10-10-0.4593," Hugging Face model card, revision f826b80d28226b62986cc218e5cec390b1096902. [Online]. Available: https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593/tree/f826b80d28226b62986cc218e5cec390b1096902

[3] J. F. Gemmeke et al., "Audio Set: An ontology and human-labeled dataset for audio events," in *Proc. IEEE Int. Conf. Acoustics, Speech and Signal Processing (ICASSP)*, 2017, pp. 776–780, doi: 10.1109/ICASSP.2017.7952261. [Online]. Available: https://doi.org/10.1109/ICASSP.2017.7952261

[4] K. J. Piczak, "ESC: Dataset for environmental sound classification," in *Proc. 23rd ACM Int. Conf. Multimedia (MM '15)*, 2015, pp. 1015–1018, doi: 10.1145/2733373.2806390. [Online]. Available: https://doi.org/10.1145/2733373.2806390

[5] K. J. Piczak, "ESC-50: Dataset for environmental sound classification," GitHub repository, commit 33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6. [Online]. Available: https://github.com/karolpiczak/ESC-50/tree/33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6

Notebook specification followed: [DIMER notebook specification](https://github.com/kurtvalcorza/ml-worker/blob/main/integrations/dimer/fleet-specs/NOTEBOOK_SPEC.md). Exact source and dependency identities accompany the exported evidence.""", {}),
]


if __name__ == "__main__":
    raise SystemExit(main())
