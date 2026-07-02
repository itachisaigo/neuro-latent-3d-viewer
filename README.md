# Neuro Latent 3D Viewer

Workspace for making lightweight 3D stimulus maps from neural responses and API-based AI latent spaces.

## What is here

- `code/Triple-N/`: optional local clone of the official MATLAB/demo/preprocessing code from `liyipeng-moon/Triple-N`. Ignored by git in this repo.
- `code/TripleNpy/`: optional local clone of Python reproduction code from `JustMuteAll/TripleNpy`. Ignored by git in this repo.
- `data/metadata/`: ScienceDB file metadata captured for the processed dataset.
- `data/processed/`: local target for `Processed_ses*.mat` files. These are ignored by git.
- `data/stimuli/`: local target for `StimuliNNN.zip` and extracted images. These are ignored by git.
- `scripts/download_scidb_subset.mjs`: downloads only the useful subset, not the multi-TB raw data.
- `scripts/plot_response_pca_images.py`: builds an image-on-PCA plot from processed responses.
- `scripts/export_triplen_api_viewer_data.py`: adds Gemini/Jina API image latent layouts to Triple-N.
- `scripts/export_nsd_person_viewer_data.py`: exports NSD/COCO person-image latent layouts.
- `scripts/export_narratives_audio_viewer_data.py`: exports click-to-play Narratives audio latent layouts.
- `scripts/export_japaneeg_audio_viewer_data.py`: exports click-to-play JapanEEG/OpenNeuro speech-listening clips.

## Data Choice

The parent Triple-N package is DOI `10.57760/sciencedb.33556`. Its processed child dataset is DOI `10.57760/sciencedb.31427`, ScienceDB `dataSetId=41f8bf18260c41a89f629cf580006e03`.

The full processed child dataset still reports about 585 GB because it also contains `Raw`. For this project, use only:

- `/V1/Processed`: 90 MAT files, about 603 MB total.
- `/V1/others/StimuliNNN.zip`: about 124 MB.
- small metadata helpers from `/V1/others`: `AreaXYZ.xlsx`, `exclude_area.xls`, `ClusInfo.mat`.

Avoid `/V1/Raw` unless you explicitly need H5/raw electrophysiology exports.

## Download

```bash
cd /Users/shioyakeisuke/Desktop/a.s.ist/triple-n-pca
node scripts/download_scidb_subset.mjs --processed --stimuli --helpers --jobs=8
unzip -n data/stimuli/StimuliNNN.zip -d data/stimuli/
```

## PCA Plot

Install Python packages if needed:

```bash
/Users/shioyakeisuke/.pyenv/shims/python3 -m pip install -r requirements.txt
```

Then run:

```bash
/Users/shioyakeisuke/.pyenv/shims/python3 scripts/plot_response_pca_images.py \
  --processed-dir data/processed \
  --stimuli-dir data/stimuli \
  --output outputs/response_pca_images.png
```

The script uses `response_best[:, :1000]`, z-scores each unit across images, concatenates units across sessions, computes image-wise PCA, and overlays a sampled set of thumbnails.

Thumbnail sampling is grid-balanced across the PCA plane, so center regions are represented instead of only the outer ring. To make the plot denser, increase `--thumbnails` and reduce `--zoom`, for example:

```bash
/Users/shioyakeisuke/.pyenv/shims/python3 scripts/plot_response_pca_images.py \
  --processed-dir data/processed \
  --stimuli-dir data/stimuli \
  --output outputs/response_pca_images_balanced.png \
  --thumbnails 260 \
  --zoom 0.40 \
  --thumbnail-grid-factor 2.4 \
  --point-alpha 0.12
```

On this Mac, `/Applications/Xcode.app/.../python3` currently has an incompatible NumPy install, so the pyenv Python above is the verified interpreter.

## 3D Viewer

Generate the 3D PCA coordinates and browser thumbnails:

```bash
/Users/shioyakeisuke/.pyenv/shims/python3 scripts/export_pca_3d_viewer_data.py \
  --processed-dir data/processed \
  --stimuli-dir data/stimuli \
  --output-dir viewer/public/viewer-data \
  --thumbnail-size 128
```

This writes one JSON file with three precomputed 3D embeddings:

- `PCA`: linear global variance view.
- `t-SNE`: nonlinear local-neighborhood view, computed after PCA pre-reduction.
- `UMAP`: nonlinear manifold/neighborhood view, computed after PCA pre-reduction.

Run the Three.js viewer:

```bash
cd viewer
npm install
npm run dev
```

The viewer uses a CAD-like camera surface:

- Embedding buttons: PCA, t-SNE, UMAP.
- Orbit/pan/dolly through Three.js `OrbitControls`.
- View presets: isometric, top, front, right, left, bottom.
- Zoom to fit and perspective/orthographic projection toggle.
- Click image sprites or points to inspect a stimulus and its current embedding coordinates.

For t-SNE and UMAP, axis numbers are embedding coordinates, not interpretable neural axes like PCA loadings. Use them for neighborhood/cluster inspection rather than global metric comparisons.

## GitHub Pages Deploy

The app can be deployed as a static Vite build through GitHub Pages. The workflow is already configured in `.github/workflows/deploy-pages.yml`.

1. Create a public GitHub repository for this `triple-n-pca` directory.
2. Push the repository to the `main` branch.
3. In GitHub, open `Settings > Pages`.
4. Set `Source` to `GitHub Actions`.
5. Run the `Deploy Viewer To GitHub Pages` workflow, or push to `main`.

Only the lightweight viewer assets under `viewer/public/viewer-data/` are meant to be published. Raw Triple-N MAT files, source stimulus archives, full JapanEEG EDF files, API embedding caches, and generated outputs under `outputs/` remain ignored by git.

The public app currently exposes two datasets:

- `Triple-N`
- `JapanEEG audio`

## API Latent Spaces

The browser only loads lightweight viewer data: 3D coordinates, stimulus previews, and minimal metadata. High-dimensional API embeddings are cached under `outputs/api_embeddings/` and are not loaded by the frontend.

Gemini is the default provider because `gemini-embedding-2` supports text, image, audio, video, and documents in one embedding space. Set an API key before running API exports:

```bash
export GEMINI_API_KEY="..."
```

Generate Triple-N brain-space plus Gemini image-latent layouts:

```bash
/Users/shioyakeisuke/.pyenv/shims/python3 scripts/export_triplen_api_viewer_data.py \
  --processed-dir data/processed \
  --stimuli-dir data/stimuli \
  --output-dir viewer/public/viewer-data/triplen-api
```

This adds a `Triple-N + AI` dataset to `viewer/public/viewer-data/datasets.json`. The dataset contains:

- `Brain PPCA`, `Brain t-SNE`, `Brain UMAP`
- `Gemini PPCA`, `Gemini t-SNE`, `Gemini UMAP`
- image thumbnails only, not raw response matrices or high-dimensional embeddings

To use Jina instead:

```bash
export JINA_API_KEY="..."
/Users/shioyakeisuke/.pyenv/shims/python3 scripts/export_triplen_api_viewer_data.py \
  --processed-dir data/processed \
  --stimuli-dir data/stimuli \
  --provider jina \
  --model jina-embeddings-v4 \
  --output-dir viewer/public/viewer-data/triplen-api
```

## NSD Person Images

Use either an image directory directly, or NSD HDF5 plus `nsd_stim_info_merged.csv` and COCO `instances_*.json` metadata.

```bash
/Users/shioyakeisuke/.pyenv/shims/python3 scripts/export_nsd_person_viewer_data.py \
  --images-dir /path/to/nsd_or_coco_images \
  --coco-instances /path/to/instances_train2017.json \
  --limit 1000
```

For the official NSD HDF5 route:

```bash
/Users/shioyakeisuke/.pyenv/shims/python3 scripts/export_nsd_person_viewer_data.py \
  --nsd-hdf5 /path/to/nsd_stimuli.hdf5 \
  --stim-info-csv /path/to/nsd_stim_info_merged.csv \
  --coco-instances /path/to/instances_train2017.json \
  --limit 1000
```

The output dataset is `NSD person` in the viewer manifest.

## Free Face Images

For a simple face-similarity PoC, put freely usable face images in one folder and run:

```bash
/Users/shioyakeisuke/.pyenv/shims/python3 scripts/export_face_images_gemini_viewer_data.py \
  --images-dir data/free_faces \
  --limit 500
```

The output dataset is `Face images` in the viewer manifest. The frontend receives only `Gemini PPCA`, `Gemini t-SNE`, `Gemini UMAP`, thumbnails, and item labels.

## Narratives Audio

After downloading OpenNeuro ds002345/Narratives locally, export short click-to-play clips:

```bash
/Users/shioyakeisuke/.pyenv/shims/python3 scripts/export_narratives_audio_viewer_data.py \
  --bids-root /path/to/ds002345 \
  --clip-seconds 6 \
  --max-clips 240
```

If you have already computed clip-aligned fMRI response features, add them as a `.npy` or CSV matrix with one row per clip:

```bash
/Users/shioyakeisuke/.pyenv/shims/python3 scripts/export_narratives_audio_viewer_data.py \
  --bids-root /path/to/ds002345 \
  --fmri-features /path/to/clip_fmri_features.npy
```

Without `--fmri-features`, the audio dataset shows Gemini audio-latent layouts only. With fMRI features, it also adds `fMRI PPCA`, `fMRI t-SNE`, and `fMRI UMAP`.

## JapanEEG Audio

JapanEEG/OpenNeuro `ds007808` is large, so this exporter does not download the full dataset. It reads the public file list, selects small `task-listening` runs, and uses HTTP Range requests to cut only the needed WAV clips.

```bash
/Users/shioyakeisuke/.pyenv/shims/python3 scripts/export_japaneeg_audio_viewer_data.py \
  --include-eeg \
  --embedding-source acoustic \
  --max-clips 100
```

The output dataset is `JapanEEG audio` in the viewer manifest. The exporter caches the matching EDF files and EEG feature matrix under `outputs/japaneeg_audio/`, but the frontend loads only 3D coordinates, short WAV clips, and transcripts. In the 3D view, JapanEEG audio is rendered as speaker-colored points instead of spectrogram sprites. If `GEMINI_API_KEY` is set, use `--embedding-source both` or `--embedding-source gemini` to add Gemini audio-latent layouts.
