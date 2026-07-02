# 脳信号データセットを3D潜在空間ビューアで眺める: Triple-NとJapanEEG

:::note warn
この記事では、データセットの元音声ファイル、EDFファイル、MATファイル、画像刺激ファイルは再配布しません。
OpenNeuro / ScienceDB 上の公開データから、ローカルで必要なサブセットと軽量ビューア用データを生成する方針です。
:::

## 何を作ったか

自然画像や音声刺激に対する脳信号を、3D散布図としてブラウザで眺めるビューアを作りました。

公開デモ:

```text
ここにGitHub PagesのURLを入れる
```

今のビューアには、脳信号に紐づく2つのデータセットだけを載せています。

| データセット | 刺激 | 脳信号 | ビューア上の表示 |
|:--|:--|:--|:--|
| Triple-N | 自然画像 | マカク視覚野のNeuropixelsスパイク応答 | 画像サムネイルを3D空間に配置 |
| JapanEEG | 日本語音声 | ヒトEEGのイベント整列帯域パワー | 音声イベントを話者/run別の点で表示し、クリックで音声再生 |

![Triple-N viewer screenshot](ここにQiitaへアップロードしたTriple-NスクリーンショットURLを入れる)

![JapanEEG audio viewer screenshot](ここにQiitaへアップロードしたJapanEEGスクリーンショットURLを入れる)

## 使ったデータセット

### Triple-N

Triple-N は、マカクが自然画像を見ているときの神経応答を記録したデータセットです。
このビューアでは、ScienceDB で公開されている processed データと刺激画像を使っています。

- Dataset: Triple-N
- ScienceDB: https://www.scidb.cn/en/detail?dataSetId=413ba7ddcb694bb2a534270caa90be36
- processed child dataset: `10.57760/sciencedb.31427`
- 使った信号: Neuropixels spiking responses
- 注意: `fMRI-guided` は記録位置のガイドであり、ここで可視化している主な応答はNeuropixels由来です

### JapanEEG

JapanEEG は、EEG、顔面EMG、音声を同期記録した日本語音声データセットです。
OpenNeuro `ds007808` として公開されています。

- Dataset: EEG-Speech Brain Decoding Dataset / JapanEEG
- OpenNeuro: https://openneuro.org/datasets/ds007808
- DOI: `10.18112/openneuro.ds007808.v1.0.0`
- License: CC0
- 今回の使用範囲: `task-listening` の一部runから100個の短い音声イベントを抽出

CC0なので著作権上の再利用制限はかなり緩いですが、音声は個人性を持ちうるデータです。
そのため、この記事では元音声やEDFを添付せず、再現手順だけを示します。

## ビューアの考え方

フロントエンドに重いデータは持たせません。
ブラウザに渡すのは、基本的に以下だけです。

- 3D座標
- 刺激の最小プレビュー
- クリック再生用の短い音声クリップ
- transcriptなどの軽いメタデータ

たとえば JapanEEG の1アイテムは概念的にはこんな形です。

```json
{
  "id": "japaneeg_0001",
  "audio": "audio/sub-02_ses-20241126_listening_run-04_event-0001.wav",
  "mediaType": "audio",
  "speaker": "Speaker A",
  "color": "#38bdf8",
  "transcript": "この場合に忘れてはならないのは",
  "eegOnset": 6.051,
  "duration": 2.172
}
```

3D座標は別に持ちます。

```json
{
  "embeddings": {
    "eeg_pca": {
      "label": "EEG PCA",
      "axes": ["PCA 1", "PCA 2", "PCA 3"],
      "coordinates": [[2.74, -0.20, -2.62]]
    }
  }
}
```

## 可視化している空間

### Triple-N

Triple-N では、各自然画像に対する神経応答ベクトルを作り、画像ごとに3D座標へ落としています。

今のビューアでは以下の切替を想定しています。

- PCA
- t-SNE
- UMAP

PCAは大域的な分散を見るため、t-SNEとUMAPは近傍構造を見るために使っています。
t-SNEやUMAPの軸そのものには、PCAのような解釈はしません。

### JapanEEG

JapanEEG では、同じ音声イベントに対して2種類の空間を用意しています。

| 表示 | 何の類似性か |
|:--|:--|
| EEG PCA / t-SNE / UMAP | 音声イベント中のEEG帯域パワー特徴の類似性 |
| Acoustic PCA / t-SNE / UMAP | 音声波形から抽出した音響特徴の類似性 |

`Acoustic PCA` は、波形そのものをそのまま3Dに置いているわけではありません。
`.wav` から、音量、ゼロ交差率、スペクトル重心、rolloff、周波数帯域ごとのパワーなどを計算し、その特徴量を3次元に圧縮しています。

一方 `EEG PCA` は、対応するEDFからイベント時間に合わせてEEGを切り出し、1-80 Hzの帯域パワー特徴を作って3次元に圧縮しています。

## 実装構成

```text
triple-n-pca/
├── scripts/
│   ├── export_pca_3d_viewer_data.py
│   ├── export_japaneeg_audio_viewer_data.py
│   └── viewer_export_utils.py
└── viewer/
    ├── src/
    │   ├── main.js
    │   └── styles.css
    └── public/
        └── viewer-data/
            ├── datasets.json
            ├── pca3d.json
            └── japaneeg-audio/
                └── index.json
```

ビューア本体は Three.js と Vite で作っています。
操作感は3D CAD寄りにして、Orbit / pan / zoom、Top / Front / Right / Iso などの視点切替を入れました。

## Triple-Nの軽量データを作る

Triple-N全体は大きいので、processed と刺激画像だけを使います。
Rawは落としません。

```bash
cd triple-n-pca

node scripts/download_scidb_subset.mjs --processed --stimuli --helpers --jobs=8
unzip -n data/stimuli/StimuliNNN.zip -d data/stimuli/
```

ビューア用JSONとサムネイルを作ります。

```bash
/Users/shioyakeisuke/.pyenv/shims/python3 scripts/export_pca_3d_viewer_data.py \
  --processed-dir data/processed \
  --stimuli-dir data/stimuli \
  --output-dir viewer/public/viewer-data \
  --thumbnail-size 128
```

## JapanEEGの軽量データを作る

JapanEEGは全体を落とすとかなり大きいです。
このスクリプトでは、OpenNeuroの公開ファイル一覧を読み、`task-listening` の小さめのrunを選び、HTTP Range requestで必要な音声部分だけを短いWAVとして切り出します。

```bash
/Users/shioyakeisuke/.pyenv/shims/python3 scripts/export_japaneeg_audio_viewer_data.py \
  --include-eeg \
  --embedding-source acoustic \
  --max-clips 100
```

この処理で作るものは以下です。

- 短い音声クリップ100個
- transcriptつきのメタデータ
- EEG特徴量から作った3D座標
- acoustic特徴量から作った3D座標

対応するEDFは `outputs/japaneeg_audio/` にキャッシュします。
ただし、フロントエンドに載せるのは3D座標と短い音声クリップだけです。

:::note warn
Qiita記事には、元のEDFや元音声ファイルを添付しません。
再現したい人はOpenNeuroから同じスクリプトで生成する形にします。
:::

## ビューアを起動する

```bash
cd viewer
npm install
npm run dev
```

起動後、ブラウザで以下を開きます。

```text
http://127.0.0.1:5173/
```

データセット選択は現在この2つに絞っています。

- Triple-N
- JapanEEG audio

## JapanEEGの点色

JapanEEG audio では、音声イベントを画像ではなく点で表示しています。
今回の100クリップは2つの元音声runから来ているため、便宜的に以下のように色分けしました。

| speaker label | 色 | 対応 |
|:--|:--|:--|
| Speaker A | blue | 1つ目のsource audio run |
| Speaker B | orange | 2つ目のsource audio run |

データセット側に明示的な話者ラベルがある場合は、この `speaker` フィールドだけを差し替えればUIはそのまま使えます。

## 気をつけたこと

### 元データを再配布しない

公開データであっても、音声や脳信号の元ファイルを記事に添付する必要はありません。
Qiitaにはスクリーンショット、コード、再現手順だけを載せます。

### APIキーを置かない

GeminiなどのAPIキーは絶対に記事にもリポジトリにも含めません。
APIを使う場合は環境変数から読む形にします。

```bash
export GEMINI_API_KEY="..."
```

### t-SNE / UMAPの軸を解釈しすぎない

t-SNEとUMAPは近傍構造を見るには便利ですが、軸そのものや距離の大域的な意味を強く解釈しないようにしています。

## まとめ

脳信号データセットは巨大になりがちですが、フロントエンドに載せる情報を3D座標、短い刺激プレビュー、メタデータに絞ると、かなり軽くインタラクティブに眺められます。

今回の構成では、自然画像に対するマカク視覚野応答と、日本語音声に対するヒトEEG応答を、同じ3Dビューア上で切り替えられるようにしました。

今後やるなら、以下を追加したいです。

- Triple-N側にもDINO / CLIP / Geminiなどの画像特徴空間を追加する
- JapanEEG側で音声埋め込みモデルの潜在空間を追加する
- EEG特徴を帯域パワーだけでなく、時間窓ごとの特徴や深層モデル特徴に変える
- データセットごとの出典・ライセンス表示をUI内にも出す

## 参考

- Qiita Markdown: https://help.qiita.com/ja/articles/qiita-markdown
- Markdown記法 チートシート: https://qiita.com/Qiita/items/c686397e4a0f4f11683d
- Triple-N ScienceDB: https://www.scidb.cn/en/detail?dataSetId=413ba7ddcb694bb2a534270caa90be36
- JapanEEG / OpenNeuro ds007808: https://openneuro.org/datasets/ds007808
- CC0 1.0 Universal: https://creativecommons.org/publicdomain/zero/1.0/
