# 脳信号データセットを3D潜在空間ビューアで眺める: Triple-NとJapanEEG

<!--
Qiita title: 脳信号データセットを3D潜在空間ビューアで眺める: Triple-NとJapanEEG
Qiita tags: Python, Three.js, OpenNeuro, Neuroscience, DataVisualization
-->

脳信号データセットは、刺激画像、音声、脳活動ファイル、メタデータが別々に置かれていて、さらにサイズも大きいので、最初の「眺める」までが重くなりがちです。

そこで、元データを丸ごとブラウザに載せるのではなく、

- 刺激ごとの3D座標
- 軽量な刺激プレビュー
- クリック時に見るための最小メタデータ
- 音声の場合は短いクリップ

だけをフロントエンドへ渡して、3D CADのように回せるビューアを作りました。

公開デモはこちらです。

https://itachisaigo.github.io/neuro-latent-3d-viewer/

コードはこちらです。

https://github.com/itachisaigo/neuro-latent-3d-viewer

## できたもの

データセットをプルダウンで切り替えながら、PCA、t-SNE、UMAPなどの3D表現を眺められるThree.jsビューアです。

![Triple-N viewer](https://raw.githubusercontent.com/itachisaigo/neuro-latent-3d-viewer/main/docs/assets/triplen-viewer.png)

Triple-Nでは、自然画像を3D空間上の画像スプライトとして表示します。

![JapanEEG viewer](https://raw.githubusercontent.com/itachisaigo/neuro-latent-3d-viewer/main/docs/assets/japaneeg-viewer.png)

JapanEEGでは、音声イベントを点として表示します。点をクリックすると、その音声クリップを再生できます。色は今回のビューア用に、由来するsource audio runをSpeaker A/Bとして分けています。公式の話者同定ラベルという意味ではありません。

## 載せているデータセット

今の公開デモには、脳信号に紐づく2つのデータセットだけを載せています。

| データセット | 刺激 | 脳信号 | ビューアでの表示 |
|:--|:--|:--|:--|
| Triple-N | 1,000枚の自然画像 | マカク視覚野のNeuropixelsスパイク応答 | 画像サムネイルを3D空間に配置 |
| JapanEEG audio | 100個の日本語音声クリップ | ヒトEEGのイベント整列特徴 | 話者/run別の点として表示し、クリックで再生 |

ビューア左上には、選択中データセットの一次リンク、DOI、ライセンスも表示するようにしました。スクリーンショットだけが切り出されても、元データに戻れるようにするためです。

## Triple-NはfMRIなのか、Neuropixelsなのか

ここは少し紛らわしいところです。

Triple-Nは、fMRI-guided Neuropixels recordingsのデータセットです。つまり、fMRIは記録位置を決めるためのガイドとして使われていて、今回ビューアで可視化している主な応答はNeuropixels由来のスパイク応答です。

今回使った出典は以下です。

- ScienceDB: https://www.scidb.cn/en/detail?dataSetId=413ba7ddcb694bb2a534270caa90be36
- processed child DOI: `10.57760/sciencedb.31427`
- parent DOI: `10.57760/sciencedb.33556`

公開デモには、元のMATファイルやRawデータは含めていません。ビューア用に作った3D座標とサムネイルだけを載せています。

## JapanEEG audio

JapanEEGは、OpenNeuroで公開されているEEG、顔面EMG、音声を含む日本語音声データセットです。

- OpenNeuro: https://openneuro.org/datasets/ds007808
- DOI: `10.18112/openneuro.ds007808.v1.0.0`
- License: CC0

今回は`task-listening`から100個の短い音声イベントを抽出し、それぞれに対応するEEG特徴と音響特徴を作りました。

公開デモには短いWAVクリップを含めていますが、元のEDF、元音声ファイル全体、OpenNeuroの大きなファイル一式は含めていません。

:::note warn
JapanEEGはCC0で公開されていますが、音声は人の声を含むデータです。再利用時はOpenNeuro上のデータページ、ライセンス、利用条件を確認し、必要以上に元データを複製しない方針にしています。
:::

## フロントエンドに載せるデータを小さくする

ブラウザに載せるJSONは、かなり単純な形にしています。

たとえばJapanEEGの1アイテムは、概念的にはこのような形です。

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

3D座標は、埋め込み手法ごとに別で持ちます。

```json
{
  "embeddings": {
    "eeg_pca": {
      "label": "EEG PPCA",
      "axes": ["PPCA 1", "PPCA 2", "PPCA 3"],
      "coordinates": [[2.74, -0.20, -2.62]]
    }
  }
}
```

高次元の応答行列、EDF、元音声、API埋め込みの生ベクトルは、フロントエンドには載せない方針にしました。

## 3D空間の作り方

### Triple-N

Triple-Nでは、自然画像ごとに神経応答ベクトルを作り、その画像を3D座標に落としています。

処理の流れは以下です。

1. `Processed_ses*.mat`から`response_best`を読む
2. 信頼性フィルタをかける
3. 画像ごとの応答ベクトルを作る
4. セッションをまたいでユニット方向に結合する
5. PCA、t-SNE、UMAPで3D化する
6. 画像サムネイルと座標だけをビューア用JSONに書き出す

PCAは大域的な分散を見るため、t-SNEとUMAPは近傍構造を見るために入れています。t-SNEやUMAPの軸そのものは、PCAの主成分のようには解釈しません。

### JapanEEG

JapanEEGでは、同じ音声イベントに対して、EEG由来の空間と音響特徴由来の空間を作っています。

| 表示 | 見ているもの |
|:--|:--|
| EEG PPCA / t-SNE / UMAP | 音声イベント中のEEG帯域パワー特徴の類似性 |
| Acoustic PPCA / t-SNE / UMAP | 音声波形から抽出した音響特徴の類似性 |

`Acoustic PPCA`は、波形そのものを画面上に置いているわけではありません。WAVから音量、ゼロ交差率、スペクトル重心、rolloff、周波数帯域ごとのパワーなどの特徴を計算し、その特徴量を3次元に圧縮しています。

`EEG PPCA`は、対応するEDFからイベント時間に合わせてEEGを切り出し、1-80 Hzの帯域パワー特徴を作って3次元に圧縮しています。

## UI

ビューア本体はVite + Three.jsです。

操作は、3D CADに近い感覚で使えるようにしました。

- OrbitControlsで回転、パン、ズーム
- Iso、Top、Front、Rightなどの視点プリセット
- Perspective / Orthographicの切り替え
- PCA、t-SNE、UMAPなどの空間切り替え
- データセット選択プルダウン
- 選択中データセットの出典リンク表示
- 点や画像をクリックしたときの詳細パネル
- 音声クリップのクリック再生

ビューアのマニフェストは、今はこのような形です。

```json
{
  "defaultDataset": "triple-n",
  "datasets": [
    {
      "id": "triple-n",
      "shortLabel": "Triple-N",
      "href": "pca3d.json",
      "sourceName": "ScienceDB",
      "sourceUrl": "https://www.scidb.cn/en/detail?dataSetId=413ba7ddcb694bb2a534270caa90be36",
      "doi": "10.57760/sciencedb.31427",
      "parentDoi": "10.57760/sciencedb.33556"
    },
    {
      "id": "japaneeg-audio",
      "shortLabel": "JapanEEG audio",
      "href": "japaneeg-audio/index.json",
      "sourceName": "OpenNeuro",
      "sourceUrl": "https://openneuro.org/datasets/ds007808",
      "doi": "10.18112/openneuro.ds007808.v1.0.0",
      "license": "CC0"
    }
  ]
}
```

新しいデータセットを足すときは、このマニフェストに1行追加し、対応するJSONを置けば同じUIで切り替えられるようにしています。

## 再現方法

リポジトリを取得します。

```bash
git clone https://github.com/itachisaigo/neuro-latent-3d-viewer.git
cd neuro-latent-3d-viewer
```

Python依存を入れます。

```bash
python3 -m pip install -r requirements.txt
```

ビューア側の依存を入れます。

```bash
cd viewer
npm install
cd ..
```

### Triple-Nの軽量ビューアデータを作る

Triple-N全体はかなり大きいので、Rawは落としません。processed、刺激画像、小さなhelperだけを使います。

```bash
node scripts/download_scidb_subset.mjs --processed --stimuli --helpers --jobs=8
unzip -n data/stimuli/StimuliNNN.zip -d data/stimuli/
```

ビューア用JSONとサムネイルを作ります。

```bash
python3 scripts/export_pca_3d_viewer_data.py \
  --processed-dir data/processed \
  --stimuli-dir data/stimuli \
  --output-dir viewer/public/viewer-data \
  --thumbnail-size 128
```

### JapanEEGの軽量ビューアデータを作る

JapanEEGも全体をローカルに落とす必要はありません。スクリプトはOpenNeuroの公開ファイル一覧を読み、必要な音声範囲をHTTP Range requestで取り出します。

```bash
python3 scripts/export_japaneeg_audio_viewer_data.py \
  --include-eeg \
  --embedding-source acoustic \
  --max-clips 100
```

この処理で作るものは以下です。

- 短い音声クリップ100個
- transcriptつきのメタデータ
- EEG特徴量から作った3D座標
- acoustic特徴量から作った3D座標

対応するEDFはローカルの`outputs/japaneeg_audio/`にキャッシュします。ただし、フロントエンドに載せるのは3D座標、短い音声クリップ、transcriptなどの軽いメタデータだけです。

### ビューアを起動する

```bash
cd viewer
npm run dev
```

ブラウザで開きます。

```text
http://127.0.0.1:5173/
```

## 公開時に気をつけたこと

### 元ファイルを記事に置かない

Qiita記事には、元のEDF、MAT、元音声、刺激画像アーカイブを添付していません。

公開デモにも、元データ一式は含めていません。含めているのは、ビューア用に軽量化したJSON、サムネイル、短い音声クリップです。

### 出典をUI内にも出す

記事本文に参考リンクを書くのとは別に、ビューア上にも選択中データセットの出典を表示しています。

スクリーンショットやデモだけを見た人が、ScienceDBやOpenNeuroの元ページへ戻れるようにするためです。

### APIキーを置かない

GeminiなどのAPIを使って画像や音声の潜在表現を作ることもできますが、APIキーはリポジトリにも記事にも入れません。

使う場合は環境変数から読む形にします。

```bash
export GEMINI_API_KEY="..."
```

### t-SNE / UMAPを解釈しすぎない

t-SNEとUMAPは見ていて面白いですが、軸や大域的距離を強く解釈しすぎないようにしています。局所的な近傍やクラスタの探索用として使うのがよさそうです。

## 何が面白いか

この形にすると、異なるモダリティの脳信号データセットを同じUIで扱えます。

- 画像刺激と視覚野応答
- 音声刺激とEEG応答
- 将来的には、画像、音声、言語、動画に対するfMRIやECoG
- さらにAIモデルの潜在表現

を、データセット選択と埋め込み手法の切り替えだけで比較できます。

特に次にやりたいのは、脳信号由来の空間とAIモデル由来の空間を同じ刺激集合で並べることです。

たとえば、

- Triple-Nの画像応答空間 vs Gemini/DINO/CLIPの画像特徴空間
- JapanEEGのEEG空間 vs 音声埋め込みモデルの空間
- VLMで画像と言語を同じ空間に置いたときの、脳応答との対応

のような比較ができます。

## まとめ

巨大な脳信号データセットでも、フロントエンドに載せるものを3D座標、刺激プレビュー、最小メタデータに絞れば、ブラウザで軽く探索できます。

今回は、Triple-Nの自然画像応答とJapanEEGの日本語音声応答を、同じ3Dビューア上で切り替えられるようにしました。

最初から大規模解析を始めるのではなく、まず空間を回して眺めるUIを作ると、データセットの癖や次に掘るべき方向が見えやすくなります。

## 参考

- 公開デモ: https://itachisaigo.github.io/neuro-latent-3d-viewer/
- GitHub: https://github.com/itachisaigo/neuro-latent-3d-viewer
- Triple-N ScienceDB: https://www.scidb.cn/en/detail?dataSetId=413ba7ddcb694bb2a534270caa90be36
- Triple-N processed DOI: https://doi.org/10.57760/sciencedb.31427
- Triple-N parent DOI: https://doi.org/10.57760/sciencedb.33556
- JapanEEG OpenNeuro: https://openneuro.org/datasets/ds007808
- JapanEEG DOI: https://doi.org/10.18112/openneuro.ds007808.v1.0.0
- Three.js: https://threejs.org/
- Vite: https://vite.dev/
