# 脳信号データセットを3D潜在空間ビューアで眺める: Triple-NとJapanEEG

<!--
Qiita title: 脳信号データセットを3D潜在空間ビューアで眺める: Triple-NとJapanEEG
Qiita tags: Python, Three.js, OpenNeuro, Neuroscience, DataVisualization
-->

脳信号データセットは、刺激画像、音声、脳活動ファイル、メタデータが別々に置かれていて、さらにサイズも大きいので、最初の「眺める」までが重くなりがちです。

そこで、刺激ごとの3D座標、軽量な刺激プレビュー、最小限のメタデータだけをブラウザに載せて、脳信号データセットを3D空間で眺められるビューアを作りました。

公開デモ:

https://itachisaigo.github.io/neuro-latent-3d-viewer/

コード:

https://github.com/itachisaigo/neuro-latent-3d-viewer

## 作ったもの

データセットをプルダウンで切り替えながら、PCA、t-SNE、UMAPなどの3D表現を眺められるThree.jsビューアです。

![Triple-N viewer](https://raw.githubusercontent.com/itachisaigo/neuro-latent-3d-viewer/main/docs/assets/triplen-viewer.png)

Triple-Nでは、自然画像を3D空間上の画像スプライトとして表示します。

![JapanEEG viewer](https://raw.githubusercontent.com/itachisaigo/neuro-latent-3d-viewer/main/docs/assets/japaneeg-viewer.png)

JapanEEGでは、音声イベントを点として表示します。点をクリックすると、その短い音声クリップを再生できます。

## 載せているデータセット

今の公開デモには、脳信号に紐づく2つのデータセットを載せています。

| データセット | 刺激 | 脳信号 | 表示 |
|:--|:--|:--|:--|
| Triple-N | 1,000枚の自然画像 | マカク視覚野のNeuropixelsスパイク応答 | 画像を3D空間に配置 |
| JapanEEG audio | 100個の日本語音声クリップ | ヒトEEGのイベント整列特徴 | 点をクリックして音声再生 |

### Triple-N

Triple-Nは、fMRI-guided Neuropixels recordingsのデータセットです。

ここで可視化している主な信号は、fMRIそのものではなくNeuropixels由来のスパイク応答です。fMRIは記録位置を決めるためのガイドとして使われています。

- ScienceDB: https://www.scidb.cn/en/detail?dataSetId=413ba7ddcb694bb2a534270caa90be36
- processed child DOI: `10.57760/sciencedb.31427`
- parent DOI: `10.57760/sciencedb.33556`

### JapanEEG audio

JapanEEGは、OpenNeuroで公開されているEEG、顔面EMG、音声を含む日本語音声データセットです。

今回は`task-listening`から100個の短い音声イベントを抽出し、それぞれに対応するEEG特徴と音響特徴を3D空間に落としました。

- OpenNeuro: https://openneuro.org/datasets/ds007808
- DOI: `10.18112/openneuro.ds007808.v1.0.0`
- License: CC0

## 何を3D化しているか

Triple-Nでは、自然画像ごとの神経応答ベクトルを作り、その画像を3D座標に落としています。

JapanEEGでは、同じ音声イベントに対して2種類の空間を用意しています。

| 表示 | 見ているもの |
|:--|:--|
| EEG PPCA / t-SNE / UMAP | 音声イベント中のEEG帯域パワー特徴の類似性 |
| Acoustic PPCA / t-SNE / UMAP | 音声波形から抽出した音響特徴の類似性 |

`Acoustic PPCA`は、波形画像をそのまま配置しているわけではありません。WAVから音量、ゼロ交差率、スペクトル重心、周波数帯域ごとのパワーなどを計算し、その特徴量を3次元に圧縮しています。

t-SNEとUMAPは近傍構造を見るために使っています。軸そのものは、PCAの主成分のようには解釈しません。

## 使った技術

ビューア本体はVite + Three.jsで作りました。前処理はPythonです。

脳信号や音響特徴をPCA、t-SNE、UMAPで3D座標に変換し、ブラウザ側ではデータセット切り替え、3D操作、クリック選択、音声再生だけを行います。

フロントエンドに載せるデータは、3D座標、サムネイル、短い音声クリップ、最小限のメタデータだけです。

## 何が面白いか

この形にすると、異なるモダリティの脳信号データセットを同じUIで扱えます。

- 画像刺激と視覚野応答
- 音声刺激とEEG応答
- 将来的には、画像、音声、言語、動画に対するfMRIやECoG
- さらにAIモデルの潜在表現

特に、同じ刺激集合に対して、脳信号由来の空間とAIモデル由来の空間を並べると面白そうです。

たとえば、

- Triple-Nの画像応答空間 vs Gemini / DINO / CLIPの画像特徴空間
- JapanEEGのEEG空間 vs 音声埋め込みモデルの空間
- VLMで画像と言語を同じ空間に置いたときの、脳応答との対応

のような比較ができます。

## まとめ

巨大な脳信号データセットでも、フロントエンドに載せるものを3D座標、刺激プレビュー、最小限のメタデータに絞れば、ブラウザで軽く探索できます。

今回は、Triple-Nの自然画像応答とJapanEEGの日本語音声応答を、同じ3Dビューア上で切り替えられるようにしました。

まず空間を回して眺めるUIを作ると、データセットの癖や次に掘るべき方向が見えやすくなります。

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
