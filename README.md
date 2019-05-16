# jdla_deep_learning

# Day1

## 活性化関数

活性化関数がないと線形回帰と同じような計算モデルになってしまうので、ニューラルネットワークでモデリングする必要性がなくなる。

- 恒等写像関数
  - 回帰問題で使用される
- ステップ関数
- シグモイド関数
  - 2クラスの分類問題で使用される
  - 0から1の間の値をとる
- tanh関数
  - RNNでよく使われる
- ReLU関数
  - DLで一般的な活性化関数
  - とりあえず最初に試すのによい
- LeakyReLU関数
- softplus関数
- Hardtanh関数
- ソフトマックス関数
  - 多クラスの分類問題で使用される
  - 出力層のノードの値を全て足すと1になる(正規化しているから)
  - exp(指数関数)をなぜ使うのか?

- 損失関数
  - 回帰問題では「2乗和誤差」が用いられる
  - 分類問題ではクロスエントロピー誤差が用いられる

# Day2

## ミニバッチ学習の特徴

#### バッチサイズを大きくする

- 勾配推定が安定する
- (GPUの)メモリ使用量が増えてしまう
- 重みの更新量が安定するので学習係数を大きくできる
  - つまり学習が早く進む

#### バッチサイズを小さくする

- 正則化の効果をもたらす（過学習を防げる）
- 更新回数が増えるため、計算時間が長くなる
- 局所的最小解にトラップされてしまうリスクを低減できる

## 微分

- 数値微分
  - 近似的に微分を求めているので誤差が生じる
  - 解析的微分に比べると計算時間も掛かる
  - 自分が実装したNNが正しく微分されているかを確かめるために使用することはある（勾配確認）
- 解析的微分
  - 公式を使って微分する
  - 誤差が含まれない
  - NNの学習ではこちらを使う

#### 勾配ベクトル

勾配ベクトルは偏微分をひとまとまりで表現したもの。

<a href="https://www.codecogs.com/eqnedit.php?latex=$$&space;f(x,y)&space;=&space;2x^2&plus;3y&space;$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$&space;f(x,y)&space;=&space;2x^2&plus;3y&space;$$" title="$$ f(x,y) = 2x^2+3y $$" /></a>

上記の数式が、

<a href="https://www.codecogs.com/eqnedit.php?latex=$$&space;\nabla&space;f&space;=(\frac{df}{dx},\frac{df}{dy})&space;=&space;(4x,&space;3)&space;$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$&space;\nabla&space;f&space;=(\frac{df}{dx},\frac{df}{dy})&space;=&space;(4x,&space;3)&space;$$" title="$$ \nabla f =(\frac{df}{dx},\frac{df}{dy}) = (4x, 3) $$" /></a>

のように∇(ナブラ)記号を用いて勾配ベクトルを表記する。

偏微分された各項がDLではパラメータである。

## 勾配法

- 勾配法
  - 勾配上昇法
    - NNやDLでは使わない（最大値を求めることをしないため）
  - 勾配降下法
    - 最小値を求める勾配法の総称
    - 最も急な方向に下る方法が「最急降下法」
    - 無作為に選び出したデータに対して勾配降下法を行うのが「確率的勾配降下法」
      - NNの学習ではこの確率的勾配降下法を使うのが普通
      
勾配法の数式は以下の通り。　※変数が2つの場合

<a href="https://www.codecogs.com/eqnedit.php?latex=$$&space;x_0&space;=&space;x_0&space;-&space;\eta\frac{df}{dx_0}&space;$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$&space;x_0&space;=&space;x_0&space;-&space;\eta\frac{df}{dx_0}&space;$$" title="$$ x_0 = x_0 - \eta\frac{df}{dx_0} $$" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=$$&space;x_1&space;=&space;x_1&space;-&space;\eta\frac{df}{dx_1}&space;\etaは更新量を表す&space;$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$&space;x_1&space;=&space;x_1&space;-&space;\eta\frac{df}{dx_1}&space;\etaは更新量を表す&space;$$" title="$$ x_1 = x_1 - \eta\frac{df}{dx_1} $$" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=$$&space;\eta&space;$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$&space;\eta&space;$$" title="$$ \eta $$" /></a>　は学習率を表している。

ニューラルネットワークのパラメータ（重みやバイアス）は学習によって自動で獲得されるパラメータであるが、学習率は人の手によって設定されるパラメータなのでハイバーパラメータとよばれる。

## 誤差逆伝播法

- 勾配降下法で最小値を求めるには各変数（パラメータ）の微分値（偏微分値）が必要になるが、パラメータは膨大な数なのでGPUを使っても現実的な時間で処理が終わらない。
- 誤差逆伝播法は出力層から入力層に導関数を伝播する手法でDLでは必須の技術。
- 連鎖律の原理を用いている
  - 一言で言うと「合成関数の微分」
