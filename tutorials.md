# [Project titanic](https://www.kaggle.com/c/titanic#tutorials)

目的 : タイタニックの生存者を予測する

幾つかのtutorialメモ
- その1 https://www.kaggle.com/startupsci/titanic-data-science-solutions

workflow
1. 問題の特定 
2. 学習用テスト用データの取得
3. データについて議論、準備、整形
4. データの分析、パターンの特定、調査
5. 分類器作成
6. 結果の可視化
7. 提出

workflow goals
- Classifying
  - 未知データのカテゴライズorクラス分類
- Correlating
  - 特徴の選択
  - 特徴と目的の相関を統計的に?
- Converting
  - 分類器や学習アルゴリズムに合わせたデータの整形
- Completing
  - 欠損値の扱いに気をつける
- Colrrecting
  - 外れ値や不正確性の高い特徴を省く
- Creating
  - 既存の特徴から新しい特徴を作り出す
- Charting


### 実際にデータを除く
質的データ?/量的データ?
> categorical features : Survived, Sex, and Embarked
> numerical features : Age, Fare

Mixed data typeの有無
数字と英文字が混ざるタイプのデータ
> Ticket : A/5 21171

Error or typoの有無
大きいデータセットは全て確認するのが非常に難しい、名前の項目等は多いかも?

空欄の有無
> Age, Cabin には Nan が
テスト用、学習用で数が異なるので要確認

データの型
integer, float, stringで構成される

##### データの分布を確認する
量的データの分布
- 全乗客(2224人)のうち約40%(891人)が学習データに含まれる
- Survived は 0 or 1
- 実際の生存率よりデータセットの生存率は6%高い
- 75%以上の乗客は親や子供がいない
- 30%近くの乗客は兄弟や配偶者が乗っていた
- 1%以下の乗客が非常に高い料金を払っている
- 高齢者の乗客は1%以下

質的データの分布
- Nameは被りが非常に少ない
- 乗客の65%は男性
- Cabinは殆どが同じような値
- Embarkedは3種類の値
- Ticketは22%が重複した値

### データ分布を確認してみて
- Ageの欠損値を埋める
- Embarkedは生存との相関が高そう
- Ticketは重複率が高いので特徴として使わない
- Cabinは高い不正確性、Nullが多く含まれるので使わない
- Passenger Idも相関がなさそう
- 乗客名も相関がなさそう

### データの基づく特徴の作成
- Parch(子供の数), sibsp(兄弟、配偶者の数)を元に家族数を作る
- Age -> Age bands
- Fare -> Fare range

### 仮説
- 女性の殆どは生き残る
- 子供の多くも生き残る
- グレードの高い部屋に入った人の多くは生き残る

---
各特徴への依存を見たいなら`注目特徴 vs Survived`で値を見ると良さそう
Sex vs Survivedで各性別毎にSurvived率の平均を見るとか

## 欠損値の補完
- 平均と標準偏差の間からランダムに選ぶ
- 中央値を使う
  - 全体の中央値でなく、性別、Pclass毎の中央値をそれぞれ使う


ここまでのメモ
- 全ての特徴を使う為に、数値に直すor新しい量的特徴に直す
  - その際にsuvived率を見る
    - あまり相関がなければ、特徴を別の区切りで見てみる
      - ex: 10代毎 -> 20代未満vs20代以上
      - ex: 家族人数 -> 家族居るvs居ない
    - n分割する時は`pandas.qcut`とか便利

## モデルで学習
classification or regressionのモデルを色々試す->決定木、ランダムフォレストが良い感じだった(86.7%)

#### 相関を見てる?
sklearn.linear_model.LogisticRegressionでは、`coef_`を使うと各パラメータの係数が確認できる。この係数が、大きいほどポジティブな確率が出る。ただ相関として見ることはできなさそう?

---
### Comments
- FamilySizeをそのまま使った方が良くない? (86.6 -> 88.55)
- agebandで4つ目のカテゴリ値を付け忘れ
- Completing a numerical continuous featureで2番目の方法が良さそう
- 35%の乗客が兄弟も一緒は違う感じ、自分の分析では16%、配偶者は17%の乗客が一緒だった
- ticket idとfareが同じ。

---

その2 : https://www.kaggle.com/helgejo/an-interactive-data-science-tutorial

その1との違いを主に書いていく
### データの相関を見る
- ヒートマップを使う
- 各特徴とsurvivalとの関連を見る
  - 性別vsSurvival
    - 男女毎に違いが見られたので有効っぽい特徴と考える

### カテゴリデータの用意
- 先の例と異なるやりかた
先程のサンプル
```
Embarked:
C -> 0
Q -> 1
S -> 2
```
今回のサンプル
```
C -> 1,0,0
Q -> 0,1,0
S -> 0,0,1
```
連続値ではなく各カテゴリを作る

客室は、アルファベットのイニシャルを使い9分割

家族人数は、Single, Small, Largeで分ける

### Feature importance
重要な特徴量の選択
- スクリプト実行した方が良さそう


---

その3 : https://www.kaggle.com/jeffd23/scikit-learn-ml-from-start-to-finish

これまでとの違い
- データのノーマライズをしている
[sklearn.preprocessing.LabelEncoder()](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)

- 分類器のパラメタをグリッドサーチする

---
その4 : https://www.kaggle.com/datacanary/xgboost-example-python/code/code


---
##### memo
http://smrmkt.hatenablog.jp/entry/2013/01/04/192628
便利な関数
- qcut
- rvm
- 特徴選択 http://qiita.com/nazoking@github/items/b9eb61f0c981af2cbdd0
- pipeline http://tkzs.hatenablog.com/entry/2016/06/26/093502
