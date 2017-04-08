import numpy as np
import tensorflow as tf

def tensorflow_FizzBuzz():
    NUM_DIGITS = 10

    def binary_encode(i, num_digits):
        #a >> b bビットを右シフト
        # 11 >> 1 [11を1bit右シフト]
        #000001011 → 11
        #000000101 → 5
        #& AND:論理和
        #iを2進数にして10桁にする
        #iの値に応じて10個の(0か1)要素をもった行列(1,10)をつくる(num_digits=10)
        return np.array([i >> d & 1 for d in range(num_digits)])

    def fizz_buzz_encode(i):
        if i % 15 == 0:#15で割り切れたら(余りが0なら)
            return np.array([0, 0, 0, 1])
        elif i % 5 == 0:#5で割り切れたら
            return np.array([0, 0, 1, 0])
        elif i % 3 == 0:#3で割り切れたら
            return np.array([0, 1, 0, 0])
        else:
            return np.array([1, 0, 0, 0])

    #101~1023(2の10乗-1)までの数字で訓練させる
    #101~1023を10桁の2進数に表現し、(101~1023,10)行列にする→(1行目)101の2進数10桁,(2行目)102の2進数10桁....
    trX = np.array([binary_encode(i, NUM_DIGITS) for i in range(101, 2 ** NUM_DIGITS)])
    #101~1023をFizzBuzzで4通りに振り分け、(101~1023,4)行列にしたもの
    trY = np.array([fizz_buzz_encode(i) for i in range(101, 2 ** NUM_DIGITS)])

    def init_weights(shape):
        #重みフィルタ作成
        #stddev 標準偏差
        #random_normal  Tensorを正規分布なランダム値で初期化する
        #e.g. 3x3x1のフィルタで畳込み、アウトプットを64チャネルにする
        #     weight = tf.random_normal([3, 3, 1, 64])
        return tf.Variable(tf.random_normal(shape, stddev=0.01))

    def model(X, w_h, w_o):
        #仮説関数
        #mutmul関数でテンソル同士の掛け算(行列の掛け算)
        #畳込み層のアクティベーション関数(活性化関数)としてReLuを使用する
        h = tf.nn.relu(tf.matmul(X, w_h))
        return tf.matmul(h, w_o)

    #訓練データを入れる変数:ノードの作成
    # X = [?,10] Y = [?,4] →trX,trYにあわせて行列をつくる
    # Noneとなっているのは訓練データをいくつでも入れられるようにするため
    X = tf.placeholder(tf.float32, [None, NUM_DIGITS])
    Y = tf.placeholder(tf.float32, [None, 4])

    #隠れ層のユニット数
    NUM_HIDDEN = 100

    # モデルパラメータ(入力層:10ノード, 隠れ層:100ノード, 出力層:4ノード)
    #shape = [10,100] 10x100行列
    w_h = init_weights([NUM_DIGITS, NUM_HIDDEN])
    #shape = [100,4] 100x4行列
    w_o = init_weights([NUM_HIDDEN, 4])

    py_x = model(X, w_h, w_o)

    #コスト関数:誤差の合計→平均
    #交差エントロピー
    #softmax_cross_entropy_with_logits = ↓
    #バックプロパゲーションの損失関数(誤差関数)として使用できる
    #多クラス識別(分類)問題→多クラス用クロスエントロピー
    #reduce_mean = 損失関数などで使用する平均を算出する関数
    # py_x = 仮説関数
    # Y = トレーニングデータ値
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=py_x, name=None))

    #勾配降下法(最急降下法) コスト関数を最小にすることが目標
    train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)

    #数学的には argmax f(x)を最大にするxの集合 <-> argmin f(x)を最小にするxの集合
    #tf.argmax:いくつかの軸に沿ったテンソルで最大値となるインデックスを一つ返す。
    #1に一番近いインデックス（予測）が正解とあっているか検証
    predict_op = tf.argmax(py_x, 1)

    def fizz_buzz(i, prediction):
        return [str(i), "fizz", "buzz", "fizzbuzz"][prediction]

    #バッチサイズ
    BATCH_SIZE = 128

    with tf.Session() as sess:
        #初期化
        tf.initialize_all_variables().run()
        # summary_op = tf.merge_all_summaries()
        # summary_writer = tf.train.SummaryWriter('fizzbuzz_data',graph = sess.graph)

        #訓練開始　
        #10000回繰り返す。
        for epoch in range(10000):
            p = np.random.permutation(range(len(trX)))
            trX, trY = trX[p], trY[p]

            for start in range(0, len(trX), BATCH_SIZE):
                end = start + BATCH_SIZE
                sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})

            #予測出力
            if epoch % 100 == 0:
                print(epoch, np.mean(np.argmax(trY, axis=1) == sess.run(predict_op,feed_dict={X: trX, Y: trY})))

        #学習データ出力
        numbers = np.arange(1, 101)
        teX = np.transpose(binary_encode(numbers, NUM_DIGITS))
        teY = sess.run(predict_op, feed_dict={X: teX})
        output = np.vectorize(fizz_buzz)(numbers, teY)

        print(output)
        return output

def fizzbuzz(n):
    if n % 3 == 0 and n % 5 == 0:
        return 'fizzbuzz'
    elif n % 3 == 0:
        return 'fizz'
    elif n % 5 == 0:
        return 'buzz'
    else:
        return str(n)

if __name__ == '__main__':
    tf_output = tensorflow_FizzBuzz()
    count = 0
    for i in range(1, 101):
        result = tf_output[i-1]
        answer = fizzbuzz(i)
        if result == answer:
            count += 1
    print(str(count) + '%')
