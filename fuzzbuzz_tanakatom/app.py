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
        #iの値に応じて10個の(0か1)要素をもった行列(10,1)をつくる(num_digits=10)
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

    trX = np.array([binary_encode(i, NUM_DIGITS) for i in range(101, 2 ** NUM_DIGITS)])
    trY = np.array([fizz_buzz_encode(i) for i in range(101, 2 ** NUM_DIGITS)])

    def init_weights(shape):
        #重みフィルタ作成
        #stddev 標準偏差
        #random_normal  Tensorを正規分布なランダム値で初期化する
        #e.g. 3x3x1のフィルタで畳込み、アウトプットを64チャネルにする
        #     weight = tf.random_normal([3, 3, 1, 64])
        return tf.Variable(tf.random_normal(shape, stddev=0.01))

    def model(X, w_h, w_o):
        #mutmul関数でテンソル同士の掛け算
        #畳込み層のアクティベーション関数としてReLuを使用する
        h = tf.nn.relu(tf.matmul(X, w_h))
        return tf.matmul(h, w_o)

    #訓練データを入れる変数
    # X = [,10] Y = [,4]
    # Noneとなっているのは訓練データをいくつでも入れられるようにするため
    X = tf.placeholder("float", [None, NUM_DIGITS])
    Y = tf.placeholder("float", [None, 4])

    NUM_HIDDEN = 100

    #shape = [10,100] 10x100行列
    w_h = init_weights([NUM_DIGITS, NUM_HIDDEN])
    #shape = [100,4] 100x4行列
    w_o = init_weights([NUM_HIDDEN, 4])

    py_x = model(X, w_h, w_o)

    #コスト関数:誤差の合計→平均
    #交差エントロピー
    #softmax_cross_entropy_with_logits = ↓
    #バックプロパゲーションの損失関数(誤差関数)として使用できる
    #reduce_mean = 損失関数などで使用する平均を算出する関数
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
    #勾配降下法 コスト関数を最小にすることが目標
    train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)

    #argmax f(x)を最大にするxの集合 <-> argmin f(x)を最小にするxの集合
    #テンソルの軸間で最大の値を持つインデックスを返します
    predict_op = tf.argmax(py_x, 1)

    def fizz_buzz(i, prediction):
        return [str(i), "fizz", "buzz", "fizzbuzz"][prediction]

    BATCH_SIZE = 128

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        # summary_op = tf.merge_all_summaries()
        # summary_writer = tf.train.SummaryWriter('fizzbuzz_data',graph = sess.graph)

        for epoch in range(10000):
            p = np.random.permutation(range(len(trX)))
            trX, trY = trX[p], trY[p]

            for start in range(0, len(trX), BATCH_SIZE):
                end = start + BATCH_SIZE
                sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})

            if epoch % 100 == 0:
                print(epoch, np.mean(np.argmax(trY, axis=1) == sess.run(predict_op,feed_dict={X: trX, Y: trY})))

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
