import tensorflow as tf

var  = tf.Variable(0)
holder = tf.placeholder(tf.int32)
add_op = tf.add(var,holder)
update_var = tf.assign(var,add_op)
mul_op = tf.mul(add_op,update_var)

with tf.Session() as sess:
    summary_writer = tf.train.SummaryWriter('./sample',graph=sess.graph)
    sess.run(tf.initialize_all_variables())

    result = sess.run(mul_op, feed_dict={ holder: 5 })

    print(result)
