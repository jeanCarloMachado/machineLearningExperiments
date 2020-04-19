import tensorflow as tf
import grimoire as g


def run():
    const = tf.constant(2.0, name='const')

    b = tf.Variable(2.0, name='b')
    c = tf.Variable(1.0, name='c')

    d = tf.add(b, c, name='d')
    e = tf.add(c, const, name='e')
    a = tf.multiply(d, e, name='ap1')


    print(f"Variable is {a}")



g.d.execution_time(run)
