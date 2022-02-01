
import tensorflow as tf
import grimoire as g


from datetime import datetime




mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0


model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10),
])


predictions = model(x_train[:1]).numpy()


"""
## Probabilities of each class
"""
r = tf.nn.softmax(predictions).numpy()
r

"""
## Loss through the sparse categorical cross entropy
"""

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

result = loss_fn(y_train[:1], predictions).numpy()

result


"""
## Compile the model and fit it
"""
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
result =  model.fit(x_train, y_train, epochs=5)

a = id(result)

a
result



"""
## Evaluate
"""

evaluation_result = model.evaluate(x_test, y_test, verbose=2)

evaluation_result


"""
## Return a probability
"""


probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])

result = g.d.execution_time(probability_model, x_test[:5])

result
