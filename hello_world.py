import tensorflow as tf

#create placeholders with the names you want to display in tensorboard
x = tf.placeholder("float", name="X_name")
y = tf.placeholder("float", name = "Y_name")

addition = tf.add(x, y, name="addition_name")

#create the session
session = tf.Session()
with tf.Session() as session:
    result = session.run(addition, feed_dict={x: [5,2,1], y: [10,0,3]})

    #useful for creating graph in tensorboard
    writer = tf.summary.FileWriter("./logs/add_num_placeholder_name", session.graph)

    print(result)
