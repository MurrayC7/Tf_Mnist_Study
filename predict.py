from PIL import Image, ImageFilter
import tensorflow as tf


def imageprepare(argv):
    """
    This function returns the pixel values.
    The input is a png file location.
    """
    im = Image.open(argv).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), 255)  # creates white canvas of 28x28 pixels

    if width > height:  # check which dimension is bigger
        # width is bigger. Width becomes 20 pixels.
        nheight = int(round((20.0 / width * height), 0))  # resize height according to ratio width
        if nheight == 0:  # rate case but minimum is 1 pixel
            nheight = 1
        # resize and sharpen
        img = im.resize((20, height), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round((28 - height) / 2, 0))  # caculate horizontal position
        newImage.paste(img, (4, wtop))  # paste resized image on white canvas
    else:
        # Height is bigger. Height becomes 20 pixels.
        nwidth = int(round((20.0 / height * width), 0))  # resize width according to ratio height
        if nwidth == 0:  # rare case but minimun is 1 pixel
            nwidth = 1
        # resize and sharpen
        img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round((28 - nwidth) / 2, 0))  # caculate vertical position
        newImage.paste(img, (wleft, 4))  # paste resized image on white canvas
    # newImage.save("sample.png")

    tv = list(newImage.getdata())  # get pixel values

    # normalize pixels to 0 and 1. 0 is pure white, 1 is pure blace.
    tva = [(255 - x) * 1.0 / 255.0 for x in tv]
    return tva


def predictint(imvalue):
    """
    This function returns the predicted integer.
    The input is the pixel values from the imageprepare() function
    """

    # Define the model (same as when creating the model file)
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    init_op = tf.initialize_all_variables()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init_op)
        saver.restore(sess, "./model.ckpt")
        # print ("Model restored.")

        prediction = tf.argmax(y, 1)
        return prediction.eval(feed_dict={x: [imvalue]}, session=sess)
