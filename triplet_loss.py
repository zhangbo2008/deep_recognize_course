import logging

import keras.backend as K
import constants as c

alpha = c.ALPHA  # used in FaceNet https://arxiv.org/pdf/1503.03832.pdf


def batch_cosine_similarity(x1, x2):
    # https://en.wikipedia.org/wiki/Cosine_similarity
    # 1 = equal direction ; -1 = opposite direction #=注意我们之前数据读取时候已经经过正则化了.均值是0,方差是1.了所以这里面cos,直接就是dot就够了.
    dot = K.squeeze(K.batch_dot(x1, x2, axes=1), axis=1)
    return dot

#==========定义损失函数========用的是facenet里面的tripleloss
def deep_speaker_loss(y_true, y_pred):
    # y_true.shape = (batch_size, embedding_size)
    # y_pred.shape = (batch_size, embedding_size)
    # CONVENTION: Input is:
    # concat(BATCH_SIZE * [ANCHOR, POSITIVE_EX, NEGATIVE_EX] * NUM_FRAMES)
    # EXAMPLE:
    # BATCH_NUM_TRIPLETS = 3, NUM_FRAMES = 2
    # _____________________________________________________
    # ANCHOR 1 (512,)
    # ANCHOR 2 (512,)
    # ANCHOR 3 (512,)
    # POS EX 1 (512,)
    # POS EX 2 (512,)
    # POS EX 3 (512,)
    # NEG EX 1 (512,)
    # NEG EX 2 (512,)
    # NEG EX 3 (512,)
    # _____________________________________________________

    #elements = int(y_pred.shape.as_list()[0] / 3)
    elements = c.BATCH_SIZE
#========一共96个输出, 前32个anchor, 中间32个pos 后面32个neg
    anchor = y_pred[0:elements]
    positive_ex = y_pred[elements:2 * elements]
    negative_ex = y_pred[2 * elements:]

    sap = batch_cosine_similarity(anchor, positive_ex)
    san = batch_cosine_similarity(anchor, negative_ex)
    loss = K.maximum(san - sap + alpha, 0.0) # 最终的优化目标是拉近 a, p 的距离， 拉远 a, n 的距离 # 这个函数这么设计的原因 做maximum是防止过拟合. 只要 san 比sap 的相似度小0.2就足够区分正负样本了.就让损失直接变成0.不继续优化了.可以防止模型学的过好.
    total_loss = K.sum(loss)
    return total_loss
