from tensorflow.keras.losses import Loss
import tensorflow as tf


class WeightedBinaryCE(Loss):
    # initialize instance attributes
    def __init__(self, classWeights, labelSmoothing=0.0):
        super(WeightedBinaryCE, self).__init__()
        self.labelSmoothing = tf.constant(labelSmoothing, dtype=tf.dtypes.float32)
        self.classWeights = tf.constant(classWeights, dtype=tf.dtypes.float32)

    # Compute loss
    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        if self.labelSmoothing > 0:
            y_true = y_true * (1 - self.labelSmoothing) + self.labelSmoothing / 2
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred = tf.clip_by_value(
            y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon()
        )
        term_0 = tf.math.multiply(
            self.classWeights * tf.math.subtract(1.0, y_true),
            tf.math.log(tf.math.subtract(1.0, y_pred) + tf.keras.backend.epsilon()),
        )
        term_1 = tf.math.multiply(
            self.classWeights * y_true, tf.math.log(y_pred + tf.keras.backend.epsilon())
        )
        losses = term_0 + term_1
        return -tf.reduce_mean(losses, axis=0)


class DiceLoss(Loss):
    # initialize instance attributes
    def __init__(self):
        super(DiceLoss, self).__init__()

    # Compute loss
    def call(self, y_true, y_pred, smooth=1e-6):
        inputs = tf.squeeze(tf.cast(y_pred, tf.dtypes.float32))
        targets = tf.squeeze(tf.cast(y_true, tf.dtypes.float32))

        intersection = tf.math.reduce_sum(tf.math.multiply(targets, inputs))
        dice = (2 * intersection + smooth) / (
            tf.math.reduce_sum(targets) + tf.math.reduce_sum(inputs) + smooth
        )
        return 1 - dice


# α controls the amount of Dice term contribution in the loss f
# β ∈ [0, 1] controls the level of model penalization for false positives/negatives: when β is set to
# a value smaller than 0.5, FP are penalized more than FN
class WeightedComboLoss(Loss):
    # initialize instance attributes
    def __init__(self, labelWeights, alpha=0.5, beta=0.5):
        super(WeightedComboLoss, self).__init__()
        self.classWeights = tf.constant(labelWeights, dtype=tf.dtypes.float32)
        self.alpha = tf.constant(alpha, dtype=tf.dtypes.float32)
        self.beta = tf.constant(beta, dtype=tf.dtypes.float32)

    # Compute loss
    def call(self, y_true, y_pred, smooth=1e-6):
        inputs = tf.squeeze(tf.cast(y_pred, tf.dtypes.float32))
        targets = tf.squeeze(tf.cast(y_true, tf.dtypes.float32))

        intersection = tf.math.reduce_sum(tf.math.multiply(targets, inputs))
        dice = (2 * intersection + smooth) / (
            tf.math.reduce_sum(targets) + tf.math.reduce_sum(inputs) + smooth
        )

        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred = tf.clip_by_value(
            y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon()
        )
        term_0 = tf.math.multiply(
            (1 - self.beta) * self.classWeights * tf.math.subtract(1.0, y_true),
            tf.math.log(tf.math.subtract(1.0, y_pred) + tf.keras.backend.epsilon()),
        )
        term_1 = tf.math.multiply(
            self.beta * self.classWeights * y_true,
            tf.math.log(y_pred + tf.keras.backend.epsilon()),
        )
        losses = term_0 + term_1
        weightedCE = -tf.reduce_mean(losses, axis=0)

        combo = (self.alpha * weightedCE) - ((1 - self.alpha) * dice)

        return combo


class WeightedF1(tf.keras.metrics.Metric):
    def __init__(self, classWeights, threshold=0.5):
        super(WeightedF1, self).__init__()
        self.classWeights = tf.constant(classWeights, dtype=tf.dtypes.float32)
        self.threshold = tf.constant(threshold, dtype=tf.dtypes.float32)
        self.f1 = self.add_weight(name="f1", initializer="zeros")
        self.total = self.add_weight("total", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        m = tf.math.count_nonzero(tf.reduce_max(y_pred, axis=1))
        n = tf.shape(y_pred)[0]
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.math.greater_equal(y_pred, self.threshold)

        tp = tf.math.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
        tp = tf.cast(tp, self.dtype)
        tp = tf.math.multiply(tp, self.classWeights)
        tp = tf.cast(tp, self.dtype)

        tn = tf.math.logical_and(tf.equal(y_true, False), tf.equal(y_pred, False))
        tn = tf.cast(tn, self.dtype)
        tn = tf.math.multiply(tn, self.classWeights)
        tn = tf.cast(tn, self.dtype)

        fp = tf.math.logical_and(tf.equal(y_true, True), tf.equal(y_pred, False))
        fp = tf.cast(fp, self.dtype)
        fp = tf.math.multiply(fp, self.classWeights)
        fp = tf.cast(fp, self.dtype)

        fn = tf.math.logical_and(tf.equal(y_true, False), tf.equal(y_pred, True))
        fn = tf.cast(fn, self.dtype)
        fn = tf.math.multiply(fn, self.classWeights)
        fn = tf.cast(fn, self.dtype)

        pr = tf.math.divide(
            tf.math.reduce_sum(tp, axis=1),
            tf.math.reduce_sum(tp + fp + tf.keras.backend.epsilon(), axis=1),
        )
        m = tf.cast(m, pr.dtype)
        if m > 0:
            wPr = tf.math.reduce_sum(pr) / m
        else:
            wPr = tf.constant(0, tf.dtypes.float32)

        re = tf.math.divide(
            tf.math.reduce_sum(tp, axis=1),
            tf.math.reduce_sum(tp + fn + tf.keras.backend.epsilon(), axis=1),
        )
        n = tf.cast(n, re.dtype)
        wRe = tf.math.reduce_sum(re) / n

        res = tf.math.divide(2 * wPr * wRe, wPr + wRe + tf.keras.backend.epsilon())

        self.f1.assign_add(res)
        self.total.assign_add(1)

    def result(self):
        return tf.math.divide(self.f1, self.total)


class WeightedPrecision(tf.keras.metrics.Metric):
    def __init__(self, classWeights, threshold=0.5):
        super(WeightedPrecision, self).__init__()
        self.classWeights = tf.constant(classWeights, dtype=tf.dtypes.float32)
        self.threshold = tf.constant(threshold, dtype=tf.dtypes.float32)
        self.prec = self.add_weight(name="prec", initializer="zeros")
        self.total = self.add_weight("total", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        m = tf.math.count_nonzero(tf.reduce_max(y_pred, axis=1))
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.math.greater_equal(y_pred, self.threshold)

        tp = tf.math.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
        tp = tf.cast(tp, self.dtype)
        tp = tf.math.multiply(tp, self.classWeights)
        tp = tf.cast(tp, self.dtype)

        fp = tf.math.logical_and(tf.equal(y_true, True), tf.equal(y_pred, False))
        fp = tf.cast(fp, self.dtype)
        fp = tf.math.multiply(fp, self.classWeights)
        fp = tf.cast(fp, self.dtype)

        pr = tf.math.divide(
            tf.math.reduce_sum(tp, axis=1),
            tf.math.reduce_sum(tp + fp + tf.keras.backend.epsilon(), axis=1),
        )
        m = tf.cast(m, pr.dtype)
        if m > 0:
            wPr = tf.math.reduce_sum(pr) / m
        else:
            wPr = tf.constant(0, tf.dtypes.float32)

        self.prec.assign_add(wPr)
        self.total.assign_add(1)

    def result(self):
        return tf.math.divide(self.prec, self.total)


class WeightedRecall(tf.keras.metrics.Metric):
    def __init__(self, classWeights, threshold=0.5):
        super(WeightedRecall, self).__init__()
        self.classWeights = tf.constant(classWeights, dtype=tf.dtypes.float32)
        self.threshold = tf.constant(threshold, dtype=tf.dtypes.float32)
        self.rec = self.add_weight(name="rec", initializer="zeros")
        self.total = self.add_weight("total", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        n = tf.shape(y_pred)[0]
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.math.greater_equal(y_pred, self.threshold)

        tp = tf.math.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
        tp = tf.cast(tp, self.dtype)
        tp = tf.math.multiply(tp, self.classWeights)
        tp = tf.cast(tp, self.dtype)

        fn = tf.math.logical_and(tf.equal(y_true, False), tf.equal(y_pred, True))
        fn = tf.cast(fn, self.dtype)
        fn = tf.math.multiply(fn, self.classWeights)
        fn = tf.cast(fn, self.dtype)

        re = tf.math.divide(
            tf.math.reduce_sum(tp, axis=1),
            tf.math.reduce_sum(tp + fn + tf.keras.backend.epsilon(), axis=1),
        )
        n = tf.cast(n, re.dtype)
        wRe = tf.math.reduce_sum(re) / n

        self.rec.assign_add(wRe)
        self.total.assign_add(1)

    def result(self):
        return tf.math.divide(self.rec, self.total)


class WeightedAccuracy(tf.keras.metrics.Metric):
    def __init__(self, classWeights, threshold=0.5):
        super(WeightedAccuracy, self).__init__()
        self.classWeights = tf.constant(classWeights, dtype=tf.dtypes.float32)
        self.threshold = tf.constant(threshold, dtype=tf.dtypes.float32)
        self.acc = self.add_weight(name="acc", initializer="zeros")
        self.total = self.add_weight("total", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        n = tf.shape(y_pred)[0]
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.math.greater_equal(y_pred, self.threshold)

        tp = tf.math.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
        tp = tf.cast(tp, self.dtype)
        tp = tf.math.multiply(tp, self.classWeights)
        tp = tf.cast(tp, self.dtype)

        tn = tf.math.logical_and(tf.equal(y_true, False), tf.equal(y_pred, False))
        tn = tf.cast(tn, self.dtype)
        tn = tf.math.multiply(tn, self.classWeights)
        tn = tf.cast(tn, self.dtype)

        fp = tf.math.logical_and(tf.equal(y_true, True), tf.equal(y_pred, False))
        fp = tf.cast(fp, self.dtype)
        fp = tf.math.multiply(fp, self.classWeights)
        fp = tf.cast(fp, self.dtype)

        fn = tf.math.logical_and(tf.equal(y_true, False), tf.equal(y_pred, True))
        fn = tf.cast(fn, self.dtype)
        fn = tf.math.multiply(fn, self.classWeights)
        fn = tf.cast(fn, self.dtype)

        acc = tf.math.divide(
            tf.math.reduce_sum(tp + tn, axis=1),
            tf.math.reduce_sum(tp + fn + tn + fp + tf.keras.backend.epsilon(), axis=1),
        )
        n = tf.cast(n, acc.dtype)
        wAcc = tf.math.reduce_sum(acc) / n

        self.acc.assign_add(wAcc)
        self.total.assign_add(1)

    def result(self):
        return tf.math.divide(self.acc, self.total)
