import tensorflow as tf
import keras
from keras import layers


class TokenEmbedding(layers.Layer):
    def __init__(self, num_vocab=1000, max_len=100, num_hidden=64):
        super().__init__()
        self.embed = layers.Embedding(num_vocab, num_hidden)
        self.pos_embed = layers.Embedding(input_dim=max_len, output_dim=num_hidden)

    def call(self, x):
        max_len = tf.shape(x)[-1]
        x = self.embed(x)
        positions = tf.range(start=0, limit=max_len, delta=1)
        positions = self.pos_embed(positions)
        return x + positions


class SpeechFeatureEmbedding(layers.Layer):
    def __init__(self, num_hidden=64):
        super().__init__()
        self.conv1 = layers.Conv1D(
            num_hidden,
            kernel_size=11,
            strides=2,
            padding="same",
            activation="relu",
        )
        self.conv2 = layers.Conv1D(
            num_hidden,
            kernel_size=11,
            strides=2,
            padding="same",
            activation="relu",
        )
        self.conv3 = layers.Conv1D(
            num_hidden,
            kernel_size=11,
            strides=2,
            padding="same",
            activation="relu",
        )

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.conv3(x)


class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, ffn_dim, dropout_rate=0.1):
        super().__init__()
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim,
        )
        self.ffn = keras.Sequential(
            [
                layers.Dense(ffn_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, inputs, training=False):
        ATTENTION_OUT = self.attention(inputs, inputs)
        ATTENTION_OUT = self.dropout1(ATTENTION_OUT, training=training)
        OUT1 = self.layernorm1(inputs + ATTENTION_OUT)
        FFN_OUT = self.ffn(OUT1)
        FFN_OUT = self.dropout2(FFN_OUT, training=training)
        return self.layernorm2(OUT1 + FFN_OUT)


class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, ffn_neurons):
        super().__init__()
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)
        self.self_attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim,
        )
        self.cross_attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim,
        )
        self.self_dropout = layers.Dropout(0.5)
        self.cross_dropout = layers.Dropout(0.1)
        self.ffn_dropout = layers.Dropout(0.1)
        self.ffn = keras.Sequential(
            [
                layers.Dense(ffn_neurons, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )

    def mask(self, batch_size, n_dest, n_src, dtype):
        i = tf.range(n_dest)[:, None]
        j = tf.range(n_src)
        m = i >= j - n_src + n_dest
        mask = tf.cast(m, dtype)
        mask = tf.reshape(mask, [1, n_dest, n_src])
        mult = tf.concat(
            [
                tf.expand_dims(batch_size, -1),
                tf.constant([1, 1], dtype=tf.int32),
            ],
            0,
        )
        return tf.tile(mask, mult)

    def call(self, encoder_out, target):
        input_shape = tf.shape(target)
        batch_size = input_shape[0]
        sequence_len = input_shape[1]
        causal_mask = self.mask(batch_size, sequence_len, sequence_len, tf.bool)
        CAUSAL_ATTENTION = self.self_attention(
            target, target, attention_mask=causal_mask
        )
        CAUSAL_NORM = self.layernorm1(target + self.self_dropout(CAUSAL_ATTENTION))
        encoder_out = self.cross_attention(CAUSAL_NORM, encoder_out)
        CROSS_ATTENTION_NORM = self.layernorm2(
            self.cross_dropout(encoder_out) + CAUSAL_NORM
        )
        FFN_OUT = self.ffn(CROSS_ATTENTION_NORM)
        FFN_OUT_NORM = self.layernorm3(CROSS_ATTENTION_NORM + self.ffn_dropout(FFN_OUT))
        return FFN_OUT_NORM


class Transformer(keras.Model):
    def __init__(
        self,
        num_hidden=64,
        num_head=2,
        ffn_neurons=128,
        target_max_len=100,
        num_encoders=4,
        num_decoders=1,
        num_chars=10,
    ):
        super().__init__()
        self.loss_metric = keras.metrics.Mean(name="loss")
        self.num_encoders = num_encoders
        self.num_decoders = num_decoders
        self.target_max_len = target_max_len
        self.num_chars = num_chars
        self.encoder_input = SpeechFeatureEmbedding(
            num_hidden=num_hidden,
        )
        self.decoder_input = TokenEmbedding(
            num_vocab=num_chars,
            max_len=target_max_len,
            num_hidden=num_hidden,
        )
        self.encoder = keras.Sequential(
            [self.encoder_input]
            + [
                TransformerEncoder(num_hidden, num_head, ffn_neurons)
                for _ in range(num_encoders)
            ]
        )
        for i in range(num_decoders):
            setattr(
                self,
                f"dec_layer_{i}",
                TransformerDecoder(num_hidden, num_head, ffn_neurons),
            )
        self.classifier = layers.Dense(num_chars)

    def decode(self, encoder_out, target):
        y = self.decoder_input(target)
        for i in range(self.num_decoders):
            y = getattr(self, f"dec_layer_{i}")(encoder_out, y)
        return y

    def call(self, inputs):
        source = inputs[0]
        target = inputs[1]
        x = self.encoder(source)
        y = self.decode(x, target)
        return self.classifier(y)

    @property
    def metrics(self):
        return [self.loss_metric]

    def train_step(self, batch):
        source = batch["source"]
        target = batch["target"]
        decoder_input = target[:, :-1]
        decoder_target = target[:, 1:]
        with tf.GradientTape() as tape:
            predictions = self([source, decoder_input])
            one_hot = tf.one_hot(decoder_target, depth=self.num_chars)
            mask = tf.math.logical_not(tf.math.equal(decoder_target, 0))
            loss = self.compute_loss(None, one_hot, predictions, sample_weight=mask)
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.loss_metric.update_state(loss)
        return {"loss": self.loss_metric.result()}

    def test_step(self, batch):
        source = batch["source"]
        target = batch["target"]
        decoder_input = target[:, :-1]
        decoder_target = target[:, 1:]
        predictions = self([source, decoder_input])
        one_hot = tf.one_hot(decoder_target, depth=self.num_chars)
        mask = tf.math.logical_not(tf.math.equal(decoder_target, 0))
        loss = self.compute_loss(None, one_hot, predictions, sample_weight=mask)
        self.loss_metric.update_state(loss)
        return {"loss": self.loss_metric.result()}

    def generate(self, source, target_start_token_idx):
        batch_size = tf.shape(source)[0]
        encoded = self.encoder(source)
        decoder_input = (
            tf.ones((batch_size, 1), dtype=tf.int32) * target_start_token_idx
        )
        decoder_logits = []
        for _ in range(self.target_max_len - 1):
            decoder_out = self.decode(encoded, decoder_input)
            logits = self.classifier(decoder_out)
            logits = tf.argmax(logits, axis=-1, output_type=tf.int32)
            last_logit = tf.expand_dims(logits[:, -1], axis=-1)
            decoder_logits.append(last_logit)
            decoder_input = tf.concat([decoder_input, last_logit], axis=-1)
        return decoder_input


class DisplayOutputs(keras.callbacks.Callback):
    def __init__(
        self,
        batch,
        idx_to_token,
        target_start_token_idx=27,
        target_end_token_idx=28,
    ):
        self.batch = batch
        self.target_start_token_idx = target_start_token_idx
        self.target_end_token_idx = target_end_token_idx
        self.idx_to_char = idx_to_token

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 != 0:
            return
        source = self.batch["source"]
        target = self.batch["target"].numpy()
        bs = tf.shape(source)[0]
        preds = self.model.generate(source, self.target_start_token_idx)
        preds = preds.numpy()
        for i in range(bs):
            target_text = "".join([self.idx_to_char[_] for _ in target[i, :]])
            prediction = ""
            for idx in preds[i, :]:
                prediction += self.idx_to_char[idx]
                if idx == self.target_end_token_idx:
                    break
            print(f"target:     {target_text.replace('-','')}")
            print(f"prediction: {prediction}\n")


early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    verbose=1,
    restore_best_weights=True,
)


checkpoint_path = r"checkpoints\best_model.keras"
model_checkpoint = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    monitor="val_loss",
    save_best_only=True,
    verbose=1,
)


### Learning Rate Warmup followed by Linear Decay ###
class CustomSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self,
        init_lr=0.00001,
        lr_after_warmup=0.001,
        final_lr=0.00001,
        warmup_epochs=15,
        decay_epochs=85,
        steps_per_epoch=203,
    ):
        super().__init__()
        self.init_lr = init_lr
        self.lr_after_warmup = lr_after_warmup
        self.final_lr = final_lr
        self.warmup_epochs = warmup_epochs
        self.decay_epochs = decay_epochs
        self.steps_per_epoch = steps_per_epoch

    def calculate_lr(self, epoch):
        warmup_lr = (
            self.init_lr
            + ((self.lr_after_warmup - self.init_lr) / (self.warmup_epochs - 1)) * epoch
        )
        decay_lr = tf.math.maximum(
            self.final_lr,
            self.lr_after_warmup
            - (epoch - self.warmup_epochs)
            * (self.lr_after_warmup - self.final_lr)
            / self.decay_epochs,
        )
        return tf.math.minimum(warmup_lr, decay_lr)

    def __call__(self, step):
        epoch = step // self.steps_per_epoch
        epoch = tf.cast(epoch, "float32")
        return self.calculate_lr(epoch)