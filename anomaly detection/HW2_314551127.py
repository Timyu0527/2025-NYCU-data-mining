#!/usr/bin/env python3
"""Combined autoencoder training, clustering, and evaluation script."""

import argparse
import csv
import glob
import math
import os
import re
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402
import tensorflow as tf  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from tensorflow import keras  # noqa: E402
from tensorflow.keras import layers, models  # noqa: E402
from tensorflow.keras.utils import register_keras_serializable  # noqa: E402

keras.mixed_precision.set_global_policy("float32")

AUTOTUNE = tf.data.AUTOTUNE
NUM_CLASSES = 15


# -----------------------------------------------------------------------------
# Autoencoder utilities (from autoencoder.py)
# -----------------------------------------------------------------------------
def build_ssim_autoencoder(latent_ch=100, num_classes=15, input_shape=(128, 128, 1), weight_decay=1e-5):

    reg = keras.regularizers.l2(weight_decay)
    lrelu_slope = 0.2

    img_in = layers.Input(shape=input_shape, name="image")

    x = layers.Conv2D(32, 4, strides=2, padding="same", kernel_regularizer=reg)(img_in)
    x = layers.LeakyReLU(lrelu_slope)(x)
    x = layers.Conv2D(32, 4, strides=2, padding="same", kernel_regularizer=reg)(x)
    x = layers.LeakyReLU(lrelu_slope)(x)
    x = layers.Conv2D(32, 3, strides=1, padding="same", kernel_regularizer=reg)(x)
    x = layers.LeakyReLU(lrelu_slope)(x)
    x = layers.Conv2D(64, 4, strides=2, padding="same", kernel_regularizer=reg)(x)
    x = layers.LeakyReLU(lrelu_slope)(x)
    x = layers.Conv2D(64, 3, strides=1, padding="same", kernel_regularizer=reg)(x)
    x = layers.LeakyReLU(lrelu_slope)(x)
    x = layers.Conv2D(128, 4, strides=2, padding="same", kernel_regularizer=reg)(x)
    x = layers.LeakyReLU(lrelu_slope)(x)
    x = layers.Conv2D(64, 3, strides=1, padding="same", kernel_regularizer=reg)(x)
    x = layers.LeakyReLU(lrelu_slope)(x)
    x = layers.Conv2D(32, 3, strides=1, padding="same", kernel_regularizer=reg)(x)
    x = layers.LeakyReLU(lrelu_slope)(x)
    z = layers.Conv2D(latent_ch, 8, strides=1, padding="valid", kernel_regularizer=reg, name="latent")(x)

    y = layers.Conv2DTranspose(32, 8, strides=1, padding="valid", kernel_regularizer=reg)(z)
    y = layers.LeakyReLU(lrelu_slope)(y)
    y = layers.Conv2DTranspose(64, 3, strides=1, padding="same", kernel_regularizer=reg)(y)
    y = layers.LeakyReLU(lrelu_slope)(y)
    y = layers.Conv2DTranspose(128, 4, strides=2, padding="same", kernel_regularizer=reg)(y)
    y = layers.LeakyReLU(lrelu_slope)(y)
    y = layers.Conv2DTranspose(64, 3, strides=1, padding="same", kernel_regularizer=reg)(y)
    y = layers.LeakyReLU(lrelu_slope)(y)
    y = layers.Conv2DTranspose(64, 4, strides=2, padding="same", kernel_regularizer=reg)(y)
    y = layers.LeakyReLU(lrelu_slope)(y)
    y = layers.Conv2DTranspose(32, 3, strides=1, padding="same", kernel_regularizer=reg)(y)
    y = layers.LeakyReLU(lrelu_slope)(y)
    y = layers.Conv2DTranspose(32, 4, strides=2, padding="same", kernel_regularizer=reg)(y)
    y = layers.LeakyReLU(lrelu_slope)(y)
    y = layers.Conv2DTranspose(32, 4, strides=2, padding="same", kernel_regularizer=reg)(y)
    y = layers.LeakyReLU(lrelu_slope)(y)
    out = layers.Conv2DTranspose(1, 3, strides=1, padding="same", activation="sigmoid", kernel_regularizer=reg, name="recon")(y)

    return keras.Model(inputs=img_in, outputs=out, name="ae_onehot")


@register_keras_serializable(package="custom")
def ssim_loss(y_true, y_pred):
    ssim_loss_value = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
    return ssim_loss_value


def load_data(dir_path, img_size, batch_size):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        directory=dir_path,
        labels=None,
        image_size=(img_size, img_size),
        color_mode="grayscale",
        batch_size=batch_size,
    ).map(lambda x: (tf.cast(x, tf.float32) / 255.0, tf.cast(x, tf.float32) / 255.0)).cache().prefetch(AUTOTUNE)

    return train_ds


def list_images(root, exts=(".png", ".jpg", ".jpeg", ".bmp")):
    files = []
    for e in exts:
        files += glob.glob(os.path.join(root, f"**/*{e}"), recursive=True)

    def extract_number(path):
        base = os.path.basename(path)
        match = re.search(r"\d+", base)
        return int(match.group()) if match else 0

    return sorted(files, key=extract_number)


def load_image(path, size=128):
    img = Image.open(path).convert("L")
    img = ImageOps.exif_transpose(img)
    img = img.resize((size, size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, -1)


def decode_gray(path: tf.Tensor, image_size: int) -> tf.Tensor:
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=1, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [image_size, image_size], method=tf.image.ResizeMethod.BILINEAR)
    return img


def augment_pair(x, y):
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)
    k = tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32)
    x = tf.image.rot90(x, k)
    return x, y


def evaluate_model(model, test_root, outdir="eval", size=128):
    os.makedirs(outdir, exist_ok=True)
    paths = list_images(test_root)
    scores = []
    for p in paths:
        x = load_image(p, size)
        x_in = np.expand_dims(x, 0)
        xhat = model.predict(x_in, verbose=0)
        x32 = tf.convert_to_tensor(x_in, dtype=tf.float32)
        y32 = tf.convert_to_tensor(xhat, dtype=tf.float32)
        ssim = tf.image.ssim(x32, y32, max_val=1.0)
        score = float(1.0 - ssim.numpy().mean())
        scores.append(score)

        base = os.path.splitext(os.path.basename(p))[0]
        recon = (xhat.squeeze() * 255).astype(np.uint8)
        Image.fromarray(recon).save(os.path.join(outdir, f"{base}_recon.png"))

    with open(os.path.join(outdir, "texture_scores.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("path", "mean_anomaly"))
        writer.writerows(zip(paths, [f"{s:.6f}" for s in scores]))
    return paths, scores


def otsu_threshold(values, n_bins=256):
    v = np.asarray(values)
    v = (v - v.min()) / (v.max() - v.min() + 1e-12)
    hist, edges = np.histogram(v, bins=n_bins, range=(0, 1))
    prob = hist / hist.sum()
    omega = np.cumsum(prob)
    bins = (edges[:-1] + edges[1:]) / 2.0
    mu = np.cumsum(prob * bins)
    mu_t = mu[-1]
    sigma_b2 = (mu_t * omega - mu) ** 2 / (omega * (1 - omega) + 1e-12)
    return float(bins[np.nanargmax(sigma_b2)])


def write_submission(scores, out_csv, thresh=None, method="otsu", percentile=95.0, mean_k=2.0):
    if thresh is None:
        if method == "otsu":
            thresh = otsu_threshold(scores)
        elif method == "percentile":
            thresh = float(np.percentile(scores, percentile))
        elif method == "meanstd":
            thresh = float(np.mean(scores) + mean_k * np.std(scores))
    preds = [1 if s >= thresh else 0 for s in scores]
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("id", "prediction"))
        for i, p in enumerate(preds):
            writer.writerow((i, p))
    print(f"[SUBMIT] {out_csv} saved (threshold={thresh:.6f})")


def predict_test_and_submit(
    model,
    test_root,
    outdir,
    out_csv,
    image_size,
    batch=64,
    thresh=None,
    method="otsu",
    percentile=95.0,
    mean_k=2.0,
):
    os.makedirs(outdir, exist_ok=True)

    paths = list_images(test_root)

    ds = tf.data.Dataset.from_tensor_slices(tf.constant(paths, dtype=tf.string))
    ds = ds.map(lambda p: decode_gray(p, image_size), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch).prefetch(AUTOTUNE)

    scores = []
    for x in ds:
        xhat = model.predict(x, verbose=0)
        x32 = tf.cast(x, tf.float32)
        xhat32 = tf.cast(xhat, tf.float32)
        ssim = tf.image.ssim(x32, xhat32, max_val=1.0)
        loss_b = 1.0 - ssim
        scores.extend(loss_b.numpy().tolist())

    with open(os.path.join(outdir, "texture_scores.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("path", "mean_anomaly"))
        for p, s in zip(paths, scores):
            writer.writerow((p, f"{s:.6f}"))

    if thresh is None:
        if method == "otsu":
            thresh = otsu_threshold(scores)
        elif method == "percentile":
            thresh = float(np.percentile(scores, percentile))
        elif method == "meanstd":
            thresh = float(np.mean(scores) + mean_k * np.std(scores))
        else:
            raise ValueError("unknown threshold method")

    preds = [1 if s >= thresh else 0 for s in scores]

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("id", "prediction"))
        for i, p in enumerate(preds):
            writer.writerow((i, p))

    print(f"[EVAL] wrote scores to {outdir}/texture_scores.csv")
    print(f"[SUBMIT] {out_csv} saved. threshold={thresh:.6f}")

    return paths, scores, preds, thresh


def visualize_recon(model, img_path, size=128, out_path="compare.png"):
    x = load_image(img_path, size)
    x_in = np.expand_dims(x, 0)
    xhat = model.predict(x_in, verbose=0)

    orig = x.squeeze()
    recon = np.clip(xhat.squeeze(), 0, 1)
    diff = np.abs(orig - recon)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    axes[0].imshow(orig, cmap="gray")
    axes[0].set_title("Original")
    axes[1].imshow(recon, cmap="gray")
    axes[1].set_title("Reconstructed")
    axes[2].imshow(diff, cmap="inferno")
    axes[2].set_title("Error map")
    for axis in axes:
        axis.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[VIS] saved {out_path}")


# -----------------------------------------------------------------------------
# Clustering utilities (from cluster.py)
# -----------------------------------------------------------------------------
def ssim_mse_loss(mse_weight=0.1):
    def loss(y_true, y_pred):
        mse = tf.reduce_mean(tf.math.squared_difference(y_true, y_pred))
        return 0.9 * ssim_loss(y_true, y_pred) + mse_weight * mse

    return loss


def build_autoencoder(input_shape=(128, 128, 1), latent_dim=128, wd=1e-5):
    reg = keras.regularizers.l2(wd)
    lrelu = 0.2
    h, _, c = input_shape

    inp = layers.Input(shape=input_shape)
    x = inp
    x = layers.Conv2D(32, 4, strides=2, padding="same", kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(lrelu)(x)

    x = layers.Conv2D(64, 4, strides=2, padding="same", kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(lrelu)(x)

    x = layers.Conv2D(128, 4, strides=2, padding="same", kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(lrelu)(x)

    x = layers.Conv2D(256, 4, strides=2, padding="same", kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(lrelu)(x)

    x = layers.Flatten()(x)
    z = layers.Dense(latent_dim, kernel_regularizer=reg, name="latent")(x)

    side = h // 16
    d = layers.Dense(side * side * 256, kernel_regularizer=reg)(z)
    d = layers.LeakyReLU(lrelu)(d)
    d = layers.Reshape((side, side, 256))(d)

    d = layers.Conv2DTranspose(128, 4, strides=2, padding="same", kernel_regularizer=reg)(d)
    d = layers.BatchNormalization()(d)
    d = layers.LeakyReLU(lrelu)(d)

    d = layers.Conv2DTranspose(64, 4, strides=2, padding="same", kernel_regularizer=reg)(d)
    d = layers.BatchNormalization()(d)
    d = layers.LeakyReLU(lrelu)(d)

    d = layers.Conv2DTranspose(32, 4, strides=2, padding="same", kernel_regularizer=reg)(d)
    d = layers.BatchNormalization()(d)
    d = layers.LeakyReLU(lrelu)(d)

    out = layers.Conv2DTranspose(c, 4, strides=2, padding="same", activation="sigmoid")(d)

    ae = keras.Model(inp, out, name="ae")
    enc = keras.Model(inp, z, name="encoder")
    return ae, enc


def infer_features(encoder, ds):
    feats = []
    for chunk in ds:
        z = encoder.predict(chunk, verbose=0)
        feats.append(z)
    return np.concatenate(feats, axis=0)


def clusters_fit(encoder, ds, k):
    z = infer_features(encoder, ds)
    z = z.astype(np.float64, copy=False)
    z = StandardScaler().fit_transform(z)
    print(f"[Feat] Z shape: {z.shape}, K={k}")

    pred = KMeans(n_clusters=k, n_init=100, random_state=42).fit_predict(z)
    return pred


def save_csv(pred, k):
    out = os.path.join("./", f"clusters_k{k}.csv")
    with open(out, "w", encoding="utf-8") as f:
        f.write("id,pred\n")
        for i, c in enumerate(pred):
            f.write(f"{i},{c}\n")
    print(f"[Save] Clusters file -> {out}")


def save_model(encoder):
    os.makedirs("saved_models", exist_ok=True)
    encoder.save("saved_models/encoder.keras")
    print("[Save] Encoder -> saved_models/encoder.keras")


# -----------------------------------------------------------------------------
# Training pipeline (from train.py)
# -----------------------------------------------------------------------------
def split_by_label(ds, batch_size):
    parts = [0] * NUM_CLASSES
    for c in range(NUM_CLASSES):
        part = (
            ds.filter(lambda x, y: tf.equal(y, c))
            .map(lambda x, y: (x, x))
            .cache()
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        parts[c] = part
    return parts


def relabel_sequential(labels):
    mapping = {}
    remapped = np.empty_like(labels, dtype=np.int32)
    next_idx = 0
    for idx, value in enumerate(labels):
        value = int(value)
        if value not in mapping:
            mapping[value] = next_idx
            next_idx += 1
        remapped[idx] = mapping[value]
    return remapped


def cluster_predict(path, ds):
    ck = os.path.join(path, "encoder.keras")
    encoder = keras.models.load_model(ck, custom_objects={"ssim_loss": ssim_loss})
    labels = clusters_fit(encoder, ds, NUM_CLASSES)
    labels = relabel_sequential(labels)
    return labels


def predict_test_by_cluster(
    test_root,
    encoder_dir,
    model_dir,
    outdir,
    out_csv,
    image_size,
    batch=64,
    thresh=None,
):
    paths = list_images(test_root)
    if not paths:
        raise ValueError(f"No test images found under {test_root}")

    os.makedirs(outdir, exist_ok=True)

    path_to_index = {p: i for i, p in enumerate(paths)}

    base_ds = tf.data.Dataset.from_tensor_slices(tf.constant(paths, dtype=tf.string))
    feature_ds = base_ds.map(lambda p: decode_gray(p, image_size), num_parallel_calls=tf.data.AUTOTUNE)
    feature_ds = feature_ds.batch(batch).prefetch(tf.data.AUTOTUNE)

    labels = cluster_predict(encoder_dir, feature_ds)
    labels = np.asarray(labels, dtype=np.int32)
    if labels.shape[0] != len(paths):
        raise RuntimeError("Mismatch between predicted labels and file list length")

    grouped_paths = [[] for _ in range(NUM_CLASSES)]
    for path, label in zip(paths, labels):
        grouped_paths[int(label)].append(path)

    scores = np.zeros(len(paths), dtype=np.float32)
    preds = np.zeros(len(paths), dtype=np.int32)
    cluster_thresholds = {}

    for cluster_idx, cluster_paths in enumerate(grouped_paths):
        if not cluster_paths:
            continue

        model_path = os.path.join(model_dir, f"best_{cluster_idx}.keras")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing autoencoder checkpoint for cluster {cluster_idx}: {model_path}")

        model = keras.models.load_model(model_path, custom_objects={"ssim_loss": ssim_loss})

        cluster_ds = tf.data.Dataset.from_tensor_slices(tf.constant(cluster_paths, dtype=tf.string))
        cluster_ds = cluster_ds.map(lambda p: decode_gray(p, image_size), num_parallel_calls=tf.data.AUTOTUNE)
        cluster_ds = cluster_ds.batch(batch).prefetch(tf.data.AUTOTUNE)

        cluster_scores = []
        for batch_x in cluster_ds:
            recon = model.predict(batch_x, verbose=0)
            batch_loss = 1.0 - tf.image.ssim(tf.cast(batch_x, tf.float32), tf.cast(recon, tf.float32), max_val=1.0)
            cluster_scores.extend(batch_loss.numpy().tolist())

        if isinstance(thresh, dict):
            thr = thresh.get(cluster_idx)
        elif isinstance(thresh, (list, tuple)):
            thr = thresh[cluster_idx] if cluster_idx < len(thresh) else None
        else:
            thr = thresh

        thr = otsu_threshold(cluster_scores)

        cluster_thresholds[cluster_idx] = thr

        cluster_csv = os.path.join(outdir, f"cluster_{cluster_idx}_scores.csv")
        with open(cluster_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(("path", "mean_anomaly"))
            for path, score in zip(cluster_paths, cluster_scores):
                writer.writerow((path, f"{score:.6f}"))

        for path, score in zip(cluster_paths, cluster_scores):
            idx = path_to_index[path]
            scores[idx] = score
            preds[idx] = 1 if score >= thr else 0

        del model

    summary_csv = os.path.join(outdir, "texture_scores.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("path", "cluster", "mean_anomaly"))
        for path, label, score in zip(paths, labels.tolist(), scores.tolist()):
            writer.writerow((path, int(label), f"{score:.6f}"))

    if out_csv:
        with open(out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(("id", "prediction"))
            for idx, pred in enumerate(preds.tolist()):
                writer.writerow((idx, pred))

    return {
        "paths": paths,
        "labels": labels.tolist(),
        "scores": scores.tolist(),
        "predictions": preds.tolist(),
        "thresholds": cluster_thresholds,
    }


def run_train(argv):
    parser = argparse.ArgumentParser(description="TF3 SSIM Autoencoder")
    parser.add_argument("--data_dir", default="Dataset")
    parser.add_argument("--latent", type=int, default=500, help="Encoder 潛在維度")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--outdir", default="runs/ssim_tf3")
    parser.add_argument("--ckpt", default="")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--submit_csv", default="")
    parser.add_argument("--thresh", type=float, default=None)
    parser.add_argument("--thresh_method", type=str, default="otsu")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--encoder_dir", default="saved_models", help="Path to trained encoder for clustering test data")
    args = parser.parse_args(argv)

    tr = os.path.join(args.data_dir, "train")
    te = os.path.join(args.data_dir, "test")

    os.makedirs(args.outdir, exist_ok=True)

    if not args.eval:
        models = [
            build_ssim_autoencoder(
                args.latent,
                input_shape=(args.img_size, args.img_size, 1),
                weight_decay=args.weight_decay,
            )
            for _ in range(NUM_CLASSES)
        ]

        for model in models:
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=args.lr),
                loss=ssim_loss,
                jit_compile=False,
            )

        train_ds = load_data(tr, args.img_size, None)
        features = train_ds.map(lambda x, y: x).batch(args.batch)

        labels = cluster_predict(args.encoder_dir, features)
        print(labels)
        print(np.unique(np.array(labels)))

        counts = np.bincount(labels, minlength=NUM_CLASSES)

        labels = tf.convert_to_tensor(labels, dtype=tf.int32)
        labels = tf.data.Dataset.from_tensor_slices(labels)
        ds = tf.data.Dataset.zip((features.unbatch(), labels))
        per_class_ds = split_by_label(ds, args.batch)

        for idx, (ds_k, model, count) in enumerate(zip(per_class_ds, models, counts)):
            callbacks = [
                keras.callbacks.ModelCheckpoint(
                    os.path.join(args.outdir, f"best_{idx}.keras"), save_best_only=True, monitor="loss"
                )
            ]
            steps = math.ceil(count / args.epochs)
            model.fit(
                ds_k.repeat().prefetch(tf.data.AUTOTUNE),
                steps_per_epoch=steps,
                epochs=args.epochs,
                callbacks=callbacks,
            )

    else:
        eval_dir = os.path.join(args.outdir, "eval")
        submission_csv = args.submit_csv or os.path.join(eval_dir, "submission.csv")
        results = predict_test_by_cluster(
            test_root=te,
            encoder_dir=args.encoder_dir,
            model_dir=args.outdir,
            outdir=eval_dir,
            out_csv=submission_csv,
            image_size=args.img_size,
            batch=args.batch,
            thresh=args.thresh,
        )
        print(f"[EVAL] {len(results['paths'])} test samples processed")
        print(f"[EVAL] cluster thresholds -> {results['thresholds']}")


def run_cluster(argv):
    parser = argparse.ArgumentParser(description="Autoencoder + KMeans clustering")
    parser.add_argument("--data_dir", type=str, default="Dataset/train", help="影像根目錄；遞迴讀取")
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--latent", type=int, default=128, help="Encoder 潛在維度")
    parser.add_argument("--clusters", type=int, default=18, help="K 值（已預設 15）")
    args = parser.parse_args(argv)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        directory=args.data_dir,
        labels=None,
        image_size=(args.img_size, args.img_size),
        color_mode="grayscale",
        batch_size=args.batch,
    ).map(
        lambda x: (tf.cast(x, tf.float32) / 255.0, tf.cast(x, tf.float32) / 255.0),
        num_parallel_calls=AUTOTUNE,
    ).cache().prefetch(AUTOTUNE)

    ae, encoder = build_autoencoder((args.img_size, args.img_size, 1), args.latent)
    ae.compile(
        optimizer=keras.optimizers.Adam(2e-4),
        loss=ssim_mse_loss(mse_weight=0.1),
        jit_compile=False,
    )

    ae.fit(train_ds, epochs=args.epochs, verbose=1)

    num_classes = args.clusters
    train_x = load_data(args.data_dir, args.img_size, args.batch)
    train_x = train_x.map(lambda x, y: x)
    pred = clusters_fit(encoder, train_x, num_classes)
    counts = np.bincount(pred, minlength=num_classes)
    print(counts)

    save_csv(pred, num_classes)
    save_model(encoder)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0].lower() == "cluster":
        run_cluster(argv[1:])
    else:
        run_train(argv)


if __name__ == "__main__":
    main()
