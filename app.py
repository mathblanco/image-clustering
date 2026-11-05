import io

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from PIL import Image, ImageOps
from scipy.ndimage import binary_dilation, gaussian_filter
from scipy.spatial import ConvexHull
from scipy.ndimage import label as ndlabel
from skimage.color import rgb2lab
from skimage.measure import find_contours
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Photo Clustering", page_icon="🎨", layout="wide")

PALETTE = np.array([
    [230, 25, 75], [60, 180, 75], [0, 130, 200], [255, 225, 25], [145, 30, 180],
    [245, 130, 48], [70, 240, 240], [240, 50, 230], [170, 110, 40], [128, 128, 128],
    [0, 128, 128], [128, 0, 0], [170, 255, 195], [0, 0, 128], [255, 215, 180],
], dtype=np.uint8)


# ---------------------------------------------------------------- functions
def load_image(uploaded) -> Image.Image:
    img = Image.open(uploaded)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def extract_features(img_rgb: np.ndarray, color_space: str, use_pos: bool, pos_weight: float):
    h, w, _ = img_rgb.shape
    if color_space == "Lab":
        feats = rgb2lab(img_rgb / 255.0).reshape(-1, 3) / np.array([100, 128, 128])
    else:
        feats = (img_rgb / 255.0).reshape(-1, 3)
    if use_pos:
        yy, xx = np.mgrid[0:h, 0:w]
        pos = np.stack([xx.ravel() / max(w - 1, 1), yy.ravel() / max(h - 1, 1)], axis=1)
        feats = np.hstack([feats, pos * pos_weight])
    return feats


def assign_to_centroid(feats, centroids, block=20000):
    out = np.empty(len(feats), dtype=np.int32)
    for i in range(0, len(feats), block):
        d = ((feats[i:i + block, None, :] - centroids[None]) ** 2).sum(-1)
        out[i:i + block] = d.argmin(1)
    return out


def cluster_pixels(feats, k, method, linkage, n_sample, seed=42):
    rng = np.random.default_rng(seed)
    if method == "K-Means":
        modelo = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(feats)
        return modelo.labels_
    # Hierarchical: O(n²) memory -> fit on a sample and propagate to all pixels
    n = len(feats)
    idx = rng.choice(n, size=min(n_sample, n), replace=False)
    sample = feats[idx]
    rot = AgglomerativeClustering(n_clusters=k, linkage=linkage).fit_predict(sample)
    centroids = np.stack([sample[rot == c].mean(0) for c in range(k)])
    return assign_to_centroid(feats, centroids)


def sort_by_size(labels, k):
    order = np.argsort(-np.bincount(labels, minlength=k))
    label_map = np.empty(k, dtype=np.int32)
    label_map[order] = np.arange(k)
    return label_map[labels]


def outlines(label_map, thickness):
    b = np.zeros(label_map.shape, bool)
    b[:, :-1] |= label_map[:, :-1] != label_map[:, 1:]
    b[:-1, :] |= label_map[:-1, :] != label_map[1:, :]
    if thickness > 1:
        b = binary_dilation(b, iterations=thickness - 1)
    return b


def to_png(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------- interface
st.title("🎨 Pattern detection in photos with clustering")
st.caption("Upload a photo, pick the algorithm and the number of groups, and see the clusters drawn over your image.")

with st.sidebar:
    st.header("⚙️ Settings")
    uploaded = st.file_uploader("Photo", type=["jpg", "jpeg", "png", "webp", "bmp"])

    method = st.radio("Method", ["K-Means", "Hierarchical"])
    k = st.slider("Number of groups", min_value=2, max_value=15, value=4)

    linkage = "ward"
    n_sample = 3000
    if method == "Hierarchical":
        linkage = st.selectbox("Linkage", ["ward", "complete", "average", "single"])
        n_sample = st.slider(
            "Pixels used to fit the model", 500, 5000, 3000, step=500,
            help="Hierarchical clustering needs quadratic memory. "
                 "The model is fitted on a random sample and the remaining pixels are assigned to the nearest group.",
        )

    st.subheader("Features")
    color_space = st.radio("Color space", ["RGB", "Lab"], horizontal=True,
                      help="Lab is closer to human color perception.")
    use_pos = st.checkbox("Include position (x, y)", value=False,
                           help="Favors spatially contiguous groups.")
    pos_weight = st.slider("Position weight", 0.0, 2.0, 0.5, 0.1) if use_pos else 0.0

    st.subheader("Visualization")
    mode = st.radio("How to plot on the photo", ["Group outlines", "Colored points", "Color mask"])
    alpha = st.slider("Fill opacity", 0.0, 1.0,
                      {"Group outlines": 0.2, "Colored points": 0.8}.get(mode, 0.45), 0.05)
    min_region, n_points, point_size, show_outline = 0.5, 1500, 40, True
    style, smoothing, thickness = "Exact outline", 2.0, 3
    if mode == "Group outlines":
        style = st.radio("Style", ["Exact outline", "Convex hull"],
                          help="Convex hull = a 'rubber band' around each region.")
        thickness = st.slider("Line thickness", 1, 8, 3)
        smoothing = st.slider("Outline smoothing", 0.0, 10.0, 2.0, 0.5)
        min_region = st.slider("Minimum region size (% of photo)", 0.1, 10.0, 0.5, 0.1,
                               help="Smaller regions are ignored.")
    elif mode == "Colored points":
        n_points = st.slider("Number of points", 200, 5000, 1500, step=100)
        point_size = st.slider("Point size", 5, 200, 40, step=5)
    else:
        show_outline = st.checkbox("Draw outlines", value=True)
    res = st.slider("Processing resolution (longest side, px)", 80, 400, 200, step=20,
                    help="Lower = faster. The result is rescaled to the original photo.")

if uploaded is None:
    st.info("👈 Upload a photo in the sidebar to get started.")
    st.stop()

img = load_image(uploaded)
W, H = img.size
scale = res / max(W, H)
small = img.resize((max(1, int(W * scale)), max(1, int(H * scale))), Image.LANCZOS) if scale < 1 else img
small_arr = np.asarray(small)
h, w, _ = small_arr.shape

with st.spinner("Clustering..."):
    feats = extract_features(small_arr, color_space, use_pos, pos_weight)
    labels = cluster_pixels(feats, k, method, linkage, n_sample)
    labels = sort_by_size(labels, k)

    # silhouette on a sample, to keep it fast
    try:
        sil = silhouette_score(feats, labels, sample_size=min(2000, len(feats)), random_state=0)
    except ValueError:
        sil = float("nan")

# rescale the (2D!) label map to the original size
label_map = np.asarray(Image.fromarray(labels.reshape(h, w).astype(np.uint8)).resize((W, H), Image.NEAREST))
original = np.asarray(img)
colors = PALETTE[np.arange(k) % len(PALETTE)]
colors_f = colors / 255.0

means = np.stack([original[label_map == c].mean(0) if (label_map == c).any() else colors[c] for c in range(k)]).astype(np.uint8)
segmented = means[label_map]
proportions = np.bincount(label_map.ravel(), minlength=k) / label_map.size


def draw(mode):
    fig, ax = plt.subplots(figsize=(9, 9 * H / W))
    ax.axis("off")
    if mode == "Color mask":
        ov = ((1 - alpha) * original + alpha * colors[label_map]).astype(np.uint8)
        if show_outline:
            ov[outlines(label_map, max(1, round(max(W, H) / 600)))] = 255
        ax.imshow(ov)
    else:
        ax.imshow(original)
        if mode == "Group outlines":
            min_area = min_region / 100 * W * H
            sig = smoothing * max(W, H) / 400
            for c in range(k):
                comp, n = ndlabel(label_map == c)
                if n == 0:
                    continue
                areas = np.bincount(comp.ravel())[1:]
                if style == "Exact outline":
                    mask = np.isin(comp, np.nonzero(areas >= min_area)[0] + 1).astype(float)
                    if not mask.any():
                        continue
                    m = np.pad(gaussian_filter(mask, sig) if sig > 0 else mask, 1)
                    if alpha > 0:
                        ax.contourf(np.arange(-1, W + 1), np.arange(-1, H + 1), m,
                                    levels=[0.5, 1.5], colors=[colors_f[c]], alpha=alpha)
                    for ct in find_contours(m, 0.5):
                        ax.plot(ct[:, 1] - 1, ct[:, 0] - 1, color="white", lw=thickness + 2)
                        ax.plot(ct[:, 1] - 1, ct[:, 0] - 1, color=colors_f[c], lw=thickness)
                else:
                    for i in np.nonzero(areas >= min_area)[0] + 1:
                        cts = find_contours(np.pad((comp == i).astype(float), 1), 0.5)
                        pts = np.vstack(cts)
                        if len(pts) < 3:
                            continue
                        poly = pts[ConvexHull(pts).vertices][:, ::-1] - 1
                        ax.add_patch(Polygon(poly, closed=True, facecolor=(*colors_f[c], alpha),
                                             edgecolor="white", linewidth=thickness + 2))
                        ax.add_patch(Polygon(poly, closed=True, fill=False,
                                             edgecolor=colors_f[c], linewidth=thickness))
            ax.set_xlim(-0.5, W - 0.5)
            ax.set_ylim(H - 0.5, -0.5)
        else:  # Colored points
            rng = np.random.default_rng(1)
            idx = rng.choice(W * H, size=min(n_points, W * H), replace=False)
            ys, xs = np.divmod(idx, W)
            ax.scatter(xs, ys, c=colors_f[label_map[ys, xs]], s=point_size, alpha=alpha,
                       edgecolors="white", linewidths=0.4)
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=colors_f[c], markersize=11,
                      label=f"Group {c + 1} ({proportions[c] * 100:.1f}%)") for c in range(k)]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    fig.tight_layout()
    return fig


c1, c2 = st.columns(2)
c1.subheader("Original")
c1.image(original, width="stretch")
c2.subheader(f"Groups over the photo ({method}, k={k})")
fig = draw(mode)
c2.pyplot(fig, width="stretch")
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
plt.close(fig)
c2.download_button("⬇️ Download plot (PNG)", buf.getvalue(), "clusters_plot.png", "image/png")

tab1, tab2 = st.tabs(["Groups and proportions", "Imagem segmented"])

with tab1:
    m1, m2, m3 = st.columns(3)
    m1.metric("Method", method if method == "K-Means" else f"Hierarchical ({linkage})")
    m2.metric("Groups", k)
    m3.metric("Silhouette", "—" if np.isnan(sil) else f"{sil:.3f}",
              help="Ranges from -1 to 1. Higher means better-separated groups.")

    cols = st.columns(min(k, 5))
    for c in range(k):
        col = cols[c % len(cols)]
        cor_hex = "#%02x%02x%02x" % tuple(colors[c])
        media_hex = "#%02x%02x%02x" % tuple(means[c])
        col.markdown(
            f"""<div style="border:1px solid #8884;border-radius:8px;padding:8px;margin-bottom:8px">
            <b>Group {c + 1}</b> — {proportions[c] * 100:.1f}%<br>
            <span style="display:inline-block;width:22px;height:22px;background:{cor_hex};
            border-radius:4px;vertical-align:middle"></span> marker<br>
            <span style="display:inline-block;width:22px;height:22px;background:{media_hex};
            border-radius:4px;vertical-align:middle"></span> mean color {media_hex}</div>""",
            unsafe_allow_html=True,
        )

with tab2:
    st.image(segmented, caption="Cada pixel recebe a mean color do seu grupo", width="stretch")
    st.download_button("⬇️ Download segmentation (PNG)", to_png(segmented), "clusters_segmented.png", "image/png")