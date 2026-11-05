# Image Clustering

Streamlit app that applies cluster analysis to the pixels of a photo and draws the resulting groups over the image.

![Image Clustering app](docs/screenshot.png)

## Idea

Cluster analysis is a multivariate statistical technique that groups observations by similarity, without labels (Hair et al., 1998). In an image, each pixel is an observation described by its color (RGB or Lab) and, optionally, its position (x, y). Clustering pixels reveals dominant colors and regions, which is useful in computer vision for segmentation and pattern detection.

## Features

- K-Means or hierarchical clustering (ward, complete, average, single)
- 2 to 15 groups
- RGB or Lab color space, optional pixel position
- Groups drawn as outlines, points or a color mask

## Run

```bash
git clone https://github.com/mathblanco/mathblanco.git
cd mathblanco/image-clustering
pip install -r requirements.txt
streamlit run app.py
```

## Reference

Hair, J. F., Tatham, R. L., Anderson, R. E. and Black, W. (1998). *Multivariate Data Analysis* (5th ed.). Prentice Hall.
