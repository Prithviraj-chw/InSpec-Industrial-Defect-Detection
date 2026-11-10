# InSpec: Industrial Defect Detection with CNN + Grad-CAM

Automatic classification of steel surface defects using transfer learning, with Grad-CAM heatmaps to show which image regions drove each prediction.

> **Status:**  In progress. Sections marked `TODO` will be filled in as each phase is completed.

## Problem

Manual inspection of manufactured products is slow, expensive and error-prone. This project builds a computer vision system that, given an image of a steel surface, reports:

- the **type of defect** present
- the model's **confidence** in that prediction
- a **Grad-CAM heatmap** showing roughly which regions the model focused on

> **Note on scope:** Grad-CAM shows where the model *looked*, not an exact defect boundary. This is an explainability tool, not an object detector.

## Dataset

[NEU Surface Defect Database](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database) (NEU-DET version).

- 1,800 grayscale images, 200x200 px
- 6 defect classes, 300 images each (balanced)

| Class | Description |
|---|---|
| Crazing | TODO |
| Inclusion | TODO |
| Patches | TODO |
| Pitted surface | TODO |
| Rolled-in scale | TODO |
| Scratches | TODO |

Train / validation / test split: `TODO` (stratified).

## Approach

1. **Baseline:** small CNN trained from scratch
2. **Transfer learning:** ImageNet-pretrained backbones with a new 6-class head, frozen first, then partially fine-tuned
3. **Data augmentation:** flips, rotations, brightness changes
4. **Model comparison:** ResNet50 vs EfficientNetB0 vs ConvNeXtTiny under identical training conditions
5. **Error analysis:** confusion matrix, per-class metrics, inspection of misclassified images
6. **Explainability:** Grad-CAM implemented with `tf.GradientTape`
7. **Deployment:** Streamlit web app

## Results

> TODO: fill in with real numbers after training. Do not estimate.

| Model | Test accuracy | Macro F1 | Parameters | Inference time (ms/img) |
|---|---|---|---|---|
| Baseline CNN | TODO | TODO | TODO | TODO |
| ResNet50 | TODO | TODO | TODO | TODO |
| EfficientNetB0 | TODO | TODO | TODO | TODO |
| ConvNeXtTiny | TODO | TODO | TODO | TODO |

### Key findings

- TODO: which model gave the best accuracy / speed trade-off
- TODO: which classes were most confused, and why
- TODO: does Grad-CAM focus on the actual defect or on background artifacts

### Screenshots

TODO: confusion matrix, Grad-CAM examples, web app

## Web app

Streamlit app with:

- image upload
- predicted class and confidence
- original image next to the Grad-CAM overlay
- inspection history (SQLite)
- simple analytics page (defect counts per class)

**Live demo:** TODO (Hugging Face Spaces link)

## Tech stack

Python, TensorFlow / Keras, NumPy, Pandas, scikit-learn, Matplotlib, Streamlit, SQLite, Google Colab

## Project structure

```
InSpec-Industrial-Defect-Detection/
├── notebooks/      # exploration, training, comparison
├── src/            # data loading, models, Grad-CAM
├── app/            # Streamlit app
├── models/         # saved weights
└── README.md
```

## How to run

```bash
git clone https://github.com/Prithviraj-chw/InSpec-Industrial-Defect-Detection.git
cd InSpec-Industrial-Defect-Detection
pip install -r requirements.txt
streamlit run app/app.py
```

> TODO: confirm commands once the app and `requirements.txt` exist.

## Limitations

- The dataset is small and collected under controlled conditions; performance on real factory images may differ.
- Grad-CAM gives coarse localization, not precise defect boundaries.
- TODO: add limitations found during error analysis.

## Acknowledgements

- NEU Surface Defect Database, Northeastern University (K. Song and Y. Yan)
- Grad-CAM: Selvaraju et al., 2017
