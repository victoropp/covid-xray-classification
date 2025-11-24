# Grad-CAM Visualization Examples

This directory contains Grad-CAM (Gradient-weighted Class Activation Mapping) visualizations showing which regions of chest X-rays the model focuses on when making predictions.

## What is Grad-CAM?

Grad-CAM provides visual explanations for deep learning model decisions by highlighting the important regions in the input image that influence the classification.

## How to Generate

Run the Grad-CAM visualization script:

```bash
python src/gradcam.py --image_path <path_to_xray> --model_path models/resnet50_best.pth
```

## Example Output

Each visualization shows:
- **Original X-ray**: Input chest X-ray image
- **Heatmap**: Grad-CAM attention map (red = high importance, blue = low importance)
- **Overlay**: Heatmap superimposed on original image
- **Prediction**: Model's classification with confidence score

## Use in Streamlit App

The interactive Grad-CAM visualizations are available in the Streamlit dashboard:
- Navigate to **Clinical Predictions** page
- Upload a chest X-ray image
- View real-time Grad-CAM explanations

---

*Grad-CAM helps build trust in AI predictions by showing which lung regions contribute to the diagnosis.*
