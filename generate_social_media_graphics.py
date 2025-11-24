"""
Generate professional social media graphics for COVID-19 X-ray Classification
Creates LinkedIn and Kaggle cover images with medical theme (NO METRICS)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
import numpy as np

# Set style
plt.style.use('dark_background')

def create_linkedin_post():
    """Generate LinkedIn post image (1200x627px) - Medical theme, no metrics"""
    fig, ax = plt.subplots(figsize=(12, 6.27), dpi=100)

    # Medical gradient background (blue to cyan)
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    gradient = np.vstack((gradient, gradient))
    ax.imshow(gradient, aspect='auto', cmap='Blues',
              extent=[0, 12, 0, 6.27], alpha=0.6)

    # Dark overlay for depth
    overlay = Rectangle((0, 0), 12, 6.27,
                        facecolor='#0a1929',
                        alpha=0.7)
    ax.add_patch(overlay)

    # Main content box
    main_box = FancyBboxPatch((0.5, 1.5), 11, 4,
                             boxstyle="round,pad=0.15",
                             edgecolor='#00b4d8',
                             facecolor='#1a2332',
                             alpha=0.9,
                             linewidth=3)
    ax.add_patch(main_box)

    # Title
    ax.text(6, 5.2, 'COVID-19 X-ray Classification',
            fontsize=48, weight='bold', ha='center',
            color='white', family='sans-serif')

    ax.text(6, 4.5, 'AI-Powered Medical Imaging Analysis',
            fontsize=22, ha='center',
            color='#00b4d8', style='italic')

    # Four classification boxes (visual only, no percentages)
    boxes_y = 3.5
    box_width = 2.3
    box_spacing = 0.2
    start_x = 1.2

    classes = [
        ('COVID-19', '#ff4444'),
        ('Viral Pneumonia', '#ff8800'),
        ('Lung Opacity', '#ffbb00'),
        ('Normal', '#00dd88')
    ]

    for i, (class_name, color) in enumerate(classes):
        x_pos = start_x + i * (box_width + box_spacing)

        # Class box
        class_box = FancyBboxPatch((x_pos, boxes_y - 0.4), box_width, 0.8,
                                  boxstyle="round,pad=0.08",
                                  facecolor=color,
                                  edgecolor='white',
                                  linewidth=2,
                                  alpha=0.8)
        ax.add_patch(class_box)

        # Class name
        ax.text(x_pos + box_width/2, boxes_y, class_name,
                fontsize=13, ha='center', va='center',
                color='white', weight='bold')

    # Technology stack
    tech_text = 'PyTorch  •  ResNet50  •  Grad-CAM  •  Streamlit'
    ax.text(6, 2.3, tech_text,
            fontsize=18, ha='center',
            color='#90e0ef', weight='bold')

    # Key features
    features = [
        'Deep Learning',
        'Transfer Learning',
        'Explainable AI',
        'Clinical Decision Support'
    ]

    features_y = 1.8
    for i, feature in enumerate(features):
        x_pos = 1.8 + i * 2.5
        ax.text(x_pos, features_y, f'• {feature}',
                fontsize=13, ha='left',
                color='white', alpha=0.9)

    # Bottom banner
    bottom_bar = Rectangle((0, 0), 12, 1.2,
                          facecolor='#012a4a',
                          alpha=0.95)
    ax.add_patch(bottom_bar)

    ax.text(0.3, 0.7, 'Victor Collins Oppon',
            fontsize=16, ha='left',
            color='white', weight='bold')

    ax.text(0.3, 0.3, 'Data Scientist | Healthcare AI Specialist',
            fontsize=12, ha='left',
            color='#00b4d8', alpha=0.9)

    ax.text(11.7, 0.5, 'github.com/victoropp',
            fontsize=13, ha='right',
            color='#90e0ef')

    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.27)
    ax.axis('off')

    plt.tight_layout(pad=0)
    plt.savefig('social_media/linkedin_post.png',
                dpi=100, bbox_inches='tight',
                facecolor='#0a1929', edgecolor='none')
    print("[OK] LinkedIn post image created: social_media/linkedin_post.png")
    plt.close()


def create_kaggle_thumbnail():
    """Generate Kaggle notebook thumbnail (640x512px) - Medical theme, no metrics"""
    fig, ax = plt.subplots(figsize=(6.4, 5.12), dpi=100)

    # Medical gradient
    gradient = np.linspace(0, 1, 256).reshape(-1, 1)
    ax.imshow(gradient, aspect='auto', cmap='winter',
              extent=[0, 6.4, 0, 5.12], alpha=0.7)

    # Dark overlay
    overlay = Rectangle((0, 0), 6.4, 5.12,
                        facecolor='#001219',
                        alpha=0.6)
    ax.add_patch(overlay)

    # Main box
    main_box = FancyBboxPatch((0.4, 0.8), 5.6, 3.8,
                             boxstyle="round,pad=0.15",
                             edgecolor='#0077b6',
                             facecolor='#013a63',
                             alpha=0.95,
                             linewidth=3)
    ax.add_patch(main_box)

    # Title
    ax.text(3.2, 4.3, 'COVID-19',
            fontsize=42, weight='bold', ha='center',
            color='white')

    ax.text(3.2, 3.75, 'X-ray Classification',
            fontsize=32, weight='bold', ha='center',
            color='#00b4d8')

    ax.text(3.2, 3.25, 'Deep Learning for Medical Imaging',
            fontsize=15, ha='center',
            color='white', alpha=0.85, style='italic')

    # Classification badges
    badges_y = 2.5
    badge_data = [
        ('COVID-19', '#ee4266'),
        ('Pneumonia', '#ff6b35'),
        ('Lung Opacity', '#ffd23f'),
        ('Normal', '#06ffa5')
    ]

    for i, (label, color) in enumerate(badge_data):
        row = i // 2
        col = i % 2
        x = 1.5 + col * 3
        y = badges_y - row * 0.6

        badge = FancyBboxPatch((x - 0.7, y - 0.2), 1.4, 0.4,
                              boxstyle="round,pad=0.05",
                              facecolor=color,
                              edgecolor='white',
                              linewidth=1.5,
                              alpha=0.9)
        ax.add_patch(badge)

        ax.text(x, y, label, fontsize=11, ha='center', va='center',
                color='white', weight='bold')

    # Tech stack
    ax.text(3.2, 1.3, 'PyTorch  •  ResNet50  •  Grad-CAM',
            fontsize=13, ha='center',
            color='#90e0ef', weight='bold')

    # Bottom section
    bottom_bar = Rectangle((0, 0), 6.4, 0.9,
                          facecolor='#001d3d',
                          alpha=0.95)
    ax.add_patch(bottom_bar)

    ax.text(3.2, 0.55, 'Explainable AI for Clinical Decision Support',
            fontsize=13, ha='center',
            color='white', weight='bold')

    ax.text(3.2, 0.15, 'by Victor Collins Oppon',
            fontsize=11, ha='center',
            color='#0077b6')

    ax.set_xlim(0, 6.4)
    ax.set_ylim(0, 5.12)
    ax.axis('off')

    plt.tight_layout(pad=0)
    plt.savefig('social_media/kaggle_thumbnail.png',
                dpi=100, bbox_inches='tight',
                facecolor='#001219', edgecolor='none')
    print("[OK] Kaggle thumbnail created: social_media/kaggle_thumbnail.png")
    plt.close()


if __name__ == "__main__":
    print("\nGenerating Professional Social Media Graphics (No Metrics)...\n")

    create_linkedin_post()
    create_kaggle_thumbnail()

    print("\nAll graphics generated successfully!")
    print("\nFiles created in 'social_media/' directory:")
    print("   - linkedin_post.png (1200x627px)")
    print("   - kaggle_thumbnail.png (640x512px)")
    print("\nUsage:")
    print("   - LinkedIn: Upload as post image or article cover")
    print("   - Kaggle: Use as notebook thumbnail")
    print("\nReady to share your project!\n")
