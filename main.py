import cv2
import numpy as np
import math
import os

# =========================================================
# 1. PROJECT SETTINGS
# =========================================================

IMAGE_A = "image_A.png"
IMAGE_B = "image_B.png"

OUTPUT_FOLDER = "output"

# Sun information for Image A and Image B
# Change these values according to your actual metadata.
SUN_AZIMUTH_A = 70
SUN_AZIMUTH_B = 120

SUN_ELEVATION_A = 35
SUN_ELEVATION_B = 50

# Approximate shadow movement
SHADOW_SHIFT = 25


# =========================================================
# 2. CREATE OUTPUT FOLDER
# =========================================================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# =========================================================
# 3. READ TWO IMAGES
# =========================================================

print("Step 1: Reading images...")

image_A = cv2.imread(IMAGE_A, cv2.IMREAD_GRAYSCALE)
image_B = cv2.imread(IMAGE_B, cv2.IMREAD_GRAYSCALE)

if image_A is None:
    print("ERROR: image_A.png not found!")
    exit()

if image_B is None:
    print("ERROR: image_B.png not found!")
    exit()

# Make both images the same size
height, width = image_A.shape

image_B = cv2.resize(
    image_B,
    (width, height)
)

print("Images loaded successfully.")


# =========================================================
# 4. IMAGE NORMALIZATION
# =========================================================

print("Step 2: Normalizing images...")

image_A = cv2.normalize(
    image_A,
    None,
    0,
    255,
    cv2.NORM_MINMAX
)

image_B = cv2.normalize(
    image_B,
    None,
    0,
    255,
    cv2.NORM_MINMAX
)


# =========================================================
# 5. SHADOW DETECTION
# =========================================================

def detect_shadow(image):

    # Convert image into floating point
    img = image.astype(np.float32) / 255.0

    # Dark pixels have higher shadow probability
    darkness = 1.0 - img

    # Calculate local average brightness
    local_mean = cv2.GaussianBlur(
        img,
        (0, 0),
        9
    )

    # Difference between local brightness and pixel
    local_darkness = local_mean - img

    local_darkness = np.clip(
        local_darkness,
        0,
        1
    )

    # Calculate image gradient
    gx = cv2.Sobel(
        img,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    gy = cv2.Sobel(
        img,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    gradient = cv2.magnitude(
        gx,
        gy
    )

    gradient = cv2.normalize(
        gradient,
        None,
        0,
        1,
        cv2.NORM_MINMAX
    )

    # Shadow probability
    shadow_probability = (
        0.60 * darkness
        +
        0.30 * local_darkness
        +
        0.10 * (1 - gradient)
    )

    return shadow_probability


shadow_probability_A = detect_shadow(image_A)
shadow_probability_B = detect_shadow(image_B)


# =========================================================
# 6. CREATE SHADOW MASK
# =========================================================

print("Step 3: Detecting shadows...")

def create_shadow_mask(probability):

    threshold = np.percentile(
        probability,
        35
    )

    threshold = max(
        0.45,
        min(0.80, threshold)
    )

    mask = (
        probability >= threshold
    ).astype(np.uint8) * 255

    # Remove small noise
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    return mask


shadow_A = create_shadow_mask(
    shadow_probability_A
)

shadow_B = create_shadow_mask(
    shadow_probability_B
)


cv2.imwrite(
    OUTPUT_FOLDER + "/shadow_A.png",
    shadow_A
)

cv2.imwrite(
    OUTPUT_FOLDER + "/shadow_B.png",
    shadow_B
)


# =========================================================
# 7. SUN GEOMETRY
# =========================================================

print("Step 4: Calculating Sun geometry...")

def sun_vector(angle):

    angle = math.radians(angle)

    x = math.cos(angle)
    y = math.sin(angle)

    return np.array(
        [x, y],
        dtype=np.float32
    )


sun_A = sun_vector(
    SUN_AZIMUTH_A
)

sun_B = sun_vector(
    SUN_AZIMUTH_B
)


# Shadow moves approximately opposite to Sun direction

shadow_direction_A = -sun_A
shadow_direction_B = -sun_B


# =========================================================
# 8. SHADOW MIGRATION PREDICTION
# =========================================================

print("Step 5: Predicting shadow migration...")

direction_change = (
    shadow_direction_B
    -
    shadow_direction_A
)


# Sun elevation affects approximate shadow length

elevation_A = max(
    0.1,
    math.sin(
        math.radians(
            SUN_ELEVATION_A
        )
    )
)

elevation_B = max(
    0.1,
    math.sin(
        math.radians(
            SUN_ELEVATION_B
        )
    )
)

elevation_factor = (
    1 / elevation_A
    +
    1 / elevation_B
) / 2


shift = (
    direction_change
    *
    SHADOW_SHIFT
    *
    elevation_factor
)


print(
    "Predicted shadow movement:",
    shift
)


# =========================================================
# 9. MOVE SHADOW A
# =========================================================

def move_shadow(
    mask,
    shift_x,
    shift_y
):

    matrix = np.float32([
        [1, 0, shift_x],
        [0, 1, shift_y]
    ])

    moved = cv2.warpAffine(
        mask,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT
    )

    return moved


predicted_shadow_A = move_shadow(
    shadow_A,
    shift[0],
    shift[1]
)


cv2.imwrite(
    OUTPUT_FOLDER + "/predicted_shadow.png",
    predicted_shadow_A
)


# =========================================================
# 10. SHADOW MIGRATION MAP
# =========================================================

print("Step 6: Creating Shadow Migration Map...")

A = predicted_shadow_A.astype(
    np.float32
) / 255.0

B = shadow_B.astype(
    np.float32
) / 255.0


migration_map = np.abs(
    A - B
)

migration_map = cv2.GaussianBlur(
    migration_map,
    (0, 0),
    2
)


migration_image = (
    migration_map * 255
).astype(np.uint8)


cv2.imwrite(
    OUTPUT_FOLDER + "/shadow_migration_map.png",
    migration_image
)


# =========================================================
# 11. STABILITY MAP
# =========================================================

print("Step 7: Finding stable and unstable features...")

# Less shadow = more stable
stability_map = (
    1 -
    shadow_probability_A
)

# Shadow migration reduces reliability
stability_map *= (
    1 -
    0.75 * migration_map
)

# Keep values between 0.15 and 1
stability_map = np.clip(
    stability_map,
    0.15,
    1.0
)


stability_image = (
    stability_map * 255
).astype(np.uint8)


cv2.imwrite(
    OUTPUT_FOLDER + "/stability_map.png",
    stability_image
)


# =========================================================
# 12. FEATURE DETECTION
# =========================================================

print("Step 8: Detecting features...")

sift = cv2.SIFT_create(
    nfeatures=5000
)


keypoints_A, descriptors_A = (
    sift.detectAndCompute(
        image_A,
        None
    )
)

keypoints_B, descriptors_B = (
    sift.detectAndCompute(
        image_B,
        None
    )
)


print(
    "Features in Image A:",
    len(keypoints_A)
)

print(
    "Features in Image B:",
    len(keypoints_B)
)


# =========================================================
# 13. CALCULATE FEATURE STABILITY
# =========================================================

def get_feature_weight(
    keypoints,
    stability
):

    weights = []

    h, w = stability.shape

    for kp in keypoints:

        x = int(
            round(kp.pt[0])
        )

        y = int(
            round(kp.pt[1])
        )

        x = max(
            0,
            min(w - 1, x)
        )

        y = max(
            0,
            min(h - 1, y)
        )

        weight = stability[
            y,
            x
        ]

        weights.append(
            float(weight)
        )

    return np.array(
        weights
    )


weights_A = get_feature_weight(
    keypoints_A,
    stability_map
)

weights_B = get_feature_weight(
    keypoints_B,
    stability_map
)


# =========================================================
# 14. FEATURE MATCHING
# =========================================================

print("Step 9: Matching features...")

bf = cv2.BFMatcher(
    cv2.NORM_L2
)

matches = bf.knnMatch(
    descriptors_A,
    descriptors_B,
    k=2
)


# Lowe ratio test

good_matches = []

for pair in matches:

    if len(pair) < 2:
        continue

    m = pair[0]
    n = pair[1]

    if m.distance < 0.75 * n.distance:

        stability = (
            weights_A[m.queryIdx]
            +
            weights_B[m.trainIdx]
        ) / 2

        # Give stable features higher confidence
        confidence = (
            stability /
            (m.distance + 1e-6)
        )

        good_matches.append(
            (
                confidence,
                m
            )
        )


# Sort according to confidence

good_matches.sort(
    key=lambda x: x[0],
    reverse=True
)


weighted_matches = [
    item[1]
    for item in good_matches
]


print(
    "Good matches:",
    len(weighted_matches)
)


# =========================================================
# 15. REMOVE WRONG MATCHES USING RANSAC
# =========================================================

print("Step 10: Removing wrong matches...")

final_matches = []

if len(weighted_matches) >= 8:

    points_A = np.float32([
        keypoints_A[
            m.queryIdx
        ].pt
        for m in weighted_matches
    ])

    points_B = np.float32([
        keypoints_B[
            m.trainIdx
        ].pt
        for m in weighted_matches
    ])

    points_A = points_A.reshape(
        -1,
        1,
        2
    )

    points_B = points_B.reshape(
        -1,
        1,
        2
    )

    H, mask = cv2.findHomography(
        points_A,
        points_B,
        cv2.RANSAC,
        4.0
    )

    if mask is not None:

        mask = mask.ravel()

        for i, value in enumerate(mask):

            if value == 1:

                final_matches.append(
                    weighted_matches[i]
                )


print(
    "Final accurate matches:",
    len(final_matches)
)


# =========================================================
# 16. DRAW FINAL CORRESPONDENCES
# =========================================================

print("Step 11: Creating final correspondence image...")

result = cv2.drawMatches(
    image_A,
    keypoints_A,
    image_B,
    keypoints_B,
    final_matches,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)


cv2.imwrite(
    OUTPUT_FOLDER +
    "/final_correspondences.png",
    result
)


# =========================================================
# 17. FINISHED
# =========================================================

print()
print("====================================")
print("SHADOW MIGRATION PIPELINE COMPLETE")
print("====================================")

print()
print("Check the 'output' folder.")

print()
print("Generated files:")

print("1. shadow_A.png")
print("2. shadow_B.png")
print("3. predicted_shadow.png")
print("4. shadow_migration_map.png")
print("5. stability_map.png")
print("6. final_correspondences.png")