import os
import sys
import json
import time
import warnings
warnings.filterwarnings('ignore')

DATASET_ROOT = r"C:\Users\Anuj0\Downloads\VideoBadminton_Dataset\VideoBadminton_Dataset"
OUTPUT_DIR =  r"C:\Users\Anuj0\Downloads\Badminton_Output"
RANDOM_SEED = 42
TEST_SIZE = 0.2
TARGET_FRAMES = 64
MEDIAPIPE_DETECTION_CONF = 0.5
MEDIAPIPE_TRACKING_CONF = 0.5
MEDIAPIPE_MODEL_COMPLEXITY = 2

RESUME_FROM_STEP = 0

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "skeletons"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "features"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "models"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "plots"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "reports"), exist_ok=True)

def save_checkpoint(step_number, step_name):
    log_path = os.path.join(OUTPUT_DIR, "checkpoint_log.json")
    log = {}
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            log = json.load(f)
    log[f"step_{step_number}"] = {
        "name": step_name,
        "completed": True,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=2)
    print(f"\n STEP {step_number} COMPLETE — Checkpoint saved. Safe to close.\n")


def step1_dataset_inventory():
    print("=" * 60)
    print("STEP 1: Dataset Inventory")
    print("=" * 60)

    class_info = {}
    total_clips = 0

    for class_name in sorted(os.listdir(DATASET_ROOT)):
        class_path = os.path.join(DATASET_ROOT, class_name)
        if not os.path.isdir(class_path):
            continue

        clips = [f for f in os.listdir(class_path)
                 if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        class_info[class_name] = {
            "count": len(clips),
            "files": clips
        }
        total_clips += len(clips)
        print(f"  {class_name}: {len(clips)} clips")

    print(f"\n  TOTAL: {total_clips} clips across {len(class_info)} classes")

    save_path = os.path.join(OUTPUT_DIR, "class_info.json")
    with open(save_path, 'w') as f:
        json.dump(class_info, f, indent=2)
    print(f"  Saved to: {save_path}")

    save_checkpoint(1, "Dataset Inventory")
    return class_info


def step2_extract_skeletons():
    print("=" * 60)
    print("STEP 2: Skeleton Extraction (MediaPipe)")
    print("  This is the longest step — expect several hours.")
    print("  Progress is saved per-clip. Safe to interrupt and resume.")
    print("=" * 60)

    import mediapipe as mp
    import cv2
    import numpy as np
    from tqdm import tqdm

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=MEDIAPIPE_MODEL_COMPLEXITY,
        min_detection_confidence=MEDIAPIPE_DETECTION_CONF,
        min_tracking_confidence=MEDIAPIPE_TRACKING_CONF
    )

    with open(os.path.join(OUTPUT_DIR, "class_info.json"), 'r') as f:
        class_info = json.load(f)

    skeleton_dir = os.path.join(OUTPUT_DIR, "skeletons")
    report = {"success": 0, "failed": 0, "skipped_existing": 0, "errors": []}
    start_time = time.time()

    for class_name, info in class_info.items():
        class_skeleton_dir = os.path.join(skeleton_dir, class_name)
        os.makedirs(class_skeleton_dir, exist_ok=True)

        for clip_file in tqdm(info["files"], desc=class_name):
            npy_name = os.path.splitext(clip_file)[0] + ".npy"
            npy_path = os.path.join(class_skeleton_dir, npy_name)
            if os.path.exists(npy_path):
                report["skipped_existing"] += 1
                continue

            video_path = os.path.join(DATASET_ROOT, class_name, clip_file)
            try:
                cap = cv2.VideoCapture(video_path)
                all_frames = []

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = pose.process(rgb)

                    if result.pose_landmarks:
                        frame_data = []
                        for lm in result.pose_landmarks.landmark:
                            frame_data.extend([lm.x, lm.y, lm.z, lm.visibility])
                        all_frames.append(frame_data)
                    else:
                        all_frames.append([0.0] * 132)

                cap.release()
                skeleton = np.array(all_frames, dtype=np.float32)

                np.save(npy_path, skeleton)
                report["success"] += 1

            except Exception as e:
                report["failed"] += 1
                report["errors"].append({"file": video_path, "error": str(e)})
                print(f"\n  ERROR: {clip_file} — {e}")

    pose.close()
    elapsed = time.time() - start_time
    report["elapsed_seconds"] = round(elapsed, 1)
    report["elapsed_human"] = f"{elapsed/3600:.1f} hours"

    report_path = os.path.join(OUTPUT_DIR, "extraction_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n  Extraction report saved to: {report_path}")
    print(f"  Success: {report['success']}, Failed: {report['failed']}, "
          f"Skipped: {report['skipped_existing']}")
    print(f"  Time: {report['elapsed_human']}")

    save_checkpoint(2, "Skeleton Extraction")
    return report


def step3_quality_check():
    print("=" * 60)
    print("STEP 3: Quality Check")
    print("=" * 60)

    import numpy as np

    skeleton_dir = os.path.join(OUTPUT_DIR, "skeletons")
    quality = {"per_class": {}, "flagged_clips": [], "total": 0, "failed": 0}

    for class_name in sorted(os.listdir(skeleton_dir)):
        class_path = os.path.join(skeleton_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        class_total = 0
        class_failed = 0
        detection_rates = []

        for npy_file in os.listdir(class_path):
            if not npy_file.endswith('.npy'):
                continue
            data = np.load(os.path.join(class_path, npy_file))
            class_total += 1
            quality["total"] += 1

            zero_frames = np.sum(np.all(data == 0, axis=1))
            detection_rate = 1.0 - (zero_frames / max(len(data), 1))
            detection_rates.append(detection_rate)

            if detection_rate < 0.5:
                class_failed += 1
                quality["failed"] += 1
                quality["flagged_clips"].append({
                    "class": class_name,
                    "file": npy_file,
                    "detection_rate": round(detection_rate, 3),
                    "total_frames": len(data)
                })

        quality["per_class"][class_name] = {
            "total_clips": class_total,
            "failed_clips": class_failed,
            "mean_detection_rate": round(float(np.mean(detection_rates)), 4) if detection_rates else 0,
            "min_detection_rate": round(float(np.min(detection_rates)), 4) if detection_rates else 0
        }

        print(f"  {class_name}: {class_total} clips, "
              f"mean detection {quality['per_class'][class_name]['mean_detection_rate']:.1%}, "
              f"{class_failed} flagged")

    print(f"\n  OVERALL: {quality['failed']}/{quality['total']} clips flagged (<50% detection)")

    report_path = os.path.join(OUTPUT_DIR, "quality_report.json")
    with open(report_path, 'w') as f:
        json.dump(quality, f, indent=2)
    print(f"  Saved to: {report_path}")

    save_checkpoint(3, "Quality Check")
    return quality


def step4_feature_engineering():
    print("=" * 60)
    print("STEP 4: Feature Engineering")
    print("=" * 60)

    import numpy as np

    skeleton_dir = os.path.join(OUTPUT_DIR, "skeletons")
    features_dir = os.path.join(OUTPUT_DIR, "features")

    def compute_clip_features(seq):
        feats = []
        feats.extend(np.mean(seq, axis=0))
        feats.extend(np.std(seq, axis=0))
        feats.extend(np.max(seq, axis=0))
        feats.extend(np.min(seq, axis=0))
        feats.extend(np.max(seq, axis=0) - np.min(seq, axis=0))

        if len(seq) > 1:
            velocity = np.diff(seq, axis=0)
            feats.extend(np.mean(velocity, axis=0))
            feats.extend(np.std(velocity, axis=0))
            if len(seq) > 2:
                accel = np.diff(velocity, axis=0)
                feats.extend(np.mean(accel, axis=0))
                feats.extend(np.std(accel, axis=0))
            else:
                feats.extend([0.0] * 264)
        else:
            feats.extend([0.0] * 528)

        return np.array(feats, dtype=np.float32)

    def pad_or_truncate(seq, target=TARGET_FRAMES):
        if len(seq) >= target:
            return seq[:target]
        padding = np.zeros((target - len(seq), seq.shape[1]), dtype=np.float32)
        return np.vstack([seq, padding])

    X_features = []
    X_sequences = []
    y_labels = []
    class_names = sorted([d for d in os.listdir(skeleton_dir)
                          if os.path.isdir(os.path.join(skeleton_dir, d))])

    for label_idx, class_name in enumerate(class_names):
        class_path = os.path.join(skeleton_dir, class_name)
        npy_files = [f for f in os.listdir(class_path) if f.endswith('.npy')]

        for npy_file in npy_files:
            skeleton = np.load(os.path.join(class_path, npy_file))

            if len(skeleton) == 0:
                continue

            skeleton = np.nan_to_num(skeleton, nan=0.0, posinf=0.0, neginf=0.0)

            X_features.append(compute_clip_features(skeleton))
            X_sequences.append(pad_or_truncate(skeleton))
            y_labels.append(label_idx)

    X_features = np.array(X_features)
    X_sequences = np.array(X_sequences)
    y_labels = np.array(y_labels)

    print(f"  Summary features: {X_features.shape} (clips x features)")
    print(f"  Padded sequences: {X_sequences.shape} (clips x frames x joints)")
    print(f"  Labels: {y_labels.shape}, Classes: {len(class_names)}")
    print(f"  Classes: {class_names}")

    np.save(os.path.join(features_dir, "X_features.npy"), X_features)
    np.save(os.path.join(features_dir, "X_sequences.npy"), X_sequences)
    np.save(os.path.join(features_dir, "y_labels.npy"), y_labels)

    with open(os.path.join(features_dir, "class_names.json"), 'w') as f:
        json.dump(class_names, f, indent=2)

    print(f"  Saved to: {features_dir}/")

    save_checkpoint(4, "Feature Engineering")
    return X_features, X_sequences, y_labels, class_names


def step5_train_test_split():
    print("=" * 60)
    print("STEP 5: Train/Test Split")
    print("=" * 60)

    import numpy as np
    from sklearn.model_selection import train_test_split

    features_dir = os.path.join(OUTPUT_DIR, "features")

    X = np.load(os.path.join(features_dir, "X_features.npy"))
    X_seq = np.load(os.path.join(features_dir, "X_sequences.npy"))
    y = np.load(os.path.join(features_dir, "y_labels.npy"))

    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    X_train, X_test = X[train_idx], X[test_idx]
    X_train_seq, X_test_seq = X_seq[train_idx], X_seq[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print(f"  Train: {len(X_train)} clips")
    print(f"  Test:  {len(X_test)} clips")

    np.save(os.path.join(features_dir, "X_train.npy"), X_train)
    np.save(os.path.join(features_dir, "X_test.npy"), X_test)
    np.save(os.path.join(features_dir, "X_train_seq.npy"), X_train_seq)
    np.save(os.path.join(features_dir, "X_test_seq.npy"), X_test_seq)
    np.save(os.path.join(features_dir, "y_train.npy"), y_train)
    np.save(os.path.join(features_dir, "y_test.npy"), y_test)
    np.save(os.path.join(features_dir, "train_indices.npy"), train_idx)
    np.save(os.path.join(features_dir, "test_indices.npy"), test_idx)

    split_info = {
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "test_ratio": TEST_SIZE,
        "random_seed": RANDOM_SEED,
        "stratified": True
    }
    with open(os.path.join(features_dir, "split_info.json"), 'w') as f:
        json.dump(split_info, f, indent=2)

    print(f"  Saved to: {features_dir}/")

    save_checkpoint(5, "Train/Test Split")


def step6_train_classifiers():
    print("=" * 60)
    print("STEP 6: Train Classifiers")
    print("=" * 60)

    import numpy as np
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, classification_report

    features_dir = os.path.join(OUTPUT_DIR, "features")
    models_dir = os.path.join(OUTPUT_DIR, "models")

    X_train = np.load(os.path.join(features_dir, "X_train.npy"))
    X_test = np.load(os.path.join(features_dir, "X_test.npy"))
    y_train = np.load(os.path.join(features_dir, "y_train.npy"))
    y_test = np.load(os.path.join(features_dir, "y_test.npy"))

    with open(os.path.join(features_dir, "class_names.json"), 'r') as f:
        class_names = json.load(f)

    print("  Scaling features...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))
    print("  Scaler saved.")

    results = {}

    print("\n  Training Random Forest...")
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    rf.fit(X_train_s, y_train)
    rf_time = time.time() - t0

    rf_pred = rf.predict(X_test_s)
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_report = classification_report(y_test, rf_pred, target_names=class_names,
                                       output_dict=True, zero_division=0)

    rf_mean_cls_acc = float(np.mean([rf_report[c]['recall'] for c in class_names]))

    print(f"  Random Forest — Top1: {rf_acc:.4f}, Mean Cls Acc: {rf_mean_cls_acc:.4f}, "
          f"Time: {rf_time:.1f}s")

    joblib.dump(rf, os.path.join(models_dir, "rf_model.pkl"))
    np.save(os.path.join(models_dir, "rf_predictions.npy"), rf_pred)
    print("  Random Forest model saved.")

    results["random_forest"] = {
        "top1_accuracy": round(rf_acc, 4),
        "mean_class_accuracy": round(rf_mean_cls_acc, 4),
        "training_time_seconds": round(rf_time, 1),
        "per_class_report": rf_report
    }

    print("\n  Training SVM (this may take a few minutes)...")
    t0 = time.time()
    svm = SVC(
        kernel='rbf',
        C=10,
        gamma='scale',
        random_state=RANDOM_SEED,
        decision_function_shape='ovr'
    )
    svm.fit(X_train_s, y_train)
    svm_time = time.time() - t0

    svm_pred = svm.predict(X_test_s)
    svm_acc = accuracy_score(y_test, svm_pred)
    svm_report = classification_report(y_test, svm_pred, target_names=class_names,
                                        output_dict=True, zero_division=0)
    svm_mean_cls_acc = float(np.mean([svm_report[c]['recall'] for c in class_names]))

    print(f"  SVM — Top1: {svm_acc:.4f}, Mean Cls Acc: {svm_mean_cls_acc:.4f}, "
          f"Time: {svm_time:.1f}s")

    joblib.dump(svm, os.path.join(models_dir, "svm_model.pkl"))
    np.save(os.path.join(models_dir, "svm_predictions.npy"), svm_pred)
    print("  SVM model saved.")

    results["svm"] = {
        "top1_accuracy": round(svm_acc, 4),
        "mean_class_accuracy": round(svm_mean_cls_acc, 4),
        "training_time_seconds": round(svm_time, 1),
        "per_class_report": svm_report
    }

    results_path = os.path.join(OUTPUT_DIR, "reports", "training_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Full results saved to: {results_path}")

    save_checkpoint(6, "Train Classifiers")
    return results


def step7_generate_visualizations():
    print("=" * 60)
    print("STEP 7: Generate Visualizations for Poster")
    print("=" * 60)

    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    from sklearn.decomposition import PCA
    import joblib

    features_dir = os.path.join(OUTPUT_DIR, "features")
    models_dir = os.path.join(OUTPUT_DIR, "models")
    plots_dir = os.path.join(OUTPUT_DIR, "plots")

    X_test = np.load(os.path.join(features_dir, "X_test.npy"))
    y_test = np.load(os.path.join(features_dir, "y_test.npy"))
    rf_pred = np.load(os.path.join(models_dir, "rf_predictions.npy"))
    svm_pred = np.load(os.path.join(models_dir, "svm_predictions.npy"))
    scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))

    with open(os.path.join(features_dir, "class_names.json"), 'r') as f:
        class_names = json.load(f)

    with open(os.path.join(OUTPUT_DIR, "reports", "training_results.json"), 'r') as f:
        results = json.load(f)

    X_test_s = scaler.transform(X_test)

    print("  Generating confusion matrix (Random Forest)...")
    cm_rf = confusion_matrix(y_test, rf_pred)
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(f'Confusion Matrix — Random Forest (Skeleton-Only)\n'
                 f'Top-1: {results["random_forest"]["top1_accuracy"]:.1%} | '
                 f'Mean Cls Acc: {results["random_forest"]["mean_class_accuracy"]:.1%}',
                 fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "confusion_matrix_rf.png"), dpi=300)
    plt.close()
    print("  confusion_matrix_rf.png saved")

    print("  Generating confusion matrix (SVM)...")
    cm_svm = confusion_matrix(y_test, svm_pred)
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Oranges',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(f'Confusion Matrix — SVM (Skeleton-Only)\n'
                 f'Top-1: {results["svm"]["top1_accuracy"]:.1%} | '
                 f'Mean Cls Acc: {results["svm"]["mean_class_accuracy"]:.1%}',
                 fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "confusion_matrix_svm.png"), dpi=300)
    plt.close()
    print("  confusion_matrix_svm.png saved")

    print("  Generating PCA scatter plot...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_test_s)

    fig, ax = plt.subplots(figsize=(12, 10))
    unique_labels = np.unique(y_test)
    cmap = plt.cm.get_cmap('tab20', len(unique_labels))

    for i, label in enumerate(unique_labels):
        mask = y_test == label
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=[cmap(i)], label=class_names[label],
                   alpha=0.5, s=15, edgecolors='none')

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)", fontsize=12)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)", fontsize=12)
    ax.set_title("PCA of Skeleton Features — 18 Stroke Classes", fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, markerscale=2)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "pca_scatter.png"), dpi=300,
                bbox_inches='tight')
    plt.close()
    print("  pca_scatter.png saved")

    pca_info = {
        "pc1_variance": round(float(pca.explained_variance_ratio_[0]), 4),
        "pc2_variance": round(float(pca.explained_variance_ratio_[1]), 4),
        "total_2pc_variance": round(float(sum(pca.explained_variance_ratio_[:2])), 4)
    }
    with open(os.path.join(plots_dir, "pca_info.json"), 'w') as f:
        json.dump(pca_info, f, indent=2)

    print("  Generating per-class accuracy chart...")
    rf_report = results["random_forest"]["per_class_report"]
    svm_report = results["svm"]["per_class_report"]

    rf_recalls = [rf_report[c]['recall'] for c in class_names]
    svm_recalls = [svm_report[c]['recall'] for c in class_names]

    x = np.arange(len(class_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(16, 7))
    bars1 = ax.bar(x - width/2, rf_recalls, width, label='Random Forest', color='steelblue')
    bars2 = ax.bar(x + width/2, svm_recalls, width, label='SVM', color='coral')

    ax.set_ylabel('Recall (Per-Class Accuracy)', fontsize=12)
    ax.set_title('Skeleton-Only Classification Accuracy by Stroke Type', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
    ax.legend()
    ax.axhline(y=np.mean(rf_recalls), color='steelblue', linestyle='--', alpha=0.5,
               label=f'RF Mean: {np.mean(rf_recalls):.1%}')
    ax.axhline(y=np.mean(svm_recalls), color='coral', linestyle='--', alpha=0.5,
               label=f'SVM Mean: {np.mean(svm_recalls):.1%}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "per_class_accuracy.png"), dpi=300)
    plt.close()
    print("  per_class_accuracy.png saved")

    print("  Generating benchmark comparison chart...")
    benchmarks = {
        "ST-GCN\n(skeleton)": 61.44,
        "PoseC3D\n(skeleton)": 67.18,
        "R(2+1)D\n(RGB)": 66.97,
        "Swin3D\n(RGB)": 69.93,
        "SlowFast\n(RGB)": 73.80,
        "TimeSformer\n(RGB)": 57.70,
        "Ours: RF\n(skeleton)": results["random_forest"]["mean_class_accuracy"] * 100,
        "Ours: SVM\n(skeleton)": results["svm"]["mean_class_accuracy"] * 100,
    }

    names = list(benchmarks.keys())
    values = list(benchmarks.values())
    colors = ['#4C72B0' if 'skeleton' in n.lower() else '#DD8452' for n in names]
    colors[-2] = '#2ca02c'
    colors[-1] = '#2ca02c'

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(names, values, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_ylabel('Mean Class Accuracy (%)', fontsize=12)
    ax.set_title('Benchmark Comparison — VideoBadminton Mean Class Accuracy', fontsize=14)
    ax.set_ylim(0, 100)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#4C72B0', label='Published Skeleton-Based'),
        Patch(facecolor='#DD8452', label='Published RGB-Based'),
        Patch(facecolor='#2ca02c', label='Ours (Skeleton-Only, CPU)')
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "benchmark_comparison.png"), dpi=300)
    plt.close()
    print("  benchmark_comparison.png saved")

    save_checkpoint(7, "Generate Visualizations")


def step8_poster_content_summary():
    print("=" * 60)
    print("STEP 8: Poster Content Summary")
    print("=" * 60)

    import numpy as np

    features_dir = os.path.join(OUTPUT_DIR, "features")

    with open(os.path.join(OUTPUT_DIR, "reports", "training_results.json"), 'r') as f:
        results = json.load(f)
    with open(os.path.join(features_dir, "class_names.json"), 'r') as f:
        class_names = json.load(f)
    with open(os.path.join(OUTPUT_DIR, "quality_report.json"), 'r') as f:
        quality = json.load(f)
    with open(os.path.join(OUTPUT_DIR, "plots", "pca_info.json"), 'r') as f:
        pca_info = json.load(f)

    rf_report = results["random_forest"]["per_class_report"]
    class_recalls = {c: rf_report[c]['recall'] for c in class_names}
    sorted_classes = sorted(class_recalls.items(), key=lambda x: x[1], reverse=True)

    top_3 = sorted_classes[:3]
    bottom_3 = sorted_classes[-3:]

    best_model = "random_forest" if (
        results["random_forest"]["mean_class_accuracy"] >=
        results["svm"]["mean_class_accuracy"]
    ) else "svm"

    poster = {
        "dataset": {
            "name": "VideoBadminton",
            "total_clips": quality["total"],
            "num_classes": len(class_names),
            "class_names": class_names,
            "failed_extractions": quality["failed"],
            "extraction_success_rate": round(1 - quality["failed"]/max(quality["total"],1), 4)
        },
        "results": {
            "best_model": best_model,
            "random_forest": {
                "top1_accuracy": results["random_forest"]["top1_accuracy"],
                "mean_class_accuracy": results["random_forest"]["mean_class_accuracy"]
            },
            "svm": {
                "top1_accuracy": results["svm"]["top1_accuracy"],
                "mean_class_accuracy": results["svm"]["mean_class_accuracy"]
            }
        },
        "analysis": {
            "best_classified_strokes": [{"class": c, "recall": round(r, 4)} for c, r in top_3],
            "worst_classified_strokes": [{"class": c, "recall": round(r, 4)} for c, r in bottom_3],
            "pca_variance_explained_2pc": pca_info["total_2pc_variance"]
        },
        "privacy_argument": {
            "skeleton_values_per_frame": 132,
            "rgb_values_per_frame_1080p": 6220800,
            "data_reduction_percent": 99.998
        },
        "benchmark_context": {
            "st_gcn_mean_cls_acc": 61.44,
            "posec3d_mean_cls_acc": 67.18,
            "slowfast_mean_cls_acc": 73.80,
            "note": "Our skeleton-only CPU pipeline vs GPU-trained deep models"
        },
        "plots_directory": os.path.abspath(os.path.join(OUTPUT_DIR, "plots")),
        "files_generated": [
            "confusion_matrix_rf.png",
            "confusion_matrix_svm.png",
            "pca_scatter.png",
            "per_class_accuracy.png",
            "benchmark_comparison.png"
        ]
    }

    poster_path = os.path.join(OUTPUT_DIR, "reports", "poster_content.json")
    with open(poster_path, 'w') as f:
        json.dump(poster, f, indent=2)

    print(f"\n  POSTER CONTENT SUMMARY:")
    print(f"  Best model: {best_model}")
    print(f"  RF  — Top1: {results['random_forest']['top1_accuracy']:.1%}, "
          f"Mean Cls: {results['random_forest']['mean_class_accuracy']:.1%}")
    print(f"  SVM — Top1: {results['svm']['top1_accuracy']:.1%}, "
          f"Mean Cls: {results['svm']['mean_class_accuracy']:.1%}")
    print(f"\n  Best strokes:  {[f'{c} ({r:.0%})' for c, r in top_3]}")
    print(f"  Worst strokes: {[f'{c} ({r:.0%})' for c, r in bottom_3]}")
    print(f"\n  All poster data saved to: {poster_path}")
    print(f"  All plots saved to: {os.path.join(OUTPUT_DIR, 'plots')}/")

    save_checkpoint(8, "Poster Content Summary")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("BADMINTON STROKE CLASSIFICATION PIPELINE")
    print(f"Resume from step: {RESUME_FROM_STEP}")
    print(f"Output directory:  {OUTPUT_DIR}")
    print("=" * 60 + "\n")

    if RESUME_FROM_STEP <= 1:
        step1_dataset_inventory()

    if RESUME_FROM_STEP <= 2:
        step2_extract_skeletons()

    if RESUME_FROM_STEP <= 3:
        step3_quality_check()

    if RESUME_FROM_STEP <= 4:
        step4_feature_engineering()

    if RESUME_FROM_STEP <= 5:
        step5_train_test_split()

    if RESUME_FROM_STEP <= 6:
        step6_train_classifiers()

    if RESUME_FROM_STEP <= 7:
        step7_generate_visualizations()

    if RESUME_FROM_STEP <= 8:
        step8_poster_content_summary()

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print(f"All outputs in: {OUTPUT_DIR}")
    print("=" * 60)
