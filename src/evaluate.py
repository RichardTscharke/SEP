from evaluation.scripts.inference import calculate_inference
from evaluation.scripts.epoch_curves import plot_epoch_curves
from evaluation.scripts.precision_recall_f1_per_class import plot_prec_recall_f1_p_class
from evaluation.scripts.confusion_matrix import plot_confusion_matrix

def main():

    calculate_inference(MODEL_PATH = "models/model1_v0.pth")
    print(f"[INFO] Inference calculated.")

    plot_epoch_curves(csv_path = "src/evaluation/outputs/epoch_metrics.csv")
    print(f"[INFO] Epoch curves drawn.")

    plot_prec_recall_f1_p_class()
    print(f"[INFO] Precision, Recall and F1 per class diagramm created.")

    plot_confusion_matrix(normalized = True)
    print(f"[INFO] Confusion Matrix drawn.")


if __name__ == "__main__":
    main()
