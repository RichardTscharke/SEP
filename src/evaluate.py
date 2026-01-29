from evaluation.scripts.inference import calculate_inference
from evaluation.scripts.epoch_curves import plot_epoch_curves
from evaluation.scripts.precision_recall_f1_per_class import plot_prec_recall_f1_p_class
from evaluation.scripts.confusion_matrix import plot_confusion_matrix

def main():

    calculate_inference(MODEL_PATH = "models/model1_v0.pth")

    plot_epoch_curves(csv_path = "src/evaluation/outputs/epoch_metrics.csv")

    plot_prec_recall_f1_p_class()

    plot_confusion_matrix(normalized = True)


if __name__ == "__main__":
    main()