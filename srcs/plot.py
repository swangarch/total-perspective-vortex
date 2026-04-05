from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, classification_report


def show_confusion_matrix(y_test, y_pred,
                          labels=["Prediction","left hand", "right hand"]):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=labels[1:],
                                  )

    f1 = f1_score(y_test, y_pred, average="weighted")
    acc = (y_test == y_pred).mean()
    
    disp.plot(colorbar=False)
    disp.ax_.set_title(f"{labels[0]} -> f1: {f1:.3f} | acc: {acc:.3f}")
    plt.pause(5)
    plt.show()
    plt.close()


def show_single_epoch(np_data: np.array):
    show_epoch = 2
    plt.title("Single epoch in 1 channel example.")
    plt.xlabel("time")
    plt.ylabel("amplitude")
    for i in range(len(np_data)):
        plt.plot(np_data[i][show_epoch], linewidth=0.5, label="event")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.ylim(-120e-6, 120e-6)
    plt.legend()
    plt.show()


def show_edf(raw, montage, file: str, title: str) -> None:
    print(f"Visualize {title} file: {file}")
    raw.plot(
        scalings={"eeg": 120e-6},
    )
    plt.show()

    raw.set_montage(montage)
    raw.compute_psd().plot(spatial_colors=True)
    plt.show()
    
    raw.compute_psd().plot_topomap()