import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Set serif font globally (match style in plot_combined_figure.py)
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["DejaVu Serif", "Liberation Serif", "Times New Roman", "serif"]
plt.rcParams["mathtext.fontset"] = "stix"

# Configuration
BASE_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(BASE_DIR, "fewdata")

COLORS = {
	"TTRL": "#4C78A8",  # blue
	"TTCS": "#F58518",  # orange
}

PROPORTIONS = ["10%", "20%", "30%", "100%"]
DATA = {
	"TTRL": [9.48, 11.77, 11.88, 13.23],
	"TTCS": [13.33, 14.17, 14.48, 19.79],
}


def plot_fewdata_curve():
	"""Plot accuracy curves under different data proportions."""
	TITLE_SIZE = 12
	LABEL_SIZE = 14
	TICK_SIZE = 14
	LEGEND_SIZE = 12
	POINT_LABEL_SIZE = 9

	x = np.arange(len(PROPORTIONS))

	fig, ax = plt.subplots(figsize=(4.8, 5.2))

	for xpos in x:
		ax.axvline(x=xpos, color="#D0D0D0", linewidth=0.8, alpha=0.8, zorder=0)

	ax.plot(
		x,
		DATA["TTRL"],
		label="TTRL",
		color=COLORS["TTRL"],
		marker="o",
		linewidth=2.5,
		markersize=7,
	)
	ax.plot(
		x,
		DATA["TTCS"],
		label="TTCS(Ours)",
		color=COLORS["TTCS"],
		marker="D",
		linewidth=2.5,
		markersize=7,
	)

	ax.set_ylim(8.5, 21)
	ax.set_xticks(x)
	ax.set_xticklabels(PROPORTIONS, fontsize=TICK_SIZE)
	ax.set_xlabel("Test data proportion", fontsize=LABEL_SIZE)
	ax.set_ylabel("Accuracy (%)", fontsize=LABEL_SIZE)
	ax.tick_params(axis="y", labelsize=TICK_SIZE)
	ax.grid(True, linestyle="-", alpha=0.6, color="#B0B0B0", linewidth=0.8, axis="y")

	for idx, value in enumerate(DATA["TTRL"]):
		ax.annotate(
			f"{value:.2f}",
			xy=(x[idx], value),
			xytext=(0, 6),
			textcoords="offset points",
			ha="center",
			va="bottom",
			fontsize=POINT_LABEL_SIZE,
		)
	for idx, value in enumerate(DATA["TTCS"]):
		ax.annotate(
			f"{value:.2f}",
			xy=(x[idx], value),
			xytext=(0, 6),
			textcoords="offset points",
			ha="center",
			va="bottom",
			fontsize=POINT_LABEL_SIZE,
		)

	legend = ax.legend(
		loc="lower right",
		ncol=1,
		fontsize=LEGEND_SIZE - 2,
		frameon=False,
	)
	for handle in legend.legend_handles:
		handle.set_linewidth(2)
		if hasattr(handle, "set_markersize"):
			handle.set_markersize(5)
	fig.subplots_adjust(top=0.86)
	plt.tight_layout(rect=[0, 0, 1, 0.86])

	output_png = os.path.join(OUTPUT_DIR, "fewdata_accuracy_curve.png")
	plt.savefig(output_png, dpi=300, bbox_inches="tight", facecolor="white")
	output_pdf = os.path.join(OUTPUT_DIR, "fewdata_accuracy_curve.pdf")
	plt.savefig(output_pdf, bbox_inches="tight", facecolor="white")
	plt.close(fig)
	print(f"Figure saved to: {output_png}")
	print(f"PDF saved to: {output_pdf}")


def main():
	os.makedirs(OUTPUT_DIR, exist_ok=True)
	print(f"Output directory: {OUTPUT_DIR}")
	plot_fewdata_curve()
	print("Done!")


if __name__ == "__main__":
	main()
