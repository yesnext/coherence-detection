import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = PROJECT_ROOT / "outputs" / "official_roberta_full"
DEFAULT_THRESHOLD = 0.5
DEFAULT_WINDOW_SIZE = 5
DEFAULT_STRIDE = 1
ABBREVIATIONS = {
    "Mr.",
    "Mrs.",
    "Ms.",
    "Dr.",
    "Prof.",
    "Sr.",
    "Jr.",
    "St.",
    "vs.",
    "e.g.",
    "i.e.",
    "etc.",
    "U.S.",
    "U.K.",
}


@dataclass
class WindowScore:
    window_index: int
    start_sentence: int
    end_sentence: int
    paragraph_index: Optional[int]
    coherent_score: float
    incoherent_score: float
    label: str
    severity: str
    text: str


@dataclass
class StoryAnalysis:
    sentence_count: int
    window_size: int
    stride: int
    segmentation_mode: str
    threshold: float
    overall_score: float
    min_window_score: float
    label: str
    severity: str
    advice: List[str]
    suspicious_sentences: List[int]
    sentences: List[str]
    windows: List[WindowScore]


def split_sentences(text: str) -> List[str]:
    """Dependency-free sentence splitter with small abbreviation protection."""

    normalized = re.sub(r"\s+", " ", text.strip())
    if not normalized:
        return []

    placeholders = {}
    protected = normalized
    for idx, abbr in enumerate(sorted(ABBREVIATIONS, key=len, reverse=True)):
        token = f"<ABBR{idx}>"
        protected = protected.replace(abbr, token)
        placeholders[token] = abbr

    pieces = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", protected)
    sentences = [piece.strip() for piece in pieces if piece.strip()]

    # If the user provided line-separated sentences without punctuation.
    if len(sentences) == 1 and "\n" in text:
        sentences = [piece.strip() for piece in text.splitlines() if piece.strip()]

    for token, abbr in placeholders.items():
        sentences = [sentence.replace(token, abbr) for sentence in sentences]

    return sentences


def split_paragraphs(text: str) -> List[str]:
    paragraphs = re.split(r"\n\s*\n+", text.strip())
    return [paragraph.strip() for paragraph in paragraphs if paragraph.strip()]


def severity_from_score(score: float) -> str:
    if score >= 0.85:
        return "Strongly coherent"
    if score >= 0.65:
        return "Mostly coherent"
    if score >= 0.45:
        return "Uncertain"
    if score >= 0.25:
        return "Likely incoherent"
    return "Strongly incoherent"


def ascii_bar(score: float, width: int = 24) -> str:
    filled = int(round(score * width))
    return "#" * filled + "-" * (width - filled)


def make_windows(sentences: List[str], window_size: int, stride: int) -> List[tuple[int, Optional[int], List[str]]]:
    if window_size <= 0:
        raise ValueError("window_size must be greater than 0.")
    if stride <= 0:
        raise ValueError("stride must be greater than 0.")
    if not sentences:
        return []
    if len(sentences) <= window_size:
        return [(0, None, sentences)]

    windows = []
    for start in range(0, len(sentences) - window_size + 1, stride):
        windows.append((start, None, sentences[start : start + window_size]))

    last_window = sentences[-window_size:]
    if windows[-1][2] != last_window:
        windows.append((len(sentences) - window_size, None, last_window))
    return windows


def make_paragraph_windows(text: str, window_size: int, stride: int) -> tuple[List[str], List[tuple[int, Optional[int], List[str]]]]:
    all_sentences = []
    windows = []
    sentence_offset = 0

    for paragraph_index, paragraph in enumerate(split_paragraphs(text), start=1):
        paragraph_sentences = split_sentences(paragraph)
        if not paragraph_sentences:
            continue

        all_sentences.extend(paragraph_sentences)
        paragraph_windows = make_windows(paragraph_sentences, window_size=window_size, stride=stride)
        for local_start, _, window in paragraph_windows:
            windows.append((sentence_offset + local_start, paragraph_index, window))
        sentence_offset += len(paragraph_sentences)

    return all_sentences, windows


class CoherenceAnalyzer:
    def __init__(
        self,
        model_dir: Path = DEFAULT_MODEL_DIR,
        threshold: float = DEFAULT_THRESHOLD,
        device: Optional[str] = None,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.threshold = threshold
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        config_path = self.model_dir / "run_config.json"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
        else:
            self.config = {"model_name": "roberta-base", "max_length": 128}

        self.max_length = int(self.config.get("max_length", 128))
        self.model_name = self.config.get("model_name", "roberta-base")
        self.checkpoint_path = self.model_dir / "best_model.pt"
        self.tokenizer_path = self.model_dir / "tokenizer"

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Missing model checkpoint: {self.checkpoint_path}")
        if not self.tokenizer_path.exists():
            raise FileNotFoundError(f"Missing tokenizer directory: {self.tokenizer_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, local_files_only=True)
        config = AutoConfig.from_pretrained(self.model_name, num_labels=2, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_config(config)
        self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device, weights_only=True))
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def score_sentences(self, sentences: Iterable[str]) -> tuple[float, float]:
        story_text = " [SEP] ".join(sentences)
        encoding = self.tokenizer(
            story_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()[0]
        incoherent_score = float(probs[0])
        coherent_score = float(probs[1])
        return incoherent_score, coherent_score

    def analyze_text(
        self,
        text: str,
        window_size: int = DEFAULT_WINDOW_SIZE,
        stride: int = DEFAULT_STRIDE,
        segmentation_mode: str = "paragraph",
    ) -> StoryAnalysis:
        if segmentation_mode == "paragraph":
            sentences, windows = make_paragraph_windows(text, window_size=window_size, stride=stride)
        elif segmentation_mode == "sliding":
            sentences = split_sentences(text)
            windows = make_windows(sentences, window_size=window_size, stride=stride)
        else:
            raise ValueError("segmentation_mode must be 'paragraph' or 'sliding'.")

        window_scores = []

        for idx, (start, paragraph_index, window) in enumerate(windows, start=1):
            start_sentence = start + 1
            end_sentence = start_sentence + len(window) - 1
            incoherent_score, coherent_score = self.score_sentences(window)
            label = "Coherent" if coherent_score >= self.threshold else "Incoherent"
            severity = severity_from_score(coherent_score)
            window_scores.append(
                WindowScore(
                    window_index=idx,
                    start_sentence=start_sentence,
                    end_sentence=end_sentence,
                    paragraph_index=paragraph_index,
                    coherent_score=coherent_score,
                    incoherent_score=incoherent_score,
                    label=label,
                    severity=severity,
                    text=" ".join(window),
                )
            )

        coherent_scores = [window.coherent_score for window in window_scores]
        overall_score = float(np.mean(coherent_scores)) if coherent_scores else 0.0
        min_window_score = float(np.min(coherent_scores)) if coherent_scores else 0.0
        label = "Coherent" if overall_score >= self.threshold and min_window_score >= self.threshold else "Incoherent"
        severity = severity_from_score(min_window_score if label == "Incoherent" else overall_score)
        suspicious_sentences = find_suspicious_sentences(window_scores, len(sentences), self.threshold)
        advice = build_advice(label, overall_score, min_window_score, suspicious_sentences, len(sentences), window_scores)

        return StoryAnalysis(
            sentence_count=len(sentences),
            window_size=window_size,
            stride=stride,
            segmentation_mode=segmentation_mode,
            threshold=self.threshold,
            overall_score=overall_score,
            min_window_score=min_window_score,
            label=label,
            severity=severity,
            advice=advice,
            suspicious_sentences=suspicious_sentences,
            sentences=sentences,
            windows=window_scores,
        )


def find_suspicious_sentences(
    windows: List[WindowScore],
    sentence_count: int,
    threshold: float,
) -> List[int]:
    if sentence_count == 0:
        return []

    penalties = np.zeros(sentence_count, dtype=float)
    coverage = np.zeros(sentence_count, dtype=float)

    for window in windows:
        penalty = max(0.0, threshold - window.coherent_score)
        for sentence_number in range(window.start_sentence, window.end_sentence + 1):
            idx = sentence_number - 1
            penalties[idx] += penalty
            coverage[idx] += 1

    scores = np.divide(penalties, coverage, out=np.zeros_like(penalties), where=coverage > 0)
    if scores.max() <= 0:
        return []

    cutoff = max(scores.max() * 0.75, 0.05)
    return [idx + 1 for idx, score in enumerate(scores) if score >= cutoff]


def build_advice(
    label: str,
    overall_score: float,
    min_window_score: float,
    suspicious_sentences: List[int],
    sentence_count: int,
    windows: List[WindowScore],
) -> List[str]:
    advice = []
    if sentence_count < DEFAULT_WINDOW_SIZE:
        advice.append(
            f"The story has fewer than {DEFAULT_WINDOW_SIZE} sentences, so the score is based on a shorter-than-trained input."
        )
    if label == "Coherent":
        advice.append("The story is consistently above the coherence threshold across all analyzed windows.")
        if overall_score < 0.75:
            advice.append("The score is positive but not very high; consider adding clearer temporal or causal links.")
        return advice

    advice.append("At least one section falls below the coherence threshold, so the full story is flagged as incoherent.")
    if min_window_score < 0.25:
        advice.append("One or more windows are strongly incoherent; check for sudden jumps in time, location, or event order.")
    if suspicious_sentences:
        sentence_list = ", ".join(f"S{idx}" for idx in suspicious_sentences)
        advice.append(f"The most suspicious sentence positions are: {sentence_list}.")

    weak_ranges = [f"S{w.start_sentence}-S{w.end_sentence}" for w in select_flagged_windows(windows, threshold=0.5)]
    if weak_ranges:
        advice.append("Review these lowest-scoring ranges first: " + ", ".join(weak_ranges) + ".")
    advice.append(
        "For long stories, paragraph-aware scoring is used by default to avoid confusing natural paragraph transitions with incoherence."
    )
    return advice


def format_report(analysis: StoryAnalysis) -> str:
    lines = [
        "Narrative Coherence Analysis",
        "=" * 28,
        f"Sentences: {analysis.sentence_count}",
        f"Window size: {analysis.window_size}",
        f"Stride: {analysis.stride}",
        f"Segmentation mode: {analysis.segmentation_mode}",
        f"Overall coherent score: {analysis.overall_score:.4f}",
        f"Lowest window score: {analysis.min_window_score:.4f}",
        f"Final label: {analysis.label}",
        f"Severity: {analysis.severity}",
        "",
        "Sentence Split",
        "-" * 14,
    ]
    for idx, sentence in enumerate(analysis.sentences, start=1):
        lines.append(f"{idx}. {sentence}")

    lines.extend(["", "Window Scores", "-" * 13])
    for window in analysis.windows:
        paragraph = f"P{window.paragraph_index} " if window.paragraph_index is not None else ""
        lines.append(
            f"Window {window.window_index} "
            f"({paragraph}S{window.start_sentence}-S{window.end_sentence}): "
            f"{window.label} | {window.severity} | "
            f"coherent={window.coherent_score:.4f} "
            f"| {ascii_bar(window.coherent_score)} | "
            f"incoherent={window.incoherent_score:.4f}"
        )

    if analysis.advice:
        lines.extend(["", "Readable Advice", "-" * 15])
        for item in analysis.advice:
            lines.append(f"- {item}")

    if analysis.suspicious_sentences:
        lines.extend(["", "Likely Problem Sentences", "-" * 24])
        for sentence_number in analysis.suspicious_sentences:
            if 1 <= sentence_number <= len(analysis.sentences):
                lines.append(f"S{sentence_number}: {analysis.sentences[sentence_number - 1]}")

    weak_windows = select_flagged_windows(analysis.windows, threshold=analysis.threshold)
    if weak_windows:
        lines.extend(["", "Flagged Sections", "-" * 16])
        for window in weak_windows:
            paragraph = f"P{window.paragraph_index}, " if window.paragraph_index is not None else ""
            lines.append(f"{paragraph}S{window.start_sentence}-S{window.end_sentence}: {window.text}")

    return "\n".join(lines)


def select_flagged_windows(windows: List[WindowScore], threshold: float = DEFAULT_THRESHOLD, limit: int = 5) -> List[WindowScore]:
    weak_windows = [window for window in windows if window.coherent_score < threshold]
    weak_windows.sort(key=lambda window: window.coherent_score)
    return weak_windows[:limit]


def save_visualization(analysis: StoryAnalysis, output_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required for --plot output.") from exc

    labels = [
        f"P{window.paragraph_index}:S{window.start_sentence}-S{window.end_sentence}"
        if window.paragraph_index is not None
        else f"S{window.start_sentence}-S{window.end_sentence}"
        for window in analysis.windows
    ]
    scores = [window.coherent_score for window in analysis.windows]
    colors = [
        "#2e7d32" if score >= 0.85 else
        "#7cb342" if score >= 0.65 else
        "#f9a825" if score >= 0.45 else
        "#ef6c00" if score >= 0.25 else
        "#c62828"
        for score in scores
    ]

    fig_width = max(8, len(labels) * 0.65)
    plt.figure(figsize=(fig_width, 5))
    bars = plt.bar(labels, scores, color=colors)
    plt.axhline(analysis.threshold, color="black", linestyle="--", linewidth=1, label="Decision threshold")
    plt.ylim(0, 1.05)
    plt.xlabel("Sentence window")
    plt.ylabel("Coherent probability")
    plt.title("Window-Level Narrative Coherence Scores")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()

    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width() / 2, score + 0.02, f"{score:.2f}", ha="center", va="bottom", fontsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def export_report(analysis: StoryAnalysis, output_path: Path, as_json: bool = False) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if as_json or output_path.suffix.lower() == ".json":
        output_path.write_text(json.dumps(asdict(analysis), indent=2), encoding="utf-8")
    else:
        output_path.write_text(format_report(analysis), encoding="utf-8")


def read_input(args: argparse.Namespace) -> str:
    if args.text:
        return args.text
    if args.file:
        return Path(args.file).read_text(encoding="utf-8")
    raise ValueError("Provide story input with --text or --file.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run narrative coherence checking on an input story.")
    parser.add_argument("--text", help="Raw story text to analyze.")
    parser.add_argument("--file", help="Path to a text file containing the story.")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR), help="Directory containing best_model.pt and tokenizer/.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument(
        "--segmentation-mode",
        choices=["paragraph", "sliding"],
        default="paragraph",
        help="Use paragraph-aware chunks for long stories, or flat sliding windows across the whole text.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON instead of a text report.")
    parser.add_argument("--output", help="Save the report to a .txt or .json file.")
    parser.add_argument("--plot", help="Save a PNG bar chart of window-level coherence scores.")
    args = parser.parse_args()

    story_text = read_input(args)
    analyzer = CoherenceAnalyzer(model_dir=Path(args.model_dir), threshold=args.threshold)
    analysis = analyzer.analyze_text(
        story_text,
        window_size=args.window_size,
        stride=args.stride,
        segmentation_mode=args.segmentation_mode,
    )

    if args.output:
        export_report(analysis, Path(args.output), as_json=args.json)
    if args.plot:
        save_visualization(analysis, Path(args.plot))

    if args.json:
        print(json.dumps(asdict(analysis), indent=2))
    else:
        print(format_report(analysis))


if __name__ == "__main__":
    main()
