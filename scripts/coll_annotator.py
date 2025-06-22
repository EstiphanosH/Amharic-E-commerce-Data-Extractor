import pandas as pd
import ast
import os
import sys
from tqdm import tqdm


class CoNLLAnnotator:
    def __init__(self, sample_path="ner_labeling_sample.csv"):
        """
        Initialize the CoNLL annotator with a sample dataset
        """
        self.sample = pd.read_csv(sample_path)
        self.labels = []
        self.entity_types = {
            "B-PRODUCT": "Product (Beginning)",
            "I-PRODUCT": "Product (Inside)",
            "B-LOC": "Location (Beginning)",
            "I-LOC": "Location (Inside)",
            "B-PRICE": "Price (Beginning)",
            "I-PRICE": "Price (Inside)",
            "O": "Other",
        }

    def start_cli_labeling(self):
        """
        Start command-line interface for labeling tokens
        """
        print("Starting CoNLL Annotation Tool")
        print("=" * 60)
        print("Entity Types:")
        for code, label in self.entity_types.items():
            print(f"{code}: {label}")
        print("=" * 60)
        print(f"Total messages to label: {len(self.sample)}")

        for i, row in self.sample.iterrows():
            os.system("cls" if os.name == "nt" else "clear")
            print(f"Message {i+1}/{len(self.sample)}")
            print(f"ID: {row['message_id']}")
            print("-" * 60)
            print(row["cleaned_text"])
            print("-" * 60)

            tokens = ast.literal_eval(row["tokens"])
            message_labels = []

            for j, token in enumerate(tokens):
                print(f"\nToken {j+1}/{len(tokens)}: {token}")
                print("Select label:")
                for code in self.entity_types:
                    print(f"{code[0]} - {code}")

                choice = input("Choice (or 'b' to go back): ").upper()
                if choice == "B":
                    if message_labels:
                        message_labels.pop()
                    tokens = tokens[: len(message_labels)]
                    if tokens:
                        token = tokens[-1]
                        j = len(tokens) - 1
                        print(f"\nRepeating token {j+1}/{len(tokens)}: {token}")
                    continue

                if choice in [k[0] for k in self.entity_types]:
                    label_code = [k for k in self.entity_types if k.startswith(choice)][
                        0
                    ]
                    message_labels.append((token, label_code))
                else:
                    message_labels.append((token, "O"))

            self.labels.append(
                {
                    "message_id": row["message_id"],
                    "tokens": tokens,
                    "labels": [l[1] for l in message_labels],
                }
            )
            print(f"Message {i+1} completed!")

        print("\nLabeling complete! Saving results...")
        self.save_to_conll()

    def save_to_conll(self, output_path="labeled_data.conll"):
        """
        Save labeled data to CoNLL format file
        """
        with open(output_path, "w", encoding="utf-8") as f:
            for item in self.labels:
                for token, label in zip(item["tokens"], item["labels"]):
                    f.write(f"{token}\t{label}\n")
                f.write("\n")
        print(f"Saved {len(self.labels)} messages to {output_path}")
        print("You can now proceed to model training with this file")


def main():
    if not os.path.exists("ner_labeling_sample.csv"):
        print("Error: ner_labeling_sample.csv not found!")
        print("Run the preprocessing notebook first to generate this file")
        return

    annotator = CoNLLAnnotator()
    annotator.start_cli_labeling()


if __name__ == "__main__":
    main()
