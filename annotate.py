import sys
import csv
import re

from enum import StrEnum, auto
from dataclasses import dataclass


# Card types we need to annotate are:
# • Artifact
# • Creature
# • Enchantment
# • Instant
# • Land
# • Planeswalker
# • Sorcery
CARD_TYPES = (
    "Artifact",
    "Creature",
    "Enchantment",
    "Instant",
    "Land",
    "Planeswalker",
    "Sorcery",
)


class CardType(StrEnum):
    ARTIFACT = auto()
    CREATURE = auto()
    ENCHANTMENT = auto()
    INSTANT = auto()
    LAND = auto()
    PLANESWALKER = auto()
    SORCERY = auto()

    def from_card_type_string(card_type_string: str) -> "CardType":
        for card_type in CARD_TYPES:
            if card_type in card_type_string:
                return CardType[card_type.upper()]
        raise ValueError(f"Unknown card type: {card_type_string}")


@dataclass
class Card:
    tokenized_name: list[str]
    type: CardType


# Map of the first word of a card title to the full card title and type.
# When multiple cards begin with the same word, they are stored under the same key,
# hence list[Card] being the value type.
card_names: dict[str, list[Card]] = {}
# List of card rule text.
card_rules: list[str] = []
# Each word in each rule text is annotated with the card type, if it is part of a card name.
card_rule_text_with_found_card_types: list[list[tuple[str, CardType | None]]] = []


def tokenize(rule_text: str) -> list[str]:
    # We will use a regular expression to split the rule text into words, while keeping punctuation as separate tokens.
    # This is important because card names may contain punctuation, and we want to be able to match them correctly.
    # Contractions are split into two tokens, where the latter token includes the apostrophe, e.g. "can't" is split into "can" and "'t".
    return re.findall(r"'?\w+|[^\w\s]", rule_text, re.UNICODE)


def read_csv_file(csv_file_path: str) -> None:
    with open(csv_file_path, "r") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            card_name = row["name"]
            tokenized_card_name = tokenize(card_name)
            card_type_string = row["type"]
            card_type = CardType.from_card_type_string(card_type_string)
            card = Card(tokenized_name=tokenized_card_name, type=card_type)

            first_word = tokenized_card_name[0]
            if first_word not in card_names:
                card_names[first_word] = []
            card_names[first_word].append(card)

            card_rules.append(row["rules"])



def annotate_card_rules() -> list[list[tuple[str, CardType | None]]]:
    annotated_card_rules: list[list[tuple[str, CardType | None]]] = []
    for rule in card_rules:
        # If a word in the rule text matches the first word of a card name, we need to check if the subsequent words in the rule text
        # match the full card name. If they do, we can annotate the words in the rule text with the card type.
        match_cards: list[Card] = []
        match_pos = 0
        words = tokenize(rule)

        # We will store the annotated rule text as a list of tuples, where each tuple contains a word and its corresponding card type (if it is part of a card name).
        annotated_rule_text: list[tuple[str, CardType | None]] = []

        for pos, word in enumerate(words):
            if len(match_cards) > 0:
                # If we have already matched some card names, we need to check if the current word is a continuation of any of the matched card names.
                for card in match_cards:
                    card_name = card.tokenized_name
                    if match_pos < len(card_name):
                        if word == card_name[match_pos]:
                            match_pos += 1
                            if match_pos == len(card_name):
                                # We have matched the full card name, so we can add it to the list of matched cards.
                                annotated_rule_text.extend([(w, card.type) for w in card_name])
                                match_cards = []  # Reset the match cards list for the next card name.
                                match_pos = 0
                        else:
                            match_cards.remove(card)
            elif word in card_names:
                match_pos = 1
                match_cards = card_names[word]
                # If any of the card names consist of only one word, we can immediately annotate it and reset the match cards list.
                for card in match_cards:
                    if card.tokenized_name == [word]:
                        annotated_rule_text.append((word, card.type))
                        match_cards.remove(card)
                        match_pos = 0
            else:
                annotated_rule_text.append((word, None))
        
        annotated_card_rules.append(annotated_rule_text)
    return annotated_card_rules


def print_to_iob2(annotated_card_rules: list[list[tuple[str, CardType | None]]]) -> None:
    for annotated_rule in annotated_card_rules:
        is_beginning = True
        for word, card_type in annotated_rule:
            if card_type is not None:
                print(f"{word}\t{'B' if is_beginning else 'I'}-{card_type.upper()}")
                is_beginning = False
            else:
                is_beginning = True
                print(f"{word}\tO")
        print()  # Print a blank line after each rule text.


def main(*args) -> None:
    csv_file_path = args[0]

    read_csv_file(csv_file_path)
    print(f"Read {len(card_names)} unique card names and {len(card_rules)} card rules from the CSV file.")

    annotated_card_rules = annotate_card_rules()
    print_to_iob2(annotated_card_rules)


if __name__ == "__main__":
    main(*sys.argv[1:])
