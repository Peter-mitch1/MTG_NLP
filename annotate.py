import sys
import csv
import re

from enum import StrEnum, auto
from dataclasses import dataclass
from typing import cast


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


# Unwanted types: map them to one of the above types
TYPE_CONVERSION_MAP = {
    'Instant': 'Instant',
    'Legendary Creature': 'Creature',
    'Artifact Creature': 'Creature',
    'Legendary Artifact Land': 'Land',
    'Legendary Enchantment': 'Enchantment',
    'Enchantment Creature': 'Creature',
    'Legendary Artifact Creature': 'Creature',
    'Legendary Artifact': 'Artifact',
    'Kindred Enchantment': 'Enchantment',
    'Kindred Sorcery': 'Sorcery',
    'Snow Land': 'Land',
    'Legendary Land': 'Land',
    'World Enchantment': 'Enchantment',
    'Legendary Enchantment Creature': 'Creature',
    'Land Creature': 'Land',
    'Artifact Land': 'Land',
    'Snow Sorcery': 'Sorcery',
    'Tribal Enchantment': 'Enchantment',
    'Legendary Enchantment Artifact': 'Enchantment',
    'Legendary Sorcery': 'Sorcery',
    'Kindred Instant': 'Instant',
    'Enchantment Land': 'Land',
    'Snow Creature': 'Creature',
    'Legendary Snow Creature': 'Creature',
    'Snow Enchantment': 'Enchantment',
    'Tribal Instant': 'Instant',
    'Legendary Snow Enchantment': 'Enchantment',
    'Kindred Artifact': 'Artifact',
    'Legendary Instant': 'Instant',
    'Legendary Snow Land': 'Land',
    'Snow Instant': 'Instant',
    'Legendary Artifact Planeswalker': 'Planeswalker',
    'Snow Artifact': 'Artifact',
    'Snow Artifact Creature': 'Creature',
    'Legendary Planeswalker': 'Planeswalker'
}


class CardType(StrEnum):
    ARTIFACT = auto()
    CREATURE = auto()
    ENCHANTMENT = auto()
    INSTANT = auto()
    LAND = auto()
    PLANESWALKER = auto()
    SORCERY = auto()

    def from_card_type_string(card_type_string: str) -> "CardType":
        card_type_string = card_type_string.split("—")[0].strip()  # Handle cases where the card type string contains a subtype, e.g. "Creature — Human"
        for card_type in CARD_TYPES:
            if card_type == card_type_string:
                return CardType[card_type.upper()]
            elif card_type_string in TYPE_CONVERSION_MAP and TYPE_CONVERSION_MAP[card_type_string] == card_type:
                print(f"Converting card type '{card_type_string}' to '{card_type}'")
                return CardType[card_type.upper()]
        raise ValueError(f"Unknown card type: {card_type_string}")


@dataclass
class Card:
    tokenized_name: list[str | None]  # Card names end with None, so that cards sharing the same beginning are classified correctly
    type: CardType


# Map of the first word of a card title to the full card title and type.
# When multiple cards begin with the same word, they are stored under the same key,
# hence list[Card] being the value type.
card_names: dict[str, list[Card]] = {}
# List of card rule text.
card_rules: list[str] = []
# Each word in each rule text is annotated with the card type, if it is part of a card name.
card_rule_text_with_found_card_types: list[list[tuple[str, CardType | None]]] = []


def tokenize(rule_text: str) -> list[str | None]:
    # We will use a regular expression to split the rule text into words, while keeping punctuation as separate tokens.
    # This is important because card names may contain punctuation, and we want to be able to match them correctly.
    # Contractions are split into two tokens, where the latter token includes the apostrophe, e.g. "can't" is split into "can" and "'t".
    # 
    # Should there be more card names that share the same beginning (Example "Giant" and "Giant Crusher"),
    # the None token at the end of the tokenized card name will help us to correctly classify the card type of "Giant" when we encounter it in the rule text, even if it is followed by "Crusher".
    return re.findall(r"'?\w+|[^\w\s]", rule_text, re.UNICODE) + [None]  # Append None to the end of the token list to indicate end of card name.


def read_csv_file(csv_file_path: str) -> None:
    with open(csv_file_path, "r") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            card_name = row["name"]
            tokenized_card_name = tokenize(card_name)
            card_type_string = row["type"]
            try:
                card_type = CardType.from_card_type_string(card_type_string)
            except ValueError as e:
                print(f"Unrecognized card type '{card_type_string}' for card '{card_name}'. Skipping this card. Error: {e}")
                continue
            card = Card(tokenized_name=tokenized_card_name, type=card_type)

            first_word = tokenized_card_name[0]
            assert first_word is not None, f"Card name '{card_name}' is empty after tokenization."
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
                # If we have already matched some card names, we need to check if the current word
                # is a continuation of any of the matched card names.
                for card in match_cards:
                    card_name = card.tokenized_name
                    if match_pos < len(card_name) - 1:
                        if word == card_name[match_pos]:
                            match_pos += 1
                            # Check next word in rule text to see if it is a continuation of the card name.
                            # If it is not, we can annotate the card name and reset the match cards list.
                            next_word = words[pos + 1] if pos < len(words) - 1 else None
                            if card_name[match_pos] is None:
                                # End of card name -> are there any matched cards that are longer?
                                other_matches = [c for c in match_cards if len(c.tokenized_name) > match_pos + 1]  # +1 = None at the end
                                if len(other_matches) > 0:
                                    # Check if the next word is a continuation of any of the longer matched card names.
                                    if any(next_word == c.tokenized_name[match_pos] for c in other_matches):
                                        continue  # The next word is a continuation of a longer card name, so we cannot annotate the current card name yet.
                                else:
                                    # We have matched the full card name, so we can add it to the list of matched cards.
                                    annotated_rule_text.extend([(cast(str, w), card.type) for w in card_name[:-1]])
                                    match_cards = []  # Reset the match cards list for the next card name.
                                    match_pos = 0
                        else:
                            try:
                                match_cards.remove(card)
                            except ValueError:
                                pass  # The card may have already been removed from the match cards list if it was matched earlier in the rule text.
            elif word in card_names:
                match_pos = 1
                match_cards = card_names[word]
                if len(match_cards) == 0:
                    continue
                # Check if the next word is a continuation of any of the matched card names. If it is not, we can annotate the current word and reset the match cards list.
                next_word = words[pos + 1] if pos < len(words) - 1 else None
                has_matched_card = any(c.tokenized_name == [word, None] for c in match_cards)  # Check if there is a card name that consists of only the current word.
                if has_matched_card and not any(next_word == c.tokenized_name[match_pos] for c in match_cards):
                    # The next word is not a continuation of any of the matched card names, so we can annotate the current word and reset the match cards list.
                    annotated_rule_text.append((cast(str, word), match_cards[0].type))  # Annotate with the card type of the first matched card name (there should only be one in this case, since we check for continuations above).
                    match_cards = []
                    match_pos = 0
            elif word is not None:
                annotated_rule_text.append((cast(str, word), None))
        
        annotated_card_rules.append(annotated_rule_text)
    return annotated_card_rules


def print_to_csv(annotated_card_rules: list[list[tuple[str, CardType | None]]], target_file: str) -> None:
    with open(target_file, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=";")
        for annotated_rule in annotated_card_rules:
            is_beginning = True
            for word, card_type in annotated_rule:
                row = [word]
                if card_type is not None:
                    row.append(f"{'B' if is_beginning else 'I'}-{card_type.upper()}")
                    is_beginning = False
                else:
                    is_beginning = True
                    row.append("O")
                csv_writer.writerow(row)


def main(*args) -> None:
    csv_input_file_path = args[0]
    csv_output_file_path = args[1]

    read_csv_file(csv_input_file_path)
    print(f"Read {len(card_names)} unique card names and {len(card_rules)} card rules from the CSV file.")

    annotated_card_rules = annotate_card_rules()
    print_to_csv(annotated_card_rules, csv_output_file_path)

if __name__ == "__main__":
    main(*sys.argv[1:])
