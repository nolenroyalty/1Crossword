#!/usr/bin/env python3

import random
import time
from collections import defaultdict
import subprocess
import json
import math
import os
import itertools
import argparse

EMPTY = " "


class Crossword:
    def __init__(self, size, word_list):
        self.size = size
        self.grid = [[EMPTY for _ in range(size)] for _ in range(size)]
        self.available_words = set(word_list)
        self.placed_word_coords = {}
        self.directions_used_for_coord = defaultdict(
            lambda: {"horizontal": False, "vertical": False}
        )
        self.unusable_positions = set()

    def get(self, row, col):
        try:
            return self.grid[row][col]
        except IndexError:
            return None

    def is_empty(self, row, col, on_oob=True):
        val = self.get(row, col)
        if val is None:
            return on_oob
        return val == EMPTY

    def abuts_existing_word(self, word, row, col, direction):
        if direction == "horizontal":
            left = col - 1
            right = col + len(word)
            abuts_left = not self.is_empty(row, left)
            abuts_right = not self.is_empty(row, right)
            return abuts_left or abuts_right
        if direction == "vertical":
            top = row - 1
            bot = row + len(word)
            top_full = not self.is_empty(top, col)
            bot_full = not self.is_empty(bot, col)
            return top_full or bot_full

    def coords_for_word(self, word, row, col, direction):
        if direction == "horizontal":
            return ((char, row, col + i) for i, char in enumerate(word))
        elif direction == "vertical":
            return ((char, row + i, col) for i, char in enumerate(word))

    def goes_oob(self, word, row, col, direction):
        for _char, row, col in self.coords_for_word(word, row, col, direction):
            if row < 0 or row >= self.size:
                return True
            if col < 0 or col >= self.size:
                return True
        return False

    def overlaps_existing_word(self, word, row, col, direction):
        # if we're placing a word horizontally, it can't overlap with any
        # other words also placed horizontally!
        coords = self.coords_for_word(word, row, col, direction)
        for char, row_, col_ in coords:
            used = self.directions_used_for_coord[(row_, col_)][direction]
            if used:
                return True
            char_at_coords = self.grid[row_][col_]
            if char_at_coords != " " and char_at_coords != char:
                return True
        return False

    def incidentally_formed_words(self, word, row, col, direction):
        def formed_word(row, col, center_char, direction):
            if direction == "vertical":
                top = row
                bot = row
                while not self.is_empty(top - 1, col):
                    top -= 1
                while not self.is_empty(bot + 1, col):
                    bot += 1

                top_word = "".join(self.get(i, col) for i in range(top, row))
                bot_word = "".join(self.get(i, col) for i in range(row + 1, bot + 1))
                word = top_word + center_char + bot_word
                coords = (top, col, "vertical")
                return (word, coords)
            elif direction == "horizontal":
                left = col
                right = col
                while not self.is_empty(row, left - 1):
                    left -= 1
                while not self.is_empty(row, right + 1):
                    right += 1

                left_word = "".join(self.get(row, i) for i in range(left, col))
                right_word = "".join(
                    self.get(row, i) for i in range(col + 1, right + 1)
                )
                word = left_word + center_char + right_word
                coords = (row, left, "horizontal")
                return (word, coords)

        # if we place a word horizontally, it can't have anything to its left or right
        # but it can have words above and below it! this prevents us from creating new
        # vertical words that are invalid
        #
        # a smarter solution would allow invalid words in the short term and try to make
        # them "valid" down the line, but let's not worry about that for now...
        if direction == "horizontal":
            top = row - 1
            bot = row + 1
            words = []
            for offset, char in enumerate(word):
                offset_col = offset + col
                if not self.is_empty(top, offset_col) or not self.is_empty(
                    bot, offset_col
                ):
                    # horizontal words form vertical crosses
                    formed = formed_word(row, offset_col, char, "vertical")
                    words.append(formed)
            return words
        elif direction == "vertical":
            left = col - 1
            right = col + 1
            words = []
            for offset, char in enumerate(word):
                offset_row = offset + row
                if not self.is_empty(offset_row, left) or not self.is_empty(
                    offset_row, right
                ):
                    formed = formed_word(offset_row, col, char, "horizontal")
                    words.append(formed)
            return words

    def can_place_word(self, word, row, col, direction):
        if self.goes_oob(word, row, col, direction):
            return False, []
        if self.abuts_existing_word(word, row, col, direction):
            return False, []
        if self.overlaps_existing_word(word, row, col, direction):
            return False, []

        newly_created_words = []
        incidentally_formed_words = self.incidentally_formed_words(
            word, row, col, direction
        )

        for incidental_word, coords in incidentally_formed_words:
            placed_coords = self.placed_word_coords.get(incidental_word, None)
            if placed_coords is not None and placed_coords != coords:
                return False, []
            if incidental_word == word:
                # INSANE edge case - you created the same word twice in one move!!
                return False, []
            elif placed_coords is None:
                if incidental_word not in self.available_words:
                    return False, []
                newly_created_words.append((incidental_word, coords))

        return True, newly_created_words

    def _place_word_aux(self, word, row, col, direction):
        self.placed_word_coords[word] = (row, col, direction)
        self.available_words.remove(word)
        for char, row, col in self.coords_for_word(word, row, col, direction):
            self.grid[row][col] = char
            self.directions_used_for_coord[(row, col)][direction] = True

    def place_word(self, word, row, col, direction, newly_created_words):
        self._place_word_aux(word, row, col, direction)

        for other, coords in newly_created_words:
            row_, col_, direction_ = coords
            self._place_word_aux(other, row_, col_, direction)

    def get_letter_locations(self):
        letter_locations = defaultdict(list)
        for row_idx, row in enumerate(self.grid):
            for col_idx, char in enumerate(row):
                if char != EMPTY:
                    letter_locations[char].append((row_idx, col_idx))
        return letter_locations

    def __repr__(self):
        return "\n".join("".join(row) for row in self.grid)

    def shrinkwrap_grid(self):
        min_row = self.size
        max_row = 0
        min_col = self.size
        max_col = 0
        for row_idx, row in enumerate(self.grid):
            for col_idx, val in enumerate(row):
                if val != EMPTY:
                    min_row = min(min_row, row_idx)
                    max_row = max(max_row, row_idx)
                    min_col = min(min_col, col_idx)
                    max_col = max(max_col, col_idx)

        new_grid = []
        for row_idx in range(min_row, max_row + 1):
            new_grid.append(self.grid[row_idx][min_col : max_col + 1])

        for word, coords in self.placed_word_coords.items():
            row, col, direction = coords
            self.placed_word_coords[word] = (row - min_row, col - min_col, direction)

        self.grid = new_grid

    def score(self):
        if not self.grid:
            return 0
        # TODO: calculate total based on min / max row and col
        filled = sum(1 for row in self.grid for cell in row if cell != EMPTY)
        rows = len(self.grid)
        cols = len(self.grid[0])
        area = rows * cols
        squareness = min(rows, cols) / max(rows, cols)
        squareness_penalty = math.pow(squareness, 1 / 3)
        filled_penalty = filled / area
        total_count = math.pow(filled, 2)
        return total_count * filled_penalty * squareness_penalty

    def find_maybe_placeable_words(self):
        letter_locations = self.get_letter_locations()
        candidates = list(self.available_words)
        random.shuffle(candidates)
        for candidate in candidates:
            for idx, char in enumerate(candidate):
                for row, col in letter_locations.get(char, []):
                    directions = ["horizontal", "vertical"]
                    random.shuffle(directions)
                    for direction in directions:
                        if (candidate, row, col, direction) in self.unusable_positions:
                            continue
                        target_row = row
                        target_col = col
                        if direction == "vertical":
                            target_row -= idx
                        if direction == "horizontal":
                            target_col -= idx

                        yield (candidate, target_row, target_col, direction)

    def try_to_place_one_word(self):
        for candidate, row, col, direction in self.find_maybe_placeable_words():
            can_place, newly_created = self.can_place_word(
                candidate, row, col, direction
            )
            if can_place:
                self.place_word(candidate, row, col, direction, newly_created)
                return True
            else:
                self.unusable_positions.add((candidate, row, col, direction))
        return False

    def generate_crossword(self, starting_word):
        pos = self.size // 2
        self.placement_count = 0
        can_place_start, _ = self.can_place_word(starting_word, pos, pos, "horizontal")

        if not can_place_start:
            return None

        self.place_word(starting_word, pos, pos, "horizontal", [])
        self.placement_count += 1

        while True:
            placed = self.try_to_place_one_word()
            if placed:
                self.placement_count += 1
            else:
                break

        self.shrinkwrap_grid()
        return self.placement_count


class QuitException(Exception):
    pass


FILLABLE = "_"


class CrosswordUi:
    def __init__(self, crossword, clues=None):
        self.crossword = crossword
        if clues is None:
            self.clues_for_word = lambda word: word
        else:
            self.clues_for_word = lambda word: clues[word]
        self.initialize_fillable_grid()
        self.initialize_clues()

    def all_squares_filled(self):
        return all(cell != FILLABLE for row in self.fillable_grid for cell in row)

    def is_correct(self):
        # I don't remember if just using equality works here, whatever
        for row_idx, row in enumerate(self.fillable_grid):
            for col_idx, cell in enumerate(row):
                cell = cell.upper()
                from_cw = self.crossword.get(row_idx, col_idx).upper()
                if from_cw != cell:
                    return False
        return True

    def initialize_fillable_grid(self):
        fillable = []
        for row in self.crossword.grid:
            new_row = []
            for val in row:
                if val == EMPTY:
                    new_row.append(EMPTY)
                else:
                    new_row.append(FILLABLE)
            fillable.append(new_row)
        self.fillable_grid = fillable

    def initialize_clues(self):
        flat_words = [
            (word, *coords)
            for word, coords in self.crossword.placed_word_coords.items()
        ]
        sorted_words = sorted(flat_words, key=lambda x: (x[1], x[2]))
        acrosses = []
        downs = []
        clue_to_coords = {}
        count = 0
        prev_row = None
        prev_col = None

        for word, row, col, direction in sorted_words:
            if row != prev_row or col != prev_col:
                count += 1
                prev_row = row
                prev_col = col
            clue = self.clues_for_word(word)
            clue_name = ""
            if direction == "horizontal":
                clue_name = f"{count}A"
                acrosses.append((clue_name, clue))
            else:
                clue_name = f"{count}D"
                downs.append((clue_name, clue))
            clue_to_coords[clue_name] = (row, col, direction, len(word), word)

        self.acrosses = acrosses
        self.downs = downs
        self.clue_to_coords = clue_to_coords

    def render_grid(self, indices_to_highlight=None):
        rows = []
        for row_idx, row in enumerate(self.fillable_grid):
            s = ["|"]
            for col_idx, cell in enumerate(row):
                if indices_to_highlight and (row_idx, col_idx) in indices_to_highlight:
                    if cell == FILLABLE:
                        s.append("*")
                    else:
                        s.append(cell)
                else:
                    s.append(cell)
            s.append("|")
            rows.append(" ".join(s))
        return rows

    def render_clues(self):
        acrosses = [f"{clue_name}: {clue}" for clue_name, clue in self.acrosses]
        max_across_len = max(len(a) for a in acrosses) + 2
        downs = []
        for idx, down in enumerate(self.downs):
            clue_name, clue = down
            s = f"{clue_name}: {clue}"
            padding = 0
            if idx < len(acrosses):
                padding = max_across_len - len(acrosses[idx])
            else:
                padding = max_across_len
            downs.append(" " * padding + s)

        combined = []
        for across, down in itertools.zip_longest(acrosses, downs, fillvalue=""):
            combined.append(f"{across} {down}")
        return combined

    def header(self):
        # from https://patorjk.com/software/taag/#p=display&f=ANSI%20Shadow&t=1Crossword
        return r""" ██╗ ██████╗██████╗  ██████╗ ███████╗███████╗██╗    ██╗ ██████╗ ██████╗ ██████╗ 
███║██╔════╝██╔══██╗██╔═══██╗██╔════╝██╔════╝██║    ██║██╔═══██╗██╔══██╗██╔══██╗
╚██║██║     ██████╔╝██║   ██║███████╗███████╗██║ █╗ ██║██║   ██║██████╔╝██║  ██║
 ██║██║     ██╔══██╗██║   ██║╚════██║╚════██║██║███╗██║██║   ██║██╔══██╗██║  ██║
 ██║╚██████╗██║  ██║╚██████╔╝███████║███████║╚███╔███╔╝╚██████╔╝██║  ██║██████╔╝
 ╚═╝ ╚═════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚══════╝ ╚══╝╚══╝  ╚═════╝ ╚═╝  ╚═╝╚═════╝ """.split(
            "\n"
        )

    def print_text_centered_in_terminal(self, lines):
        terminal_width = os.get_terminal_size().columns
        max_line_len = max(len(line) for line in lines)
        padding = (terminal_width - max_line_len) // 2
        padded = [" " * padding + line for line in lines]
        print("\n".join(padded))

    def input_centered(self, prompt_lines):
        terminal_width = os.get_terminal_size().columns
        max_line_len = max(len(line) for line in prompt_lines)
        padding = (terminal_width - max_line_len) // 2
        padded = [" " * padding + line for line in prompt_lines]
        return input("\n".join(padded)).strip().upper()

    def render(self, highlight_clue=None):
        self.clear()
        print("")
        self.print_text_centered_in_terminal(self.header())
        print("")
        indices_to_highlight = None
        if highlight_clue:
            indices_to_highlight = self.all_coords_for_clue(highlight_clue)
        printable_grid = self.render_grid(indices_to_highlight)
        max_grid_len = max(len(line) for line in printable_grid)
        grid_padding = "=" * max_grid_len
        printable_grid = [grid_padding, *printable_grid, grid_padding]
        printable_clues = self.render_clues()
        self.print_text_centered_in_terminal(printable_grid)
        print("")
        self.print_text_centered_in_terminal(printable_clues)
        print("")

    def prompt_next_clue(self):
        prompt = ["q to quit", "Enter a clue number (like 1A) to fill in: "]
        while True:
            clue_name = self.input_centered(prompt)
            if clue_name == "Q":
                raise QuitException("User quit")
            if clue_name not in self.clue_to_coords:
                self.render()
                self.print_text_centered_in_terminal(
                    [f"Invalid clue number {clue_name}, try again", ""]
                )
                continue
            return clue_name

    def prompt_answer(self, clue_name):
        answer_len = self.clue_to_coords[clue_name][3]
        answer = self.clue_to_coords[clue_name][4]
        clue = self.clues_for_word(answer)
        prompt_lines = [
            f"q to quit",
            f"Enter the answer for {clue_name} ({answer_len} letters / {clue}): ",
        ]
        while True:
            answer = self.input_centered(prompt_lines)
            if answer == "Q":
                raise QuitException("User quit")
            if len(answer) != answer_len:
                self.render()
                self.print_text_centered_in_terminal(
                    [f"Answer must be {answer_len} letters long, try again", ""]
                )
                continue
            return answer

    def all_coords_for_clue(self, clue_name):
        coords = []
        row, col, direction, answer_len, _ = self.clue_to_coords[clue_name]
        for idx in range(answer_len):
            row_ = row + idx * (1 if direction == "vertical" else 0)
            col_ = col + idx * (1 if direction == "horizontal" else 0)
            coords.append((row_, col_))
        return coords

    def fill_in_answer(self, clue_name, answer):
        for idx, (row, col) in enumerate(self.all_coords_for_clue(clue_name)):
            self.fillable_grid[row][col] = answer[idx]

    def _run_game(self):
        while True:
            self.render()
            if self.all_squares_filled():
                if self.is_correct():
                    self.print_text_centered_in_terminal(["You win!"])
                    return
                else:
                    self.print_text_centered_in_terminal(
                        [
                            "That's not quite right!",
                            "The grid is filled in, but you missed something.",
                        ]
                    )
            clue_name = self.prompt_next_clue()
            self.render(highlight_clue=clue_name)
            answer = self.prompt_answer(clue_name)
            self.fill_in_answer(clue_name, answer)

    def run_game(self):
        try:
            self._run_game()
        except QuitException:
            print("Thanks for playing!")

    def clear(self):
        os.system("cls" if os.name == "nt" else "clear")


def generate_many_crosswords(max_grid_size, word_list, max_time):
    start_time = time.time()
    best_grid = None
    best_score = 0

    while time.time() - start_time < max_time:
        g = Crossword(max_grid_size, word_list)
        word = random.choice(word_list)
        g.generate_crossword(word)
        score = g.score()
        if score > best_score:
            best_grid = g
            best_score = score

    print(f"Best grid score: {best_score}")
    print(best_grid)
    return best_grid


def gather_from_1password(vault, count=150):
    def inner():
        list_proc = subprocess.run(
            [
                "op",
                "item",
                "list",
                "--categories",
                "Login",
                "--vault",
                vault,
                "--format",
                "json",
            ],
            capture_output=True,
        )
        if list_proc.returncode != 0:
            raise Exception(
                f"Failed to list items from 1Password: {list_proc.stderr.strip().decode('utf-8')}"
            )
        js = json.loads(list_proc.stdout.strip().decode("utf-8"))
        if len(js) > count:
            js = list(random.sample(js, count))
        word_list = []
        print(f"Retrieving {len(js)} passwords from 1Password...")
        clues = {}
        for d in js:
            try:
                id = d["id"]
                pw_proc = subprocess.run(
                    [
                        "op",
                        "item",
                        "get",
                        id,
                        "--fields",
                        "username,password",
                        "--reveal",
                    ],
                    capture_output=True,
                )
                user_pw = pw_proc.stdout.strip().decode("utf-8")
                user, pw = user_pw.split(",")
            except:
                continue
            word_list.append(pw)
            title = d.get("title", "")
            clues[pw] = f"{title} ({user})"

        return word_list, clues

    try:
        return inner()
    except Exception as e:
        raise Exception(f"Failed to gather from 1Password: {e}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finally, a fun use for your passwords. Generate and solve a crossword puzzle using 1Password vault data."
    )

    parser.add_argument(
        "--vault",
        type=str,
        default="Private",
        help="The 1Password vault to connect to (default: 'Private').",
    )

    parser.add_argument(
        "--num-passwords",
        type=int,
        default=150,
        help="The number of passwords to use for crossword generation (default: 150).",
    )

    parser.add_argument(
        "--max-time",
        type=int,
        default=5,
        help="The maximum time (in seconds) to spend generating a crossword (default: 5 seconds).",
    )

    parser.add_argument(
        "--grid-size",
        type=int,
        default=20,
        help="The maximum size of the crossword grid (default: 20x20).",
    )

    parser.add_argument(
        "--i-am-a-coward-and-a-baby-invoke-with-test-word-list",
        action="store_true",
        help="For cowards. Use a short test word list instead of connecting to 1Password.",
    )

    return parser.parse_args()


TEST_WORD_LIST_FOR_LITTLE_BABY_COWARDS = [
    "CAT",
    "RAT",
    "BAT",
    "RAN",
    "CAN",
    "MAN",
    "LARA",
    "LEGO",
    "ASK",
    "MASK",
    "FOO",
    "FAWL",
]


def main():
    args = parse_args()
    if args.i_am_a_coward_and_a_baby_invoke_with_test_word_list:
        print("Using test word list for little baby cowards...")
        word_list = TEST_WORD_LIST_FOR_LITTLE_BABY_COWARDS
        clues = None
    else:
        print("Gathering passwords from 1Password (this may take a while)...")
        word_list, clues = gather_from_1password(
            vault=args.vault, count=args.num_passwords
        )

    print("Generating crossword...")
    max_grid_size = args.grid_size
    max_time = args.max_time
    g = generate_many_crosswords(max_grid_size, word_list, max_time)
    ui = CrosswordUi(g, clues=clues)
    ui.run_game()


if __name__ == "__main__":
    main()
