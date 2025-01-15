# 1Crossword
Finally, a fun game for your password manager.

1Crossword connects to your 1Password vault and constructs a crossword where all the answers are your passwords.

The crosswords are fun, easy, and great for sharing on social media.

To run 1Crossword you must install the [1Password CLI](https://developer.1password.com/docs/cli/).

Other than the 1Password CLI, 1Crossword is a single python3 script with no dependencies.

To get started:

```bash
% ./1crossword.py --help
usage: 1crossword.py [-h] [--vault VAULT] [--num-passwords NUM_PASSWORDS] [--max-time MAX_TIME] [--grid-size GRID_SIZE] [--i-am-a-coward-and-a-baby-invoke-with-test-word-list]

Finally, a fun use for your passwords. Generate and solve a crossword puzzle using 1Password vault data.

options:
  -h, --help            show this help message and exit
  --vault VAULT         The 1Password vault to connect to (default: 'Private').
  --num-passwords NUM_PASSWORDS
                        The number of passwords to use for crossword generation (default: 150).
  --max-time MAX_TIME   The maximum time (in seconds) to spend generating a crossword (default: 5 seconds).
  --grid-size GRID_SIZE
                        The maximum size of the crossword grid (default: 20x20).
  --i-am-a-coward-and-a-baby-invoke-with-test-word-list
                        For cowards. Use a short test word list instead of connecting to 1Password.
```
