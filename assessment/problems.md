# Assessment Problems

Carefully follow the [assessment instructions](assessment.md) to make sure you solve the following problems in the correct way.  

## Task 1: Bigram model of punctuation-separated tokens

Download five different free English works from [Project Gutenberg](https://www.gutenberg.org/), in `Plain Text UTF8` format.

Clean the text by:

* Removing preamble and postamble.
* Converting all letters to uppercase.
* Replacing any non-alphabetic character (except spaces and full stops) with a space.

Then tokenize the text into *words and full stops* (treat `.` as a token).

Build a **bigram model**: count how often each **pair of adjacent tokens** appears.
Store the result using a data structure of your choice.
Explain your design.

## Task 2: Probabilistic token stream generation

Using your bigram model from Task 1, generate a token stream of **3,000 tokens**, starting from the token `THE`.

At each step:

* Look up all possible next tokens for the current one.
* Choose the next token randomly based on observed frequencies.
* Append the chosen token and repeat.

Join the final tokens into a string, inserting a space between tokens, and saving it in a file called `generated.txt`.

## Task 3: Word length distribution

Use the word list from `words.txt` to check which words in your generated string from Task 2 are valid English words.

Then compute the distribution of **word lengths** (excluding full stops) and display it using a bar chart (you may use Python for this).

Also report:

* Average word length.
* Proportion of valid English words.

## Task 4: Export your model as CSV

Convert your bigram model to CSV format, with three columns: `token_1`, `token_2`, and `count`.

Save it as `bigrams.csv` in your repository.

Here are revised **Tasks 5 and 6**, now fully in Python and designed to follow on from the first four tasks:

## Task 5: Visualise transition probabilities

Using your bigram model from **Task 1**, compute the transition probabilities between tokens.

Then:

* Select the 10 most frequent starting tokens.
* For each, plot a bar chart showing the probability distribution of the most common next tokens.

Use **Matplotlib** or **Seaborn** to create the charts and save them in a folder called `figures/` with descriptive filenames like `transition_THE.png`.

## Task 6: Interactive generation in Python

Write a Python script named `generate_text.py` that:

* Loads your `bigrams.csv` file.
* Asks the user to input a starting token (e.g. `THE`) and desired number of tokens (e.g. `100`).
* Generates and prints a token stream using your bigram model.
* Writes the generated string to a file called `interactive_output.txt`.

Use `argparse` to allow optional command-line arguments for starting token and length.
