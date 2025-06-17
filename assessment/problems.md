# Assessment Problems

Follow the [assessment instructions](assessment.md) and [guidance](guidance.ipynb) closely to ensure you solve the following problems correctly.

## Problem 1: Text Cleaning

Create a `data` folder in your repository.
Download any 10 free English texts from [Project Gutenberg](https://www.gutenberg.org/), choosing works that are as large as possible and in `Plain Text UTF-8` format.
Save each file in a `texts` subfolder of your `data` folder using clear, descriptive filenames.

Then, in your notebook, process the texts as follows:

* Remove the Project Gutenberg header and footer from each text.
* Convert all letters to uppercase.
* Replace all non-alphabetic characters, except spaces and full stops, with a space.
* Join all cleaned texts into a single string.
* Save this combined string to a file named `cleaned.txt` in the `data` folder.

Clearly explain your work using MarkDown and code.

## Problem 2: 4-gram Model

Use a Python dictionary to build a 4-gram model of your cleaned text.
Count how often each sequence of four adjacent characters appears, storing each 4-character string as a key and its frequency as the value.

Save the dictionary to a file named `fourgram.pickle` in the `data` folder using the `pickle` module from the standard library:  
[https://docs.python.org/3/library/pickle.html](https://docs.python.org/3/library/pickle.html)

## Problem 3: Probabilistic Generation

Using your 4-gram model from Problem 2, generate a string of 10,000 characters starting with `THE`.

At each step, use the last three characters to look up possible next characters in the model.
Choose one randomly, with probabilities based on how often each option appeared in the cleaned text.
Append the chosen character and repeat.

If no next character is available, handle the situation gracefully - for example, by restarting with a known sequence.

Save the final string to `data/generated.txt`.

## Problem 4: Word Length Distribution

Split your generated text into words using spaces and full stops as separators.
Count how many words there are of each length (e.g. length 1, length 2, and so on).

Create a bar chart showing these counts using the `bar()` function from `matplotlib.pyplot`:  
[https://matplotlib.org/stable/api/\_as\_gen/matplotlib.pyplot.bar.html](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.bar.html)

## Problem 5: Valid English Words

Save the following list of English words in your `data` folder:  
[https://raw.githubusercontent.com/ianmcloughlin/datasets/refs/heads/main/wordlist.txt](https://raw.githubusercontent.com/ianmcloughlin/datasets/refs/heads/main/wordlist.txt)

Count how many words are of each length and create a bar chart using `matplotlib.pyplot.bar()`.
Compare this chart to the one from Problem 4. Comment on any differences you notice and suggest possible reasons for them.

***

**End**
