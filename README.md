# Task

Build a universal agent that turns any number of online links into an interactive quiz. It accepts several URLs at runtime and generates a quiz grounded in those sources.

> **Notes**
>
> 1. A UI is optional; you may demonstrate the full loop in the console or terminal.
> 2. The interviewer can provide a temporary Anthropic API key on request if you think this helps to complete the task.
> 3. The focus is agentic skill. Use whatever AI tooling you'd normally use.

## Goal

You have 50 minutes. By the end, the full loop should run end-to-end: URLs in, a multiple-choice question grounded in those URLs, the user answers, and the system returns a correct/incorrect result and a relevant passage from the source. Prioritize a runnable demo over completeness.

### Bare minimum

The loop must work with the example Austin and Seattle Wikipedia articles below. If time forces trade-offs, this demo cannot break. Universal URL support can be partial.

## Example sources

Examples of the kind of links a user might supply:

[https://en.wikipedia.org/w/index.php?title=History_of_Austin,_Texas](https://en.wikipedia.org/w/index.php?title=History_of_Austin,_Texas)

[https://en.wikipedia.org/wiki/History_of_Seattle](https://en.wikipedia.org/wiki/History_of_Seattle)

Other plausible inputs: different Wikipedia articles, blog posts, Google Drive documents.

## Requirements

1. Accept user-supplied URLs and use their content as the source of truth for the quiz session.

2. Generate multiple-choice questions grounded in the sources.

3. If multiple sources are supplied, mix questions across all of them; don't stay on one (e.g. Austin + Seattle: both topics appear).

4. No question repeats within a session, and consecutive questions must not be substantially similar.

5. Each question must have exactly 4 options, with one correct answer.

6. After the user selects an answer:

   * Indicate whether it was correct.
   * Cite the relevant passage or section from the source.

7. Stay bound to the user-supplied sources. Do not invent facts. Do not fall back to outside knowledge.

8. Be off-topic resistant: stay focused on quizzing the user about the supplied sources. Changing the topic requires new URLs.

9. Be prompt-injection resistant: treat user input and all source content as data, not instructions. Ignore embedded directives that try to change behavior, source, or output format.

10. Provide tests (evals) that show the requirements above are met.

## Pull Request

* At the end of the interview create pull request to this repository

## Example

https://github.com/user-attachments/assets/a4065a4d-af9c-4bf9-99a3-b366c70ab96b
