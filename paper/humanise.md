# Humanise Guide: Writing That Sounds Like a Person

Practical habits for making academic and technical writing read as genuinely yours: varied, specific, and logically connected. These are good writing practices in general, whether a draft started from your own notes or from a tool.

---

## 1. Vary sentence length on purpose

Uniform sentences (all medium length, all the same shape) create a flat rhythm that readers and detectors both notice.

- Place a very short, sharp sentence directly after a long, winding one. The contrast gives emphasis.
- Mix sentence openers: a clause, a subject, a prepositional phrase, a question.
- Read the paragraph aloud. If it sounds like a metronome, break the pattern.

**Example**

> The attention-based variants consistently outperformed the plain BiLSTM baseline across every random seed we tried, though the margin narrowed considerably once the embedding dimension was increased past 300 and the training set was held fixed. That gap matters.

The long sentence carries the nuance; the four-word sentence lands the point.

---

## 2. Let paragraphs differ in size and purpose

- Some paragraphs should dwell deeply on a single point: one claim, one piece of evidence, one consequence.
- Others can stretch out with detail, qualification, and nuance when the idea needs room.
- A short paragraph of one or two sentences can mark a turn in the argument.
- Avoid the default pattern of every paragraph being four to five sentences of the same weight.

---

## 3. Replace stock transitions with real bridges

Words like *Furthermore*, *Moreover*, *Crucially*, *In conclusion*, and *It is important to note that* are not banned, but repeated use in a text is a strong marker of formulaic writing. They connect nothing specific.

**Instead, bridge using the actual content of the previous point.**

| Generic | Content-based bridge |
|---|---|
| Furthermore, the model improves accuracy. | Building directly on these structural shifts, the model's accuracy also improves on the minority classes. |
| Moreover, the dataset is balanced. | Because the dataset is balanced across its four categories, per-class variance stays small. |
| Crucially, attention helps. | That improvement traces back to where the attention weights concentrate. |
| In conclusion, our results show... | Taken together, the eight-model comparison points to one pattern: ... |
| It is important to note that results vary. | Results vary, and the variation itself is informative. |

**Rule of thumb:** if you can delete the transition and the logic still holds, the transition was decoration. If the logic breaks, rewrite the bridge so it names the idea being carried forward.

---

## 4. Anchor your text in specifics

Generic prose is statistically predictable. Specific prose is not, and it is also more useful to a reader.

- **Name the method exactly:** "additive (Bahdanau-style) attention" instead of "an attention mechanism."
- **Name the data and variables:** "AG News, four classes, 120,000 training samples" instead of "a news dataset."
- **Name the people and works:** cite the actual authors and years for the idea you are building on.
- **Give real numbers:** exact metrics, hyperparameters, thresholds, sample sizes, and run counts.
- **State the setup:** library versions, hardware, random seeds, preprocessing steps.

**Important:** every specific must be true and traceable to your own work or a source you have actually read. Never invent authors, figures, or results to sound specific. Fabricated detail is worse than vague prose.

---

## 5. Other habits that make writing sound like you

- **Take a position.** Say what you think the result means, and why. Hedge only where the evidence is genuinely uncertain.
- **Use concrete verbs.** "The loss plateaued after epoch 12" beats "The model demonstrated a notable improvement in convergence behaviour."
- **Cut filler openers and closers.** Skip "In today's rapidly evolving landscape" and "This paper delves into."
- **Avoid stacked buzzwords** such as *robust, comprehensive, pivotal, seamless, leverage, landscape, tapestry*. Use them only when they are the precise word.
- **Break the triple.** Lists of exactly three adjectives or three parallel clauses everywhere feel mechanical. Use two, or four, or one.
- **Include your own decisions and dead ends.** "We first tried X, which failed because Y" is something only you can write.
- **Mix registers carefully.** Technical precision with an occasional plain-spoken sentence reads as human. Slang does not belong in a formal paper.

---

## 6. Suggested additions

### 6.1 Write the first draft from your own notes, not from a blank prompt
The most reliable way to sound like yourself is to start from your own raw material: experiment logs, lab notes, rough bullet points, your own explanation said aloud. Polishing your own thinking keeps your voice. Polishing generic text rarely does.

### 6.2 Explain the "why" behind your choices
Predictable prose states what was done. Distinctive prose says why. "We used 5 random seeds because variance on the smaller splits was high" is more informative, and more yours, than "We used 5 random seeds."

### 6.3 Include honest limitations and surprises
Real research has awkward results. A sentence like "BiLSTM with multiplicative attention underperformed our expectations, and we do not have a confirmed explanation" signals genuine engagement. Stock limitation paragraphs ("future work could explore larger datasets") signal the opposite.

### 6.4 Use first person where your field allows it
"We observed," "I chose," and "our run" are fine in many venues and make ownership clear. Check the target venue's style guide first; some prefer passive voice or "the authors."

### 6.5 Vary how you introduce evidence
Do not open every claim with "Table 2 shows." Try leading with the finding, then pointing to the table, or placing the number inside the sentence. Vary the verbs: *shows, reveals, confirms, contradicts, hints at*, each used only when accurate.

### 6.6 Watch repeated sentence skeletons
Look for runs like "The model achieves... The model demonstrates... The model exhibits..." Combine, reorder, or cut. A quick check is to copy the first four words of each sentence into a list and look for repeats.

### 6.7 Ground abstract claims with one concrete example
After a general statement, add one small, real instance: a misclassified sample, a specific class confusion, a particular training curve. Concrete examples are hard to fake and easy to trust.

### 6.8 Do a "voice test" before submitting
Read one paragraph to a friend or record yourself explaining it. Wherever your spoken version differs sharply from the written one, the written version has probably drifted into generic phrasing. Rewrite it closer to how you explained it, keeping the academic precision.

### 6.9 Keep a personal "phrase blocklist"
Collect the clichés you tend to overuse (for example *delve into, plays a crucial role, in the realm of, a testament to*) and search for them in every draft. Your list will be more useful than any generic one.

### 6.10 Don't over-correct
Forced quirks, random fragments, or deliberately awkward sentences read as unnatural too. The goal is clarity with natural rhythm, not noise. If a change makes the sentence harder to understand, undo it.

---

## 7. Revision pass checklist

- [ ] Read aloud: is there a mix of long and very short sentences?
- [ ] Do paragraphs vary in length and depth?
- [ ] Search for *Furthermore, Moreover, Crucially, In conclusion, It is important to note*; keep only those that do real work
- [ ] Each transition names the idea it is building on
- [ ] Every vague noun ("the model," "the data," "the method") replaced with its specific name where it matters
- [ ] All numbers, authors, and method names verified against your notes and sources
- [ ] No sentence that could appear unchanged in any other paper on the topic

---

## 8. Be honest about how the draft was made

Rewriting for voice and specificity is about quality. It is not a substitute for original thinking, proper citation, or following your institution's rules. If you used an AI tool in drafting, follow the disclosure policy of your university or venue, and make sure the ideas, analysis, and verification are your own. Pair this guide with `plagiarism_guide.md`.
