# Plagiarism & Similarity Guide (Turnitin-Aware)

A working checklist for drafting papers that pass a Turnitin similarity check *and* are genuinely original and properly credited.

---

## 1. What Turnitin actually does (from Turnitin's own documentation)

- **It measures similarity, not plagiarism.** The similarity score is the percentage of your paper that matches Turnitin's databases. Turnitin states it is not an assessment of whether the paper contains plagiarized material, and that it does not check for plagiarism itself.
- **Its database includes** internet content (including archived pages), periodicals, journals, publications, and past student submissions.
- **Score colours (classic reports):** Blue = 0% · Green = 1-24% · Yellow = 25-49% · Orange = 50-74% · Red = 75-100%.
- **There is no universal "safe" number.** Turnitin says to check your syllabus, assignment instructions, or institutional policy for the expected score. Your own target of **under 5%** is a stricter personal goal, not a Turnitin rule. Confirm what your university or the conference/journal requires.
- **What commonly creates "similarity" without being plagiarism:**
  - Common phrases and standard academic wording
  - Proper nouns, organisation names, titles, legal terms
  - Citation formats and reference lists
  - Properly quoted and cited text
  - Your own earlier drafts or earlier submissions of the same paper
- **A low score is not proof of originality, and a high score is not proof of plagiarism.** Turnitin notes a low score can still hide problems (for example uncited paraphrase, translated text, or diagrams, which it cannot check well).

## 2. How the report filters work

| Setting | What it does |
|---|---|
| Exclude quotes | Ignores text inside quotation marks or block quotes |
| Exclude bibliography | Ignores the reference list |
| Exclude small matches | Hides matches under a word threshold (new report default is 8 words; the lowest you can set is 8) |
| Exclude small sources | Classic report only; hides sources under a word count or percentage |
| Generate new report | Refreshes the score after changes. Resubmissions can take about 24 hours to produce a fresh report |

**Important:** exclusions you apply as a student are temporary and do not change the official score your instructor sees. Only the instructor's settings count. If you think exclusions should be applied (for example a quote-heavy paper), ask your instructor.

**Practical consequence:** because short matches below the threshold are filtered and quotes/bibliography may be excluded, the real risk lies in *longer matching strings in your body text*. That is where to focus.

---

## 3. Drafting habits

### 3.1 Write from understanding, not from the page
1. Read the source fully.
2. **Close the tab / PDF.**
3. Write the idea from memory, in your own voice and sentence structure.
4. Re-open the source and check that the **meaning** is still accurate. Fix meaning errors, but do not copy wording back in.

### 3.2 Never paste source text into your working draft
- Do not copy-paste from a website or PDF into the draft.
- If you must park a snippet temporarily, wrap it in obvious markers with the source URL beside it, for example:
  `【【 PASTED, REWRITE OR DELETE | https://source-url 】】`
  and use a distinct font colour. Search the draft for `【【` before every submission.
- Take notes as short bullet fragments, not full copied sentences.

### 3.3 Paraphrase properly
- Swapping three words while keeping the same sentence skeleton still triggers high similarity, and is still plagiarism.
- Real paraphrase changes the **structure**, not just the vocabulary:
  - Switch active ↔ passive voice
  - Split one long sentence into two, or merge two short ones
  - Reorder the idea (result first, then method, or the reverse)
  - Lead with a different subject
  - Combine the idea with your own analysis or comparison
- **A paraphrase still needs a citation.** Rewording removes the text match, but the *idea* still belongs to the source.

### 3.4 Use direct quotes sparingly
- Quote only unique, irreplaceable definitions or high-impact statements.
- Always use quotation marks **and** an immediate in-text citation (with page number where relevant).
- Keep quotes short. A paper that is mostly quotes shows little original analysis even when every quote is cited.

### 3.5 Build citations as you write
- Add the in-text citation in the same sentence you write the claim. Do not leave "[cite later]" gaps.
- Use a reference manager such as **Zotero** to store URL, authors, page numbers, publication date, and DOI as soon as you open a source.
- Before submission, cross-check: every in-text citation has a reference entry, and every reference entry is cited in the text.

---

## 4. Fixing sentences flagged in a similarity report

For each highlighted passage, ask in order:

1. **Is it a quote I cited correctly?** If yes, leave it, or trim it if it is longer than needed.
2. **Is it a common phrase, term, name, or title?** Usually leave it. If the match is long, break the sentence apart so the shared string is shorter.
3. **Is it a technical explanation?** Dissolve the string match:
   - Change voice (active/passive)
   - Split or merge sentences
   - Re-express the concept in your own framing
   - Cite the source if the idea is not yours
4. **Is it matching my own earlier draft or submission?** Check with your instructor about repository settings; do not "fix" it by distorting your text.
5. After edits, regenerate the report (allow time for the new report) and re-check.

---

## 5. Pre-submission checklist

- [ ] No pasted text remains (search for your paste markers and for colour-coded text)
- [ ] Every paraphrased idea has an in-text citation
- [ ] Every direct quote has quotation marks and a citation
- [ ] Reference list complete and consistent with in-text citations
- [ ] Sentence structures differ from the sources, not only the words
- [ ] Ran a draft check (if the institution allows) before the deadline
- [ ] Reviewed each highlighted match instead of chasing the percentage
- [ ] Confirmed the institution's acceptable similarity threshold

---

## 6. Additional essential points

### 6.1 Self-plagiarism and the repository
- Turnitin compares against past submissions, including your own. Resubmitting earlier work, or a draft already stored in the repository, can show as a high match.
- Reusing your own earlier text (a previous report, a thesis chapter, a workshop paper) needs disclosure and citation too. Many venues treat undisclosed reuse as self-plagiarism.
- Ask your instructor whether draft submissions are stored in the repository, and avoid submitting multiple full drafts of the same paper unless allowed.

### 6.2 Patchwriting
- Patchwriting means stitching together lightly edited fragments from several sources. It looks original sentence by sentence but is still copying.
- Fix: outline the argument in your own words first, then fill in the evidence and cite it.

### 6.3 Translated text
- Translating a passage from another language and presenting it as your own is plagiarism, even though a checker may not catch it. Cite the original.

### 6.4 Figures, tables, code, and datasets (important for CS papers)
- Reused figures, tables, or diagrams need a source credit in the caption and permission where the licence requires it. Turnitin cannot check diagrams well, but reviewers can.
- Code, pseudocode, and algorithm descriptions taken from papers, tutorials, or repositories need attribution and must respect the licence.
- Datasets and pretrained models (for example AG News) should be cited with the original paper or release page, and their licence terms followed.
- Equations and standard formulations still need a citation when you take them from a specific source.

### 6.5 Common knowledge vs. claims that need a citation
- Widely known facts do not need a citation. Specific statistics, results, definitions, novel claims, and anyone's interpretation do.
- When unsure, cite.

### 6.6 AI-assisted writing
- Follow your institution's and the venue's policy on AI tools, and disclose use where required.
- Text produced by an AI tool is not automatically original or correctly sourced. It can reproduce phrasing or invent references, so verify every claim and every citation yourself.
- Turnitin also offers AI-writing detection that is separate from similarity, so rewriting in your own voice from your own understanding matters.

### 6.7 Where overlap tends to cluster in technical papers
- Related work, background, and methodology sections often match because of standard descriptions (for example model architectures, dataset statistics, metric definitions).
- Describe what *you* did and why, rather than restating textbook descriptions. When a standard description is needed, keep it short and cite it.

### 6.8 Citation hygiene
- Use one citation style consistently (IEEE, APA, and so on), as the venue requires.
- Verify each reference exists and the details are correct (authors, year, venue, DOI). Do not cite papers you have not actually read.
- Cite the original source, not a secondary one that summarizes it, when possible.

### 6.9 Keep an evidence trail
- Keep notes, version history, and dated drafts. If a match is questioned, these show your writing process.
- Label notes with their source from the start, so you never wonder later where a sentence came from.

### 6.10 Collaboration and co-authors
- Agree on who writes each section, and check co-authors' contributions with the same rules. A match in a co-author's text is still a problem for the whole paper.

### 6.11 Final read-aloud pass
- Read the final text aloud. Phrases that sound unlike your voice are often leftover source wording and are worth a second look.

---

## 7. Keep the goal in mind

The aim is not to trick the checker. Text that evades a match but copies another author's ideas or structure without credit is still plagiarism, and instructors read the report match by match. A low score with honest paraphrase and thorough citation is the target.

---

## Sources

- Turnitin Guides, Using exclusions and filters (students): https://guides.turnitin.com/hc/en-us/articles/23503006879885
- Turnitin Guides, Managing filters and exclusions: https://guides.turnitin.com/hc/en-us/articles/23476785544973
- Turnitin Guides, How exclusion filters refine the Similarity Report: https://guides.turnitin.com/hc/en-us/articles/23539146689549
- Turnitin Guides, Turnitin and plagiarism: https://guides.turnitin.com/hc/en-us/articles/34400565079053
- Turnitin Guides, The Similarity Score and scenarios: https://guides.turnitin.com/hc/en-us/articles/23713493434253
- Turnitin Guides, Understanding the Similarity Report: https://fa-help.turnitin.com/feedback-studio/lti/student/the-similarity-report/interpreting-the-similarity-report.htm
- Turnitin classic filters (small matches/sources, nine-word minimum): https://fa-help.turnitin.com/ithenticate/individual/user/the-similarity-report/filters-and-exclusions.htm
