# Author decisions: alternative wording (draft; the author decides)

The three `\authordecision{...}` markers in `revision/manuscript/main.tex` are left unchanged.
Each section below offers wording options. Nothing here has been inserted into the manuscript.

## 1. Affiliation (main.tex, author block)

Marker: "current affiliation, University of Bonn: list as a present address or not"

- **A. Present address footnote.** Keep Shenzhen Technology University as the affiliation where
  the work was done, and add for the relevant author: "Present address: [department], University
  of Bonn, Bonn, Germany." The department name must come from the author.
- **B. No present address.** Keep the Shenzhen Technology University affiliation only. PeerJ
  takes affiliations from the submission system, so the system entry must match.
- **C. Dual affiliation.** Choose this only if part of the revision work was done at Bonn. Add
  `\affil[2]{[department], University of Bonn, Bonn, Germany}` and mark the author with [1,2].

## 2. Earlier public version (main.tex, Data subsection)

Marker: "an earlier version of this manuscript is public as manuscript/BDS-Lite.pdf in the GitHub
repository; if you treat it as a public preprint, add a two-sentence correction note here"

- **A. Correction note (if it is treated as a preprint):** "An earlier version of this
  manuscript is publicly available in the project repository. Its ISIC2018 results were
  evaluated largely on images that had also been used for training; they are withdrawn and
  replaced by the official-test results reported here (Supplementary S4)."
- **B. No note in the manuscript.** Disclose it in the letter to the editor instead, and remove
  or supersede the PDF in the public repository at the revision release (see
  `revision/RELEASE_GUIDE.md`, decision D-a).
- **C. Both.** Use A in the manuscript and B's repository action.

## 3. Archive DOI (main.tex, Data and code availability)

Marker: "add the Zenodo DOI of the revision release after minting it"

- **A. DOI minted before resubmission:** "... are archived at Zenodo
  (doi:10.5281/zenodo.[number]) and mirrored at https://github.com/Jingchen-Chen/bds-lite."
- **B. DOI not yet minted:** "... are available at https://github.com/Jingchen-Chen/bds-lite;
  a versioned archive with a DOI will be deposited at Zenodo before publication." Check that
  PeerJ's data and code policy accepts a pre-publication commitment.
- **C. Tagged release without DOI:** cite the GitHub release tag and commit hash. This is
  weaker for long-term availability.
