<!-- Generated from PEERJ_REQUIREMENTS_CHECKLIST.template.md by analysis/revision/render_response.py; do not edit. Line numbers refer to revision/manuscript/main.pdf. -->
# PeerJ Computer Science, AI Application article: requirements checklist

Status of the revised manuscript on 2026-11-11. "Before" is the manuscript at the start of this
editing round; "final location" gives margin line numbers of `revision/manuscript/main.pdf`.
An item is marked **met** only where the text and its evidence exist; items that depend on the
authors are marked **open**.

| # | Requirement | Before | Change made | Final location | Status |
|---|---|---|---|---|---|
| 1 | Code used in the study is supplied with the submission or in an accessible repository or archive | The "Data and code availability" section was removed from the main text at the author's request during this revision. Supplementary S16 describes two evidence archives. The submitted version cited `https://github.com/Jingchen-Chen/bds-lite` (a repository URL, not a DOI) | The data statement of the submission system names the GitHub repository and the dataset sources; Supplementary S16 and the response letter (section 5) give the same repository | Submission system; Supplementary S16; response letter, section 5 | met (repository must hold the revised code at submission) |
| 2a | Introduction: motivation for the research | One paragraph mixing clinical motivation with the DSC relation | Labelled Motivation paragraph with cited clinical uses of contours and an explicit statement that no clinical outcome is measured | lines 31–40 | met |
| 2b | Introduction: motivation for the technical choices | Compact U-Net motivated in one sentence | Paragraph "Why a compact U-Net"; the routes compared are motivated in the framework subsection | lines 41–52; lines 141–148 | met |
| 3a | Third-party datasets: name, source URL or DOI, version, split | Names, citations and splits given; no URL or DOI in the main text | URLs for ISIC2018 and ACDC and the DOI of Synapse added; splits in a labelled paragraph | lines 166–192; Supplementary S16 | met |
| 3b | Preprocessing actually performed | Present | Labelled paragraph; third-party conversion of ACDC and its reconstruction for the official test patients stated | lines 185–192 | met |
| 3c | Rationale for models, losses, teacher and comparison methods | Partly present | Reasons stated for the reference methods, the fixed segmentation loss, the teacher and each control | lines 157–164; lines 197–199; lines 226–229; lines 237–245 | met |
| 3d | Computing infrastructure: operating system, hardware, software | GPU, CUDA and PyTorch versions | Python, cuDNN and NumPy versions added from the environment recorded with each run. The operating system is not in the run records and is not stated | lines 251–260; Supplementary S16 | **partly met**: OS not recorded (author) |
| 3e | Evaluation method, metric definitions, rationale, limitations | Definitions present; no rationale | Paragraph on why Boundary F1 is primary and what it does not measure | lines 261–276 | met |
| 3f | Empty classes, aggregation unit, seeds, intervals | Present | Kept; results described as confirmed, detected or not detected, with 0.01 as a preset analysis threshold | lines 266–271; lines 278–304; lines 291–304 | met |
| 4a | Results reported clearly with tables and figures | Tables 1–5, Figures 1–3 | Added Table 1 (methods), Table 7 (missed structures and false alarms), Table 8 (published results, with volume-level scores of our U-Net), Table 9 (inference cost), Figure 3 (ACDC slices); fresh-seed intervals in Table 5B | Tables 1–9; Figures 1–3 | met |
| 4b | Comparison with published results | Supplementary S7 only | Table 8 with the protocol of each source; stated as context, not a ranking | lines 374–380 | met |
| 4c | Discussion uses the metrics defined in Methods | Present | Discussion rewritten around the title question | lines 393–400; lines 402–418; lines 420–442 | met |
| 5 | Conclusions: model type and purpose, limitations, future directions | Findings only | Three paragraphs: model type and purpose, findings, limits and future work | lines 484–494 | met |
| 6a | Acknowledgments | Present (author-supplied text) | Unchanged. The AI-tool statement names ChatGPT only; whether other tools must be disclosed is for the authors | line 496 | **open** (author to confirm wording) |
| 6b | References complete and matching citations | Present, with venue notes and arXiv version suffixes | "MIDL 2018 oral", "Poster", "NIPS 2014 Deep Learning Workshop" and three `v1` suffixes removed; build reports no undefined citation | Reference list | met |
| 7 | Funding, competing interests, author contributions, ethics | Ethics sentence in Data; the other statements are entered in the submission system | Unchanged; none was written for the authors | lines 185–192 (ethics) | **open** (author, submission system) |
