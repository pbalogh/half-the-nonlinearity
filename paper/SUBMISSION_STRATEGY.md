# Submission Strategy: "Half the Nonlinearity Is Wasted"

*Created: March 3, 2026*

---

## Paper Strengths (for targeting)

1. **Strong negative result** — token-based routing doesn't work (r < 0.05 cross-corpus). Reviewers love clean negative results.
2. **Practical positive result** — 17.3% PPL improvement via two-phase gated linearization
3. **Cross-architecture analysis** — 6 models, 2 families, systematic methodology
4. **Mechanistic insight** — context dominates routing, MLP nonlinearity is contextual not token-level
5. **Architectural implications** — concrete proposals for variable-capacity MLPs, learned routing
6. **Reproducibility** — small models (GPT-2, Pythia), CPU-runnable, clean methodology

## Paper Weaknesses (to anticipate)

1. **Small models only** — largest is 2.8B; reviewers will ask about 7B+
2. **Single-model fine-tuning** — progressive linearization only on GPT-2 Medium
3. **No wall-clock speedup measured** — FLOPs savings are theoretical
4. **No comparison to pruning/distillation/quantization** — acknowledged in caveats
5. **Independent researcher** — no institutional affiliation (can be a plus for novelty, minus for credibility)

---

## Venue Strategy

### Tier 1: Top Conferences

#### ❌ ICML 2026 — DEADLINE PASSED (Jan 28, 2026)
- Would have been ideal (efficiency + interpretability)
- Conference: Jul 6–11, Seoul

#### 🎯 NeurIPS 2026 — DEADLINE ~May 2026 (TBA)
- **Best target.** Paper fits squarely in efficiency + interpretability tracks
- Conference: ~Dec 2026
- Deadline historically mid-May (NeurIPS 2025 was May 15)
- **Action:** Watch for CFP announcement, likely March/April 2026
- **Pros:** Largest ML venue, values both negative results and practical contributions, workshop co-location opportunities
- **Cons:** Very competitive (~25% acceptance), reviewers may want larger models
- **Pitch angle:** "Systematic measurement paper with surprising negative result + practical exploitation"

#### 🎯 ICLR 2027 — DEADLINE ~Oct 2026
- Strong alternative if NeurIPS doesn't work out
- Values mechanistic interpretability work highly
- **Pitch angle:** "Understanding what MLPs actually compute, with implications for architecture design"

### Tier 1.5: Rolling Journal

#### 🎯🎯 TMLR (Transactions on Machine Learning Research) — ROLLING, SUBMIT ANYTIME
- **Strongest recommendation for first submission.** Here's why:
  - Rolling deadline = submit now, no waiting
  - Open review on OpenReview = transparent, builds credibility
  - TMLR papers can be **presented at NeurIPS/ICML/ICLR** via the Journal-to-Conference track (deadline for NeurIPS 2026: Sep 26, 2026)
  - No institutional bias — reviewed on merits
  - Double-blind
  - Faster turnaround than conferences (~2-3 months)
  - Respected venue (Editors: Hugo Larochelle, Kyunghyun Cho, etc.)
- **Strategy:** Submit to TMLR now → if accepted, request presentation slot at NeurIPS 2026 via J2C track
- **This is the optimal path for an independent researcher.**

### Tier 2: Workshops (submit in parallel with main venue)

#### 🎯 NeurIPS 2026 Mechanistic Interpretability Workshop
- Last year's: mechinterpworkshop.com
- Workshop paper deadline typically ~Sep 2026
- Lower bar, great for visibility and networking
- Can submit a shorter version while main paper is under review elsewhere

#### 🎯 ICML 2026 Workshops (Jul 10-11, Seoul)
- Workshop submission deadlines: ~May 2026
- Look for: efficient ML, interpretability, or architecture workshops
- **Still possible if you act fast on workshop CFPs**

### Tier 3: Specialized Venues

#### COLM (Conference on Language Modeling) 2026
- New venue focused specifically on language models
- If they run a 2026 edition, this paper is a perfect fit

#### WANT@NeurIPS (Workshop on Advancing Neural Network Training)
- Efficiency-focused, would value the practical results

---

## ArXiv Strategy

### Posting
- **Post to arXiv BEFORE submitting to any venue** — establishes priority
- Categories: **cs.LG** (primary), **cs.CL** (cross-list)
- This is especially important as an independent researcher — timestamp matters

### Endorsement
- Need an endorser with ≥3 papers in cs.LG or related cs.* categories
- **Options for finding an endorser:**

  1. **Bilinear MLPs authors** (Thomas Dooms et al., ICLR 2025) — most directly related work. Their paper shows bilinear MLPs (no element-wise nonlinearity) achieve competitive performance. Your paper is the empirical complement: measuring when standard MLPs are effectively bilinear already. **Strongest connection.**
     - Contact: Thomas Dooms (GitHub: tdooms, likely KU Leuven)
     - Pitch: "Our work provides empirical evidence for why bilinear MLPs work — most MLP computation is already near-linear"

  2. **Pythia team** (Stella Biderman, EleutherAI) — you use their models extensively and cite them. EleutherAI is friendly to independent researchers.
     - Contact: Stella Biderman (EleutherAI, accessible via Twitter/Discord)
     - Pitch: "Detailed Pythia analysis across 4 model sizes, including full 32-layer sweep of 2.8B"

  3. **Anthropic Transformer Circuits team** — you cite Bricken et al. and Cunningham et al. The SAE/interpretability community would be interested.
     - Less likely to get a response, but worth trying
     - Contact via Transformer Circuits Thread comments

  4. **MoEfication / Efficient Transformer researchers** — the MoE connection is natural
     - Look for authors of MoEfication, Deja Vu, etc.

  5. **ResearchGate / Twitter cold outreach** — post a thread about findings, tag relevant researchers, ask for endorsement
     - Independent researchers successfully get endorsements this way regularly

### Recommended outreach order:
1. Thomas Dooms (bilinear MLPs) — strongest fit
2. Stella Biderman (EleutherAI/Pythia) — most accessible
3. Twitter thread → organic connections

---

## Outreach Email Templates

### Template 1: Bilinear MLPs connection

> Subject: Empirical complement to your bilinear MLPs work
>
> Hi [Name],
>
> I'm an independent researcher. I've been studying when transformer MLP nonlinearity is actually necessary — empirically measuring it across GPT-2 and Pythia models (162M to 2.8B).
>
> The headline finding connects directly to your bilinear MLPs paper: in GPT-2 Large, 11 of 36 MLP layers can be replaced with a linear surrogate that *improves* perplexity. The nonlinearity at those layers is actively harmful. Your work shows bilinear MLPs achieve competitive performance; our work shows why — most MLP computation is already near-linear, and the nonlinear component is often overfitting.
>
> The paper also includes a strong negative result: nonlinearity need cannot be predicted from token identity (cross-corpus r < 0.05). It's fully contextual.
>
> I'd love to share the draft if you're interested. I'm also looking for an arXiv endorser for cs.LG — would you be willing?
>
> [Link to draft / repo]
>
> Best,
> Peter Balogh

### Template 2: Pythia team

> Subject: Full 32-layer Pythia-2.8B linearization analysis
>
> Hi [Name],
>
> I've been doing systematic MLP linearization analysis across four Pythia models (160M, 410M, 1B, 2.8B) and have some findings that might interest the EleutherAI community.
>
> Key result: Pythia-2.8B shows a dramatic 513% perplexity explosion when Layer 0 is linearized, but one middle layer (L3) actually *beats baseline* with gating — the first Pythia layer to do so. The architectural divide between GPT-2 (sequential attn→MLP) and Pythia (parallel) appears to be fundamental: GPT-2 Large's worst layer (3.7%) barely exceeds Pythia-2.8B's best (1.9%).
>
> I'm an independent researcher preparing this for submission. Would you be interested in seeing the draft? I'm also looking for an arXiv endorser for cs.LG.
>
> [Link to draft / repo]
>
> Best,
> Peter Balogh

---

## Timeline

| Date | Action |
|------|--------|
| **Week of Mar 3** | Finalize paper (current), install LaTeX, compile PDF |
| **Week of Mar 10** | Post to arXiv (need endorser by then) |
| **Mar 10-15** | Twitter thread announcing findings, tag relevant researchers |
| **Mar-Apr** | Submit to **TMLR** (rolling) |
| **~May 2026** | Submit to **NeurIPS 2026** (if deadline allows, or workshop) |
| **May-Jun** | Submit workshop paper to **ICML 2026 workshops** |
| **~Sep 2026** | If TMLR accepted: request NeurIPS J2C presentation slot (deadline Sep 26) |
| **~Oct 2026** | Backup: submit to **ICLR 2027** |

---

## Pre-Submission Checklist

- [ ] Install LaTeX, compile PDF, verify all figures render
- [ ] Proofread final PDF end-to-end
- [ ] Write author bio / create ORCID if needed
- [ ] Prepare supplementary materials (data JSONs, code)
- [ ] Create clean GitHub repo for reproducibility
- [ ] Draft Twitter thread (5-7 tweets summarizing key findings)
- [ ] Contact 2-3 potential endorsers
- [ ] Post to arXiv
- [ ] Submit to TMLR

---

## Key Selling Points by Audience

**For efficiency researchers:** 17.3% PPL improvement by removing nonlinearity. 40% of MLP FLOPs potentially saveable. Concrete architectural proposals.

**For interpretability researchers:** Strong negative result on token-based routing. Context dominates. Cautionary tale for mechanistic interpretability (function-word correlation was an artifact).

**For architecture researchers:** Sequential vs parallel computation has dramatic linearizability consequences. Variable-capacity MLP proposal backed by data. Full-sweep layer maps for GPT-2 Large (36 layers) and Pythia-2.8B (32 layers).

**For scaling researchers:** Linearizability improves with scale (Wanamaker effect). Layer 0 threshold in Pythia. U-shape nonlinearity demand curve.
