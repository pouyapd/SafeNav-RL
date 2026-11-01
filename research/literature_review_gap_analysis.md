# Literature review and gap analysis: novelty of the SafeNav-RL manuscript

Written 2026-11-01. It extends [`LITERATURE_REVIEW.md`](LITERATURE_REVIEW.md) (2026-10-25) and [`NOVELTY_ASSESSMENT.md`](NOVELTY_ASSESSMENT.md). It covers the manuscript as on branch `exp/e1-pilot` (`paper_draft/manuscript.tex`), including the E1 pilot. Nothing in the code, results or manuscript was changed for this review.

## 1. Method and limitations

**Sources searched.** I used a general web search engine and opened official pages directly: arXiv abstract and HTML pages, PMLR, the NeurIPS virtual site, IEEE Xplore and ACM DL through DOI resolution, and the Crossref REST API.

**Not accessed directly.** Google Scholar and Semantic Scholar were not queried through their own interfaces or APIs. Their results reached me only where the search engine returned them. Paywalled full texts (IEEE, Elsevier) were not read.

**Verification.**
- Every reference **new in this review** was checked in one of two ways:
  - **with a DOI:** I queried Crossref for title, first author, year and venue, and resolved the DOI;
  - **without a DOI:** I checked the official proceedings page (PMLR, NeurIPS) or the arXiv record.
- References already in `paper_draft/references.bib` were verified against Crossref on 2026-10-25 and are not re-listed here, except where this review read more of them.

**Reading depth** is tagged for each paper:
- **[full text]**: the relevant sections were read (for arXiv HTML, the version available on 2026-11-01);
- **[abstract]**: the abstract or landing page only. A claim that such a paper *does not* do something is only a hypothesis.

**Not exhaustive.** This is a targeted search: about 20 queries over the five topics in the task. It is not a systematic review. Section 6, R3, says how to close this.

**Separation of evidence.** Each paper's entry under "Content" states facts from the source. Interpretation is marked as such and is collected in Sections 4–6.

---

## 2. Newly verified references

N1–N6 and N8 were added to the manuscript on 2026-11-01 (Section 8). N7 and N9–N13 were not cited.

| # | Reference | Verified via | Content relevant to SafeNav-RL | Depth |
|---|---|---|---|---|
| N1 | Lin, J.-L., Hung, W., Yang, S.-H., Hsieh, P.-C., Liu, X. (2021). *Escaping from zero gradient: Revisiting action-constrained reinforcement learning via Frank-Wolfe policy optimization.* Proc. UAI 2021, PMLR 161:397–407. https://proceedings.mlr.press/v161/lin21b.html | PMLR page | Projection-based action-constrained RL "could suffer from the zero-gradient problem due to the tight coupling of the policy gradient and the projection". The result is sample-inefficient training and slow convergence. | [abstract] |
| N2 | Fujita, Y., Maeda, S. (2018). *Clipped action policy gradient.* Proc. ICML 2018, PMLR 80:1597–1606. https://proceedings.mlr.press/v80/fujita18a.html | PMLR page | Bounded action spaces: out-of-bound actions are clipped before execution, but standard policy-gradient estimators ignore this. The CAPG estimator uses the clipping; it is unbiased and has lower variance. | [abstract] |
| N3 | Chou, P.-W., Maturana, D., Scherer, S. (2017). *Improving stochastic policy gradients in continuous control with deep reinforcement learning using the Beta distribution.* Proc. ICML 2017, PMLR 70:834–843. https://proceedings.mlr.press/v70/chou17a.html | PMLR page | Bounded actions bias the Gaussian policy gradient. A Beta policy is "bias-free" and converged faster with TRPO and ACER. | [abstract] |
| N4 | Gros, S., Zanon, M., Bemporad, A. (2020). *Safe reinforcement learning via projection on a safe set: How to achieve optimality?* IFAC-PapersOnLine 53(2):8076–8081. [10.1016/j.ifacol.2020.12.2276](https://doi.org/10.1016/j.ifacol.2020.12.2276) | Crossref | Projecting a learned policy onto a safe set can make the learned policy suboptimal. For policy-gradient methods, simple corrections keep the gradient unbiased. | [abstract] |
| N5 | Trautman, P., Krause, A. (2010). *Unfreezing the robot: Navigation in dense, interacting crowds.* IEEE/RSJ IROS 2010, pp. 797–803. [10.1109/IROS.2010.5654369](https://doi.org/10.1109/IROS.2010.5654369) | Crossref | Origin of the "freezing robot problem" in crowd navigation. The abstract says the planner "decides that all forward paths are unsafe, and the robot freezes in place" and that "a feasible path typically exists". | [abstract] (Semantic Scholar record, checked 2026-11-01) |
| N6 | Banerjee, A., Rahmani, K., Biswas, J., Dillig, I. (2024). *Dynamic model predictive shielding for provably safe reinforcement learning.* NeurIPS 2024. https://neurips.cc/virtual/2024/poster/93108 ; arXiv [2405.13863](https://arxiv.org/abs/2405.13863) | NeurIPS site, arXiv | Model predictive shielding "often hinders task progress due to the conservative and task-oblivious nature of backup policies". DMPS replaces the fixed backup with a local planner that uses the learned policy's value. | [abstract] |
| N7 | Kasaura, K., et al. (2023). *Benchmarking actor-critic deep reinforcement learning algorithms for robotics control with action constraints.* IEEE RA-L 8(8):4449–4456. [10.1109/LRA.2023.3284378](https://doi.org/10.1109/LRA.2023.3284378) | Crossref, DOI resolves | First benchmark of action-constrained RL algorithms. A simple baseline that treats the constraint as part of the environment's transition was surprisingly effective. | [abstract] |
| N8 | Saunders, W., Sastry, G., Stuhlmüller, A., Evans, O. (2018). *Trial without error: Towards safe reinforcement learning via human intervention.* Proc. AAMAS 2018, pp. 2067–2069 (extended abstract). Crossref DOI [10.65109/mzmg4429](https://doi.org/10.65109/mzmg4429), which resolves to ACM DL 10.5555/3237383.3238074; arXiv [1707.05173](https://arxiv.org/abs/1707.05173) | Crossref, DOI resolves | A blocker prevents unsafe actions, and the agent receives a negative reward for blocked actions. This is an early intervention penalty. The AAMAS extended abstract (p. 2067) says the human "replaces the new reward … with a penalty r\*". | [full text] (3-page AAMAS extended abstract, checked 2026-11-01) |
| N9 | Yuan, Z., et al. (2022). *safe-control-gym: A unified benchmark suite for safe learning-based control and reinforcement learning in robotics.* IEEE RA-L 7(4):11142–11149. [10.1109/LRA.2022.3196132](https://doi.org/10.1109/LRA.2022.3196132) | Crossref | Benchmark with constraints and safety filters (cart-pole, quadrotors). A candidate standard platform. | [abstract] |
| N10 | Ji, J., et al. (2023). *Safety Gymnasium: A unified safe reinforcement learning benchmark.* NeurIPS 2023, Datasets and Benchmarks Track. https://neurips.cc/virtual/2023/poster/73567 ; arXiv [2310.12567](https://arxiv.org/abs/2310.12567) | NeurIPS site | Safe-navigation tasks (Goal, Button, Push, Circle) and the SafePO library (16 algorithms, including PPO-Lagrangian). | [abstract] |
| N11 | Ju, S., Peters, J., Arenz, O. (2026). *Safe reinforcement learning with receding-horizon safety critic and differentiable action shields.* **Preprint**, arXiv [2610.20255](https://arxiv.org/abs/2610.20255) (14 Oct 2026) | arXiv | A closed-form, differentiable CBF-style filter inside PPO, with an "anticipation loss to reduce policy-filter mismatch during training". Quadruped locomotion. Conceptually close to the correction penalty. | [abstract] |
| N12 | Georgescu, T.-A., Goodall, A. W., Alrajeh, D., Belardinelli, F., Uchitel, S. (2025/2026). *Adaptive GR(1) specification repair for liveness-preserving shielding in reinforcement learning.* **Preprint**, arXiv [2511.02605](https://arxiv.org/abs/2511.02605); the arXiv comments field says accepted at NeSy 2026 | arXiv | Liveness of logic-based (discrete) shields when environment assumptions are violated. Shows that liveness under shielding is studied in formal methods, though not for continuous CBF filters. | [abstract] |
| N13 | Zhang, J., Xu, Y., Aggarwal, V. (2026). *Don't freeze, don't crash: Extending the safe operating range of neural navigation in dense crowds.* **Preprint**, arXiv [2603.06729](https://arxiv.org/abs/2603.06729) | arXiv | RL crowd navigation, with no safety filter in the abstract. Contrasts learned policies with analytical solvers that are "safe but prone to freezing". Freezing is used as an evaluation outcome. | [abstract] |

**Read more deeply than before** (already in the bibliography):

- **Yang et al., *CBF-RL* (ICRA 2026; arXiv [2510.14959](https://arxiv.org/abs/2510.14959)).** Read as [full text] of the arXiv HTML.
  - The abstract says the policy's lack of knowledge of the CBF "can lead to conservative behaviors".
  - The navigation agent is a **single integrator** among randomized circular obstacles.
  - The reward term `r_cbf` "penalizes actions whenever the safety filter is activated", and it also contains an exponential term on the distance between the policy action and the safe action. So it penalises both activation and the size of the correction. This was corrected on 2026-11-01 after re-reading Section III; the first version of this report said activation only.
  - Success rates are reported over 1,000 test environments, but **without seed variance**. **Timeouts are not analysed.** Deadlock or stalling is not discussed.
  - The table's row labels were partly garbled in the HTML, so I do not quote individual numbers.
- **Oh et al., *Provably optimal RL under safety filtering* (IASEAI 2026; arXiv [2510.18082](https://arxiv.org/abs/2510.18082)).** Read as [full text], about 95 % of the HTML.
  - The optimality result assumes a **perfect, least-restrictive filter** (Definition 2): one that intervenes only when the nominal action would leave the maximal controlled-invariant set Ω\*. "Sufficiently permissive" (Remark 2) is not defined more precisely.
  - Deadlock, liveness, getting stuck at the boundary, and correction penalties are **not discussed**.
  - The reward is computed on the filtered action.
  - Experiments: Safety Gymnasium Goal and Circle, Car agent, 5 seeds.
- **Krasowski et al. (TMLR 2023)** and **Thumm et al. (IROS 2023)**, [abstract]:
  - Krasowski et al.: action replacement worked best on their tasks, and a penalty when the safety mechanism acts improved training.
  - Thumm et al.: interventions cause "undesirable behaviors, such as abrupt braking or aggressive steering"; proactive replacement or projection reduces them.
  - Neither abstract mentions stalls or deadlock.

Searches that found nothing closer:
- learned single-robot navigation with a CBF filter combined with "deadlock" or "stuck";
- liveness or deadlock-freedom of learned policies with CBFs;
- "safety filter" with "freezing" in RL.

Hits were either multi-robot work with model-based controllers (already cited: Grover, Jankovic, Chandra), multi-agent RL with an MPC filter and no deadlock analysis (Dawood et al., arXiv [2312.12861](https://arxiv.org/abs/2312.12861), [abstract]), or general conservativeness remarks.

---

## 3. The closest work, compared precisely

| Work | What it does | Overlap with SafeNav-RL | Difference (from the parts read) |
|---|---|---|---|
| **Pizarro Bejarano, Brunke, Schoellig, RA-L 2025** (in the bibliography; read as [full text] on 2026-10-25) | PPO with filtering during training, a correction penalty α‖u − u_cert‖², and safe resets. Crazyflie trajectory tracking, 5 seeds, simulation and hardware. | Same training-time filter design: the uncertified action is stored and the penalty is identical. **E1 tests exactly their penalty.** | No obstacles, no navigation, no stall or deadlock analysis; they report chattering. SafeNav-RL asks whether the penalty removes *navigation stalls*, and of which type. |
| **Yang, Werner, de Sa, Ames, ICRA 2026 (CBF-RL)** | Filtered rollouts plus a CBF reward term that penalizes filter activation; the policy is deployed with or without a runtime filter. | Navigation among circular obstacles; filter in training; an intervention-type penalty. | **Single integrator.** It has no heading, so the turn-rate geometry behind Type I stalls (Props. 1–2) does not arise. No seed variance, no timeout analysis, no mechanism classification. Its "conservative behaviors" remark is the closest published statement of the phenomenon, and it is not analysed there. |
| **Krasowski et al., TMLR 2023; Thumm et al., IROS 2023** | Taxonomy and benchmark of replacement, projection and masking; penalties on interventions; proactive reduction of interventions. | Filter or shield in training; penalties; side effects of interventions. | Pendulum, quadrotor, Safety Gym and human–robot collaboration. Side effects are framed as abrupt braking or steering, not as stalls. |
| **Oh, Nguyen, Hu, Fisac, IASEAI 2026** | Optimality of RL under a *least-restrictive* filter (filtered MDP). | The manuscript's Discussion uses it to argue that the remaining gap is a learning problem. | Its guarantee needs a filter that acts only at the boundary of Ω\*. A one-step DT-CBF speed limit (γh/a) acts before that, so **the theorem does not apply directly** (interpretation; see Section 4, item 5). |
| **Reis, Aguiar, Tabuada 2021; Tan & Dimarogonas 2024** (in the bibliography) | Undesired stable equilibria of CBF-QPs on the safe-set boundary, for model-based nominal controllers. | Type I stalls are the learned-policy counterpart; the manuscript says so. | No learned policy and no dependence on training. |
| **Banerjee et al., NeurIPS 2024 (DMPS)** | Shields with backup policies hinder task progress; DMPS plans task-aware recoveries. | The same "safety mechanism stalls progress" problem, plus a remedy class (recovery planning) that the manuscript does not discuss. | No CBF speed filter, no navigation mechanism analysis in the abstract. |
| **Lin et al., UAI 2021; Fujita & Maeda, ICML 2018; Chou et al., ICML 2017; Gros et al., IFAC 2020** | Policy gradients are degraded by projection or clipping of actions: zero gradient, bias, variance. Corrections exist: Frank-Wolfe updates, the CAPG estimator, Beta policies, unbiased actor-critic formulations. | **The manuscript's "action aliasing" hypothesis for Type II stops is the same family of effect**, applied to the filter's truncation and to the box clip of v ≤ 0. | None of them looks at navigation stalls. They supply both the mechanism and ready-made remedies for E4. |
| **Trautman & Krause, IROS 2010** (and the crowd-navigation literature after it) | The "freezing robot problem": a planner stops because no safe path seems available. | The behaviour resembles Type II self-stops. | A planner with uncertainty in crowds, not a learned policy behind a filter. The term is widely known, and reviewers may expect it to be mentioned. |

---

## 4. Novelty assessment, claim by claim (interpretation, based on Sections 2–3)

This section assesses the manuscript **before** the revision of 2026-11-01. Section 8 records what was changed.

1. **"No published study characterises how learned navigation policies deadlock behind a filter, or how the training setup changes the mechanism"** (Introduction).
   - **Status: plausible and not contradicted by this search, but still only "to our knowledge".**
   - The closest statements are general remarks on conservativeness (CBF-RL), braking and steering side effects (Thumm et al.) and stalled task progress (DMPS). None separates filter-blocked from self-stop stalls, uses a feasibility oracle, or relates the mechanism to how the agent was trained.
   - **Risk:** the sentence reads broader than the evidence. A reviewer who knows the freezing-robot literature, or CBF-RL's remark, could read it as overstated.
   - **Fix:** state the claim narrowly (Section 6, R1).
2. **Contribution 1, the geometric characterisation.**
   - **Status: correct and useful, but low novelty as theory.**
   - Proposition 1 (one-step safety does not depend on ω) follows directly from the forward-Euler update. It is the discrete-time form of the relative-degree issue that high-order CBFs address (Xiao & Belta 2022, already cited).
   - Proposition 3 is close to immediate from the definition of a self-stop.
   - The manuscript already says that the safety proof is standard. The same modesty should be applied to Props. 1 and 3, for example by calling them lemmas or observations whose value lies in what they rule out.
3. **Contribution 2, the evaluation protocol** (feasibility oracle, stall taxonomy, intervals over seeds and scenarios).
   - **Status: supported as a distinguishing feature.** In the papers read, timeouts were not analysed (CBF-RL; Guo et al. 2025), and seed variance was often missing.
   - **But "protocol" may be read as a general contribution.** As implemented, the oracle and the taxonomy are specific to static circles and a unicycle. It is better presented as a diagnostic for this setting that others can adapt.
4. **Contribution 3, the empirical evidence** (two mechanisms, their dependence on training, replication on fresh scenarios, the checkpoint effect).
   - **Status: the strongest and most original part.**
   - The pre-registered replication and the checkpoint analysis are unusual in this sub-area, where typically one run per method is reported (Guo et al. 2025; CBF-RL reports no seed variance).
   - It rests on one simulator, one filter, one robot model and five seeds. These limits are stated in the manuscript.
5. **The use of Oh et al. (2026) in the Discussion.**
   - **Status: overstated as written.** The manuscript says the filter "permits turning in place" and that the gap is therefore plausibly a learning problem.
   - Oh et al.'s guarantee assumes a **least-restrictive** filter. The DT-CBF speed limit γh/a slows the robot whenever it closes on a barrier with γh/a < v, well before the boundary of the maximal controlled-invariant set. It is therefore more restrictive than Oh et al.'s perfect filter.
   - The geometric oracle supports the "learning problem" reading on its own: a margin-keeping path exists. The citation should not be used as if the theorem applied.
6. **The self-stop explanation ("action aliasing")** in the Discussion and Hypothesis 1.
   - **Status: insufficiently grounded.** The mechanism is presented without citing the action-constrained RL literature that documents it: the zero gradient under projection (N1), bias and variance from clipping (N2, N3), and projection bias in safe RL (N4).
   - Citing it makes the hypothesis more credible, and it shows that E4 has standard remedies available: a Beta or tanh-squashed policy, or the CAPG estimator.
7. **The correction penalty (E1).**
   - **Status: a baseline test, not a novelty.** Penalties on interventions have several precedents: discrete shields (Alshiekh et al. 2018), human blockers (N8), Krasowski et al. 2023, the magnitude penalty of Pizarro Bejarano et al. 2025, and CBF-RL's activation penalty.
   - The manuscript frames E1 correctly as "the most important missing baseline".
   - **Only a one-seed pilot exists**, with no large effect, which is inconclusive. Correctly, it is not used as evidence.
8. **Positioning table (Table `tab:related`).**
   - Mostly fair, with "n.r." used carefully.
   - With the full text now read, the CBF-RL row can be made more precise: single integrator; conservativeness noted; deadlock not discussed; timeouts not reported.
   - Rows for DMPS (stalled task progress with shields) and for the freezing-robot problem would pre-empt an obvious reviewer question.

**Overall (interpretation).**
- **Defensible:** the contribution as an analysis paper. It shows that filter-induced stalls of learned navigation policies come from two mechanisms that depend on training, with a protocol that separates them.
- **Not defensible:** novelty claims about the filter, the proof or the penalty. The manuscript does not make them.
- **Weakest points:**
  - the breadth of the "no published study" sentence;
  - the uncited action-constrained RL literature behind the self-stop explanation;
  - the stretched use of Oh et al.;
  - above all, the missing E1a result.

---

## 5. Baselines and comparisons reviewers are likely to expect

| Expected comparison | Why | Cost | Status |
|---|---|---|---|
| Correction-penalty PPO (Pizarro Bejarano et al. 2025), multi-seed | The closest published remedy; already marked as missing in the manuscript | E1a: about 2–2.5 h of laptop compute | pilot only |
| PPO-Lagrangian (Stooke et al. 2020; implementations in SafePO / OmniSafe, N10) | The standard CMDP baseline; safe-RL reviewers expect a constraint-based alternative to filtering | moderate: implementation plus 5–10 runs | missing |
| A classical local planner: DWA (Fox et al. 1997) or MPC with a DT-CBF (Zeng et al. 2021) | Navigation reviewers routinely ask for one (Xiao et al. 2022 survey); also shows whether model-based planners stall on the same scenarios | days of implementation, little compute (no training) | missing |
| Policy parameterisations that avoid clipping effects: Beta policy (N3), tanh-squashed Gaussian, CAPG estimator (N2) | A direct test of the aliasing explanation for Type II stops (E4, arm 1) | small code change plus training | planned (E4) |
| Penalty form: CBF-RL's `r_cbf` (activation plus an exponential term on the correction) versus Pizarro Bejarano's α‖·‖² | The penalty forms in the closest papers differ; a reviewer may ask which one | one extra arm | optional |
| A standard platform (Safety Gymnasium Goal, N10; safe-control-gym, N9; BARN or Arena-Rosnav) | External validity beyond a custom 2D simulator | heavy | optional (E8-like) |

---

## 6. Three recommended changes that need no substantial computation

**R1. Narrow and re-anchor the novelty claims, and correct the use of Oh et al.** Writing only; zero compute.
- Replace the broad "no published study …" sentence with a scoped claim, for example: *single robot, learned policy, a CBF speed filter during training or deployment, stalls classified by mechanism against a feasibility oracle*.
- Add the closest statements of the phenomenon, so that reviewers see they were considered:
  - CBF-RL's "conservative behaviors";
  - Thumm et al.'s braking and steering side effects;
  - DMPS's "hinders task progress";
  - the freezing robot problem (N5).
- Update the CBF-RL row of the positioning table: single integrator, no timeout analysis.
- Rewrite the Oh et al. paragraph. Their theorem assumes a least-restrictive filter, which the one-step DT-CBF is not. The "learning problem" reading rests on the feasibility oracle.
- Present Props. 1 and 3 as straightforward consequences (lemmas or observations), and connect Prop. 1 to the relative-degree view of high-order CBFs.

**R2. Ground the self-stop explanation in the action-constrained RL literature.** Writing only, plus a small design change to E4.
- Cite N1–N4 where "action aliasing" is introduced. Define aliasing as the zero-gradient and clipping-bias effect that the filter's truncation and the box clip of v ≤ 0 create in the policy gradient.
- Rephrase Hypothesis 1 accordingly.
- Choose E4's first arm from the established remedies: a Beta policy (N3), a tanh-squashed Gaussian, or the CAPG estimator (N2). These are standard and cheap to implement, and make E4 easier to defend.
- **Optional zero-compute addition from existing results:** in the fresh-scenario evaluation (L3), 71 % of the 577 self-stops of filter-trained agents end at h ≥ 0.2 m. That is outside the zone where the reward's risk penalty acts, since the risk score is non-zero only within 0.8 m of the robot centre, and h = 0 lies 0.6 m from the surface. This is descriptive, not causal. It shows that "the risk penalty directly makes the robot stop" is not a sufficient explanation, and it sharpens the contrast that E4's reward arm is meant to test.

**R3. Make the literature search reproducible and finish it.** No compute; a few hours of manual work.
- Add a short search-protocol appendix or supplement: databases, query strings, dates, and inclusion and exclusion criteria.
- **Run the queries on Google Scholar and Semantic Scholar directly**; this review could not query them through their own interfaces.
- Suggested queries:
  - `"safety filter" ("deadlock" OR "stuck" OR "freezing") "reinforcement learning" navigation`;
  - `"control barrier function" "reinforcement learning" ("timeout" OR "liveness")`;
  - `"action projection" OR "action clipping" "policy gradient" bias`;
  - forward citations of Pizarro Bejarano et al. 2025, CBF-RL, Oh et al. 2026, and Reis et al. 2021.
- Forward citations of these four papers are the most likely place for a closer competitor to appear.
- This turns the "to our knowledge" statement into one that reviewers can check, and closes gap G4 in `paper_draft/evidence_gap_analysis.md`.

Larger items (E1a, PPO-Lagrangian, a classical planner) remain necessary for a strong journal submission, but they need computation or implementation; see `paper_draft/minimum_experiments.md`.

---

## 7. BibTeX for the new references (as drafted before the revision)

The entries actually added to `paper_draft/references.bib` are the authoritative versions. They differ in small details: full author lists, and the issue number 53(2) for Gros et al., which was confirmed on Crossref.

```bibtex
@inproceedings{lin2021zerogradient,
  title     = {Escaping from Zero Gradient: Revisiting Action-Constrained Reinforcement Learning via {Frank-Wolfe} Policy Optimization},
  author    = {Lin, Jyun-Li and Hung, Wei and Yang, Shang-Hsuan and Hsieh, Ping-Chun and Liu, Xi},
  booktitle = {Proceedings of the Thirty-Seventh Conference on Uncertainty in Artificial Intelligence},
  series    = {Proceedings of Machine Learning Research}, volume = {161}, pages = {397--407}, year = {2021},
  url       = {https://proceedings.mlr.press/v161/lin21b.html}
}
@inproceedings{fujita2018capg,
  title     = {Clipped Action Policy Gradient},
  author    = {Fujita, Yasuhiro and Maeda, Shin-ichi},
  booktitle = {Proceedings of the 35th International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research}, volume = {80}, pages = {1597--1606}, year = {2018},
  url       = {https://proceedings.mlr.press/v80/fujita18a.html}
}
@inproceedings{chou2017beta,
  title     = {Improving Stochastic Policy Gradients in Continuous Control with Deep Reinforcement Learning using the Beta Distribution},
  author    = {Chou, Po-Wei and Maturana, Daniel and Scherer, Sebastian},
  booktitle = {Proceedings of the 34th International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research}, volume = {70}, pages = {834--843}, year = {2017},
  url       = {https://proceedings.mlr.press/v70/chou17a.html}
}
@article{gros2020projection,
  title   = {Safe Reinforcement Learning via Projection on a Safe Set: How to Achieve Optimality?},
  author  = {Gros, Sebastien and Zanon, Mario and Bemporad, Alberto},
  journal = {IFAC-PapersOnLine}, volume = {53}, number = {2}, pages = {8076--8081}, year = {2020},
  doi     = {10.1016/j.ifacol.2020.12.2276}
}
@inproceedings{trautman2010unfreezing,
  title     = {Unfreezing the Robot: Navigation in Dense, Interacting Crowds},
  author    = {Trautman, Peter and Krause, Andreas},
  booktitle = {2010 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages     = {797--803}, year = {2010}, doi = {10.1109/IROS.2010.5654369}
}
@inproceedings{banerjee2024dmps,
  title     = {Dynamic Model Predictive Shielding for Provably Safe Reinforcement Learning},
  author    = {Banerjee, Arko and Rahmani, Kia and Biswas, Joydeep and Dillig, Isil},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)}, year = {2024},
  note      = {arXiv:2405.13863}
}
@article{kasaura2023benchmarking,
  title   = {Benchmarking Actor-Critic Deep Reinforcement Learning Algorithms for Robotics Control With Action Constraints},
  author  = {Kasaura, Kazumi and others},
  journal = {IEEE Robotics and Automation Letters}, volume = {8}, number = {8}, pages = {4449--4456}, year = {2023},
  doi     = {10.1109/LRA.2023.3284378}
}
@inproceedings{saunders2018trial,
  title     = {Trial without Error: Towards Safe Reinforcement Learning via Human Intervention},
  author    = {Saunders, William and Sastry, Girish and Stuhlm{\"u}ller, Andreas and Evans, Owain},
  booktitle = {Proceedings of the 17th International Conference on Autonomous Agents and MultiAgent Systems (AAMAS)},
  pages     = {2067--2069}, year = {2018}, doi = {10.65109/mzmg4429}
}
@article{yuan2022safecontrolgym,
  title   = {safe-control-gym: A Unified Benchmark Suite for Safe Learning-Based Control and Reinforcement Learning in Robotics},
  author  = {Yuan, Zhaocong and others},
  journal = {IEEE Robotics and Automation Letters}, volume = {7}, number = {4}, pages = {11142--11149}, year = {2022},
  doi     = {10.1109/LRA.2022.3196132}
}
@inproceedings{ji2023safetygymnasium,
  title     = {Safety Gymnasium: A Unified Safe Reinforcement Learning Benchmark},
  author    = {Ji, Jiaming and others},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track}, year = {2023},
  note      = {arXiv:2310.12567}
}
```

The full author lists of Kasaura et al., Yuan et al. and Ji et al. should be copied from the publisher records before use. The preprints N11–N13 are deliberately not included, because they are not peer-reviewed.

---

## 8. Outcome: manuscript revision of 2026-11-01

Branch `paper/positioning`. No training or evaluation was run.

**Adopted (each attributed claim was re-checked against the source named):**
- **R1, narrower novelty claim.** The Introduction now states a three-part claim: a single robot, a learned policy behind a CBF filter; stalls separated by mechanism; failures checked against an oracle; the mechanism related to training. It also says the search was targeted, not systematic. Wording was checked against the sources:
  - CBF-RL: "conservative behaviors", from the arXiv abstract. The reward term (Eqs. 22–23) penalises violations of the CBF condition and the distance to the filtered action.
  - Thumm et al.: "can result in … abrupt braking or aggressive steering", from the arXiv abstract.
  - Banerjee et al.: "often hinders task progress", from the arXiv abstract.
  - Trautman & Krause: "a feasible path typically exists", from the abstract.
- **Oh et al. corrected.** The manuscript gives their Definition 2 (a perfect, least-restrictive filter, read in the arXiv HTML). It then shows, within the model and taking the margin set `S` as the safety constraint, that the one-step filter is not least-restrictive:
  - The exit condition `v·a > h/Δt` follows from the bound in the *proof* of Lemma 1, and it is necessary, not sufficient.
  - The filter's cap `γh/a` is therefore at most 1/20 of the speed at which a least-restrictive filter would have to intervene, for a given state, heading and closing barrier.
  - A numerical check gave a ratio of exactly 0.05 against the first-order bound, and 0.038–0.05 against the exact one-step exit speed.
- **R2, self-stops.** Observations are separated from candidate causes. N1–N4 are cited only to frame the aliasing hypothesis. Hypothesis 1 names standard remedies: a Beta policy (N3) and the CAPG estimator (N2).
- **Theory framed as elementary.** "Heading freedom" and the self-stop statement became lemmas. Lemma 3 is restricted to steps with near-zero speed commands under the test-time speed limit. It explicitly excludes earlier steps and training.
- **Correction penalty.** It is described as the penalty term only (α = 1, our one-step filter, no safe resets). Pizarro Bejarano et al. were re-read in the arXiv HTML: the penalty is ‖u_uncert − u_cert‖², input constraints are part of the filter, there are box state bounds and no obstacles, and there were 5 seeds. Earlier precedents are cited: Alshiekh et al. (arXiv full text, Section 5.2, the punishment option) and Saunders et al. (the AAMAS extended abstract).
- **Bibliography.** Seven references were added (N1–N6, N8). Title capitalisation was protected for CBF-RL, BarrierNet, OptLayer, LiDAR, CBF-Based, Recovery RL, Lagrangian and Arena-Rosnav. The Hsu et al. (RSS 2021) authors were corrected: the asterisks marked equal contribution and are not part of the names (RSS proceedings page).

**Not adopted:**
- The 71 % self-stop/risk-zone figure (R2). It is a new number and needs a script and a traceable source before it enters the manuscript.
- Preprints N11–N13.
- Benchmark references N7, N9 and N10. No current claim needs them.
- The statement that Proposition 1's collapse holds for any `S`-preserving filter "with 1/Δt in place of γ". It is true only asymptotically, not by direct substitution, so it was left out.

**Still open:**
- The eight `[EVIDENCE REQUIRED]` items in the manuscript: E1 (four places), published baselines, E3, E4, and the archived release.
- A systematic search (R3, gap G4).
- A second reader for the theory, including the one-twentieth argument (G10).

**Literature checked only in part:**
- CBF-RL, Oh et al., Alshiekh et al., Thumm et al. and Pizarro Bejarano et al.: only the arXiv versions were read, not the published ones.
- Krasowski et al. and Lin, Fujita, Chou and Gros: abstracts only.
- Banerjee et al.: confirmed through the NeurIPS 2024 virtual site and arXiv; the volume and pages in the final proceedings were not found, so they are omitted.
- Google Scholar and Semantic Scholar were not searched through their own interfaces.
