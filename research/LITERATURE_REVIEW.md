# Literature review: safe RL, CBF safety filters and robot navigation

Branch `research/q1-feasibility-audit` · written 2026-10-25 · companion to [`NOVELTY_ASSESSMENT.md`](NOVELTY_ASSESSMENT.md)

## How this review was made, and how far it can be trusted

- **Bibliographic facts are verified.** Every DOI below was resolved against the Crossref API on 2026-10-25, and the title, first author, year and venue it returned match the entry. Items without a DOI (NeurIPS, ICML and PMLR proceedings, TMLR, arXiv) are marked as such. Preprints are labelled **preprint**, and I do not treat them as peer-reviewed.
- **Content claims have different strengths.** Each paper carries one of three tags:
  - **[full text]**: I read the relevant sections;
  - **[abstract]**: I read only the abstract or the publisher/arXiv landing page;
  - **[known]**: a standard result that I did not re-read for this review. These are classical, widely cited papers, and I only cite them for the result they are known for.

  A claim about what a paper *does not* do (for example "no deadlock analysis") is only as strong as its tag. For [abstract] items it is a hypothesis to check before any paper is written.
- **The search is not exhaustive.** It used targeted web searches and Crossref title queries, not a systematic review protocol. The novelty assessment flags where an exhaustive search could still find a closer competitor. The most likely places are recent arXiv preprints on deadlock in learned navigation policies, and workshop papers.

---

## 1. Control barrier functions and safety filters

| Ref | Content relevant here | Tag |
|---|---|---|
| Ames, Xu, Grizzle, Tabuada (2017), *CBF-based quadratic programs for safety critical systems*, IEEE TAC 62(8). [10.1109/TAC.2016.2638961](https://doi.org/10.1109/TAC.2016.2638961) | Continuous-time CBF condition `ḣ ≥ −α(h)` and the CBF-QP safety filter. | [known] |
| Ames et al. (2019), *Control barrier functions: theory and applications*, ECC. [10.23919/ECC.2019.8796030](https://doi.org/10.23919/ECC.2019.8796030) | Tutorial overview; cited in `cbf_filter.py`. | [known] |
| Agrawal, Sreenath (2017), *Discrete control barrier functions …*, RSS XIII. [10.15607/RSS.2017.XIII.073](https://doi.org/10.15607/RSS.2017.XIII.073) | Discrete-time CBF condition `h(x_{t+1}) ≥ (1−γ) h(x_t)`, the condition SafeNav-RL uses (with `γ·dt` in place of γ). | [known] |
| Zeng, Zhang, Sreenath (2021), *Safety-critical MPC with discrete-time CBF*, ACC. [10.23919/ACC50511.2021.9483029](https://doi.org/10.23919/ACC50511.2021.9483029) | DT-CBF constraints inside MPC: a multi-step filter, in contrast to SafeNav-RL's one-step filter. | [known] |
| Xiao, Belta (2022), *High-order control barrier functions*, IEEE TAC. [10.1109/TAC.2021.3105491](https://doi.org/10.1109/TAC.2021.3105491) | Constraints with relative degree > 1. Relevant because the unicycle's turn rate ω has no effect on the position barrier in the same step (see the audit, §2.2). | [known] |
| Hsu, Hu, Fisac (2024), *The safety filter: a unified view …*, Annu. Rev. Control Robot. Auton. Syst. [10.1146/annurev-control-071723-102940](https://doi.org/10.1146/annurev-control-071723-102940) | Survey that places CBF, HJ-reachability and predictive filters in one framework; discusses filter permissiveness. | [abstract] |
| Wabersich, Zeilinger (2021), *A predictive safety filter for learning-based control …*, Automatica. [10.1016/j.automatica.2021.109597](https://doi.org/10.1016/j.automatica.2021.109597) | MPC-based safety filter for learning controllers; a multi-step alternative to one-step CBF filters. | [known] |
| Bansal, Chen, Herbert, Tomlin (2017), *HJ reachability: a brief overview …*, CDC. [10.1109/CDC.2017.8263977](https://doi.org/10.1109/CDC.2017.8263977) | Reachability-based safe sets: a least-restrictive filter, but costly to compute. | [known] |
| Fisac et al. (2019), *A general safety framework for learning-based control …*, IEEE TAC. [10.1109/TAC.2018.2876389](https://doi.org/10.1109/TAC.2018.2876389) | Least-restrictive supervisory control around a learning controller. | [known] |
| Chen, Jankovic, Santillo, Ames (2021), *Backup control barrier functions: formulation and comparative study*, CDC. [10.1109/CDC45484.2021.9683111](https://doi.org/10.1109/CDC45484.2021.9683111) | Filters built from a backup controller's reachable set; relevant to escape/recovery designs. | [abstract] |
| Wang, Ames, Egerstedt (2017), *Safety barrier certificates for collisions-free multirobot systems*, IEEE T-RO. [10.1109/TRO.2017.2659727](https://doi.org/10.1109/TRO.2017.2659727) | CBF collision avoidance for mobile robots; the standard robotics reference. | [known] |

## 2. Safe RL: surveys and benchmarks

| Ref | Content relevant here | Tag |
|---|---|---|
| García, Fernández (2015), *A comprehensive survey on safe reinforcement learning*, JMLR 16 (no DOI). | Taxonomy: changing the optimisation criterion vs. changing exploration. | [known] |
| Brunke et al. (2022), *Safe learning in robotics …*, Annu. Rev. Control Robot. Auton. Syst. [10.1146/annurev-control-042920-020211](https://doi.org/10.1146/annurev-control-042920-020211) | Survey linking learning-based control and safe RL; safety filters as one family. | [abstract] |
| Gu et al. (2024), *A review of safe RL: methods, theories and applications*, IEEE TPAMI. [10.1109/TPAMI.2024.3457538](https://doi.org/10.1109/TPAMI.2024.3457538) | Recent broad review. | [abstract] |
| Krasowski, Thumm, Müller, Schäfer, Wang, Althoff (2023), *Provably safe RL: conceptual analysis, survey, and benchmarking*, TMLR (no DOI; arXiv 2205.06750). | Classifies provably safe RL into **action replacement, action projection and action masking**. In their benchmark, replacement worked best and **a reward penalty when the safety mechanism acts improved training**. | [abstract] |
| Achiam, Held, Tamar, Abbeel (2017), *Constrained policy optimization*, ICML, PMLR 70 (no DOI). | CMDP-based safe RL; constraint satisfaction in expectation, not per step. | [known] |
| Stooke, Achiam, Abbeel (2020), *Responsive safety in RL by PID Lagrangian methods*, ICML, PMLR 119 (no DOI). | PPO-Lagrangian variants; a standard CMDP baseline. | [known] |
| Ray, Achiam, Amodei (2019), *Benchmarking safe exploration in deep RL* (Safety Gym), OpenAI technical report (not peer-reviewed). | Navigation-style safe-RL benchmark with hazards. | [known] |

## 3. Safety filters and shields *inside* RL training (closest to this repository)

| Ref | Content relevant here | Tag |
|---|---|---|
| Alshiekh et al. (2018), *Safe reinforcement learning via shielding*, AAAI. [10.1609/aaai.v32i1.11797](https://doi.org/10.1609/aaai.v32i1.11797) | Shields that replace unsafe actions during learning. The post-posed shield can also **give the agent a punishment for the unsafe action it proposed**. This is early prior art for intervention penalties. | [known] |
| Cheng, Orosz, Murray, Burdick (2019), *End-to-end safe RL through barrier functions …*, AAAI. [10.1609/aaai.v33i01.33013387](https://doi.org/10.1609/aaai.v33i01.33013387) | CBF filter around an RL policy during training (RL-CBF), with a GP model of the dynamics. | [known] |
| Pham, De Magistris, Tachibana (2018), *OptLayer …*, ICRA. [10.1109/ICRA.2018.8460547](https://doi.org/10.1109/ICRA.2018.8460547) | Constrained projection layer on the policy output. | [known] |
| Marvi, Kiumarsi (2021), *Safe RL: a control barrier function optimization approach*, Int. J. Robust Nonlinear Control 31(6) (online 2020). [10.1002/rnc.5132](https://doi.org/10.1002/rnc.5132) | Barrier terms in the RL objective. | [abstract] |
| Emam, Notomista, Glotfelter, Kira, Egerstedt, *Safe RL using robust control barrier functions*, IEEE RA-L 10(3), 2025 (DOI issued 2022). [10.1109/LRA.2022.3216996](https://doi.org/10.1109/LRA.2022.3216996) | Robust CBF filter during RL training, with model uncertainty. | [abstract] |
| Thananjeyan et al. (2021), *Recovery RL …*, IEEE RA-L. [10.1109/LRA.2021.3070252](https://doi.org/10.1109/LRA.2021.3070252) | A learned recovery policy takes over near constraint violations. | [known] |
| Carr, Jansen, Junges, Topcu (2023), *Safe RL via shielding under partial observability*, AAAI. [10.1609/aaai.v37i12.26723](https://doi.org/10.1609/aaai.v37i12.26723) | Shields when the agent observes only part of the state; relevant because the SafeNav policy sees only 16 beams while the filter sees the true map. | [abstract] |
| Xiao et al. (2023), *BarrierNet: differentiable CBFs …*, IEEE T-RO. [10.1109/TRO.2023.3249564](https://doi.org/10.1109/TRO.2023.3249564) | A CBF-QP as a differentiable layer, so the policy is trained through the filter. | [known] |
| **Pizarro Bejarano, Brunke, Schoellig (2025), *Safety filtering while training: improving the performance and sample efficiency of RL agents*, IEEE RA-L 10(1):788–795.** [10.1109/LRA.2024.3512374](https://doi.org/10.1109/LRA.2024.3512374) | **Closest methodological competitor.** PPO trained with three modifications: (i) filtering training actions, with the *uncertified* action stored in the on-policy buffer (the same choice SafeNav-RL v0.2 makes); (ii) a **penalty α‖u_uncert − u_cert‖²** on the correction; (iii) safe resets. Five seeds. Filtering during training with the penalty gave higher return, fewer training violations and less chattering. Tasks: Crazyflie quadrotor trajectory tracking in simulation and on hardware. **No obstacles, no navigation, no deadlock analysis**; it reports chattering against the constraint, which is related. | [full text] |
| Yang, Werner, de Sa, Ames (2025), *CBF-RL: safety filtering RL in training with control barrier functions*, **preprint**, arXiv 2510.14959. | CBF filtering of rollouts during training plus a CBF-based reward term, so that the policy can be deployed without a runtime filter. Navigation tasks and a Unitree G1 humanoid. | [abstract] |
| Oh, Nguyen, Hu, Fisac (2026), *Provably optimal RL under safety filtering*, Proc. IASEAI 2(1):488–500. [10.1609/iaseai.v2i1.43046](https://doi.org/10.1609/iaseai.v2i1.43046) | Treats the filter as part of the environment (a "filtered MDP"). With a **sufficiently permissive** filter, learning is safe, standard RL convergence carries over, and the best filtered policy reaches the best safe return. Recommends training with the most permissive filter available. Deadlock is not discussed in the abstract. | [abstract] |

## 4. CBF + RL for mobile-robot navigation

| Ref | Content relevant here | Tag |
|---|---|---|
| Song et al. (2022), *Safe RL for LiDAR-based navigation via control barrier function*, ICMLA. [10.1109/ICMLA55696.2022.00045](https://doi.org/10.1109/ICMLA55696.2022.00045) | Same problem setting (LiDAR navigation, RL plus CBF). Content not read. | title and venue only |
| Marzari, Trotti, Marchesini, Farinelli (2025), *Designing CBF via probabilistic enumeration for safe RL navigation*, IEEE RA-L 10(10). [10.1109/LRA.2025.3596431](https://doi.org/10.1109/LRA.2025.3596431) | Uses neural-network verification to find unsafe regions of a trained policy, then builds a CBF layer that wraps it. Mobile-robot benchmark plus a real robot; zero violations reported. | [abstract] |
| Xie et al. (2025), *Certificated actor-critic: hierarchical RL with CBFs for safe navigation*, ICRA. [10.1109/ICRA55743.2025.11127225](https://doi.org/10.1109/ICRA55743.2025.11127225) | CBF-derived certificates in the actor-critic objective for navigation. | title and venue; arXiv abstract seen |
| Guo et al. (2025), *Risk-aware RL with dynamic safety filter for collision risk mitigation in mobile robot navigation*, Sensors 25(17):5488. [10.3390/s25175488](https://doi.org/10.3390/s25175488) | PPO plus a CBF-QP filter in Gazebo, with static obstacles and pedestrians. **One run per method, no intervals, timeouts not reported.** Shows the typical evidence level in this sub-area. | [full text] |
| Keyumarsi, Atman, Gusrialdi (2024), *LiDAR-based online CBF synthesis for safe navigation in unknown environments*, IEEE RA-L 9(2). [10.1109/LRA.2023.3339059](https://doi.org/10.1109/LRA.2023.3339059) | Builds the barrier online from LiDAR (GP-based) for unicycle robots; removes the known-map assumption that SafeNav-RL makes. | [abstract] |
| Long, Qian, Cortés, Atanasov (2021), *Learning barrier functions with memory for robust safe navigation*, IEEE RA-L 6(3). [10.1109/LRA.2021.3070250](https://doi.org/10.1109/LRA.2021.3070250) | Barrier (signed distance) learned online from range data. | [abstract] |
| Tai, Paolo, Liu (2017), *Virtual-to-real deep RL: continuous control of mobile robots for mapless navigation*, IROS. [10.1109/IROS.2017.8202134](https://doi.org/10.1109/IROS.2017.8202134) | Sparse-LiDAR (10 beams) mapless RL navigation; close to SafeNav-RL's observation model. | [known] |
| Xiao et al. (2022), *Motion planning and control for mobile robot navigation using machine learning: a survey*, Autonomous Robots. [10.1007/s10514-022-10039-8](https://doi.org/10.1007/s10514-022-10039-8) | Survey of learned navigation; criticises evaluations that lack classical baselines. | [abstract] |

## 5. Safety vs. task completion: local minima, undesired equilibria, deadlock

| Ref | Content relevant here | Tag |
|---|---|---|
| Koren, Borenstein (1991), *Potential field methods and their inherent limitations …*, ICRA. [10.1109/ROBOT.1991.131810](https://doi.org/10.1109/ROBOT.1991.131810) | Local minima and oscillations of potential fields; explains the potential-field baseline's timeouts. | [known] |
| **Reis, Aguiar, Tabuada (2021), *CBF-based QPs introduce undesirable asymptotically stable equilibria*, IEEE L-CSS.** [10.1109/LCSYS.2020.3004797](https://doi.org/10.1109/LCSYS.2020.3004797) | A nominal stabilising controller plus a CBF-QP can create **stable equilibria on the safe-set boundary**: the formal version of the "robot stops in front of the obstacle" deadlock. | [known] |
| Tan, Dimarogonas (2024), *On the undesired equilibria induced by CBF-based QPs*, Automatica. [10.1016/j.automatica.2023.111359](https://doi.org/10.1016/j.automatica.2023.111359) | General characterisation of those equilibria and their stability. | [abstract] |
| Grover, Liu, Sycara (2021), *Deadlock analysis and resolution for multi-robot systems*, WAFR (Springer SPAR). [10.1007/978-3-030-66723-8_18](https://doi.org/10.1007/978-3-030-66723-8_18) | Deadlock as a force equilibrium on the boundary of the safe set; decentralised resolution. | [abstract] |
| Grover, Liu, Sycara (2022/2023), *The before, during, and after of multi-robot deadlock*, IJRR 42(6). [10.1177/02783649221074718](https://doi.org/10.1177/02783649221074718) | Which parameters, initial conditions and goals lead CBF controllers into deadlock; provably correct resolution. | [abstract] |
| Jankovic, Santillo (2021), *Collision avoidance and liveness of multi-agent systems with CBF-based controllers*, CDC. [10.1109/CDC45484.2021.9682854](https://doi.org/10.1109/CDC45484.2021.9682854); Jankovic, Santillo, Wang (2024), *… liveness from instability*, IEEE TCST. [10.1109/TCST.2023.3324531](https://doi.org/10.1109/TCST.2023.3324531) | Liveness (no deadlock) for CBF controllers by making the deadlock equilibria unstable. | [abstract] |
| Chandra et al. (2025), *Deadlock-free, safe, and decentralized multi-robot navigation in social mini-games via discrete-time CBFs*, Autonomous Robots. [10.1007/s10514-025-10194-8](https://doi.org/10.1007/s10514-025-10194-8) | Deadlock-free navigation with **discrete-time** CBFs: the closest deadlock reference that uses DT-CBFs (multi-robot, not learning-based). | [abstract] |

**Gap visible in this section (hypothesis, to be confirmed by an exhaustive search).** The deadlock literature analyses *model-based* nominal controllers (go-to-goal, LQR, multi-robot CBF-QPs). I did not find a peer-reviewed analysis of the undesired equilibria created when the nominal controller is a **learned RL policy** that has been trained with or without the filter, nor of which training choices remove them.

## 6. Dynamic obstacles

| Ref | Content relevant here | Tag |
|---|---|---|
| Fiorini, Shiller (1998), *Motion planning in dynamic environments using velocity obstacles*, IJRR. [10.1177/027836499801700706](https://doi.org/10.1177/027836499801700706) | Velocity obstacles. | [known] |
| van den Berg, Guy, Lin, Manocha (2011), *Reciprocal n-body collision avoidance* (ORCA), ISRR, Springer STAR. [10.1007/978-3-642-19457-3_1](https://doi.org/10.1007/978-3-642-19457-3_1) | Standard reactive multi-agent baseline. | [known] |
| Fox, Burgard, Thrun (1997), *The dynamic window approach to collision avoidance*, IEEE RAM. [10.1109/100.580977](https://doi.org/10.1109/100.580977) | Standard classical local planner (DWA); the baseline most often missing from RL navigation papers. | [known] |
| Chen, Liu, Kreiss, Alahi (2019), *Crowd-robot interaction …*, ICRA. [10.1109/ICRA.2019.8794134](https://doi.org/10.1109/ICRA.2019.8794134) | CrowdNav (SARL): RL among pedestrians. | [known] |
| Everett, Chen, How (2021), *Collision avoidance in pedestrian-rich environments with deep RL*, IEEE Access. [10.1109/ACCESS.2021.3050338](https://doi.org/10.1109/ACCESS.2021.3050338) | GA3C-CADRL. | [known] |
| Long et al. (2018), *Towards optimally decentralized multi-robot collision avoidance via deep RL*, ICRA. [10.1109/ICRA.2018.8461113](https://doi.org/10.1109/ICRA.2018.8461113); Fan et al. (2020), IJRR. [10.1177/0278364920916531](https://doi.org/10.1177/0278364920916531) | PPO with raw LiDAR for multi-robot navigation; a hybrid with a classical fallback controller. | [known] |
| Xie, Dames (2023), *DRL-VO …*, IEEE T-RO. [10.1109/TRO.2023.3257549](https://doi.org/10.1109/TRO.2023.3257549) | RL with a velocity-obstacle reward in crowds. | [abstract] |
| Thontepu et al. (2023), ICC. [10.1109/ICC61519.2023.10442173](https://doi.org/10.1109/ICC61519.2023.10442173); Goswami et al. (2024), ACC. [10.23919/ACC60939.2024.10644338](https://doi.org/10.23919/ACC60939.2024.10644338); Tayal et al. (2024), ACC. [10.23919/ACC60939.2024.10644548](https://doi.org/10.23919/ACC60939.2024.10644548) | **Collision-cone CBFs** for moving obstacles (UGVs, UAVs). | [abstract] |
| Lindemann et al. (2023), *Safe planning in dynamic environments using conformal prediction*, IEEE RA-L. [10.1109/LRA.2023.3292071](https://doi.org/10.1109/LRA.2023.3292071) | Conformal prediction regions for predicted obstacle motion, giving probabilistic safety. | [abstract] |
| Shin et al. (2025), *Egocentric conformal prediction for safe and efficient navigation in dynamic cluttered environments*, CDC. [10.1109/CDC57313.2025.11312645](https://doi.org/10.1109/CDC57313.2025.11312645) | Recent conformal navigation work. | title and venue only |

## 7. Robust and uncertainty-aware safety filters

| Ref | Content relevant here | Tag |
|---|---|---|
| Jankovic (2018), *Robust control barrier functions for constrained stabilization of nonlinear systems*, Automatica. [10.1016/j.automatica.2018.07.004](https://doi.org/10.1016/j.automatica.2018.07.004) | Robust CBFs under bounded disturbances. | [known] |
| Kolathaya, Ames (2019), *Input-to-state safety with control barrier functions*, IEEE L-CSS. [10.1109/LCSYS.2018.2853698](https://doi.org/10.1109/LCSYS.2018.2853698) | ISSf: a bounded degradation of the safe set under disturbances. | [known] |
| Cosner et al. (2021), *Measurement-robust CBFs: certainty in safety with uncertainty in state*, IROS. [10.1109/IROS51168.2021.9636584](https://doi.org/10.1109/IROS51168.2021.9636584) | CBF conditions robust to bounded state-estimation error. | [abstract] |
| Cosner et al. (2023), *Robust safety under stochastic uncertainty with discrete-time CBFs*, RSS XIX. [10.15607/RSS.2023.XIX.084](https://doi.org/10.15607/RSS.2023.XIX.084) | Probabilistic safety over a finite horizon for **discrete-time** CBFs with stochastic disturbances. The natural theory for a noisy version of the SafeNav filter. | [abstract] |
| Ahmadi, Xiong, Ames (2022), *Risk-averse control via CVaR barrier functions …*, IEEE L-CSS 6. [10.1109/LCSYS.2021.3086854](https://doi.org/10.1109/LCSYS.2021.3086854) | CVaR barrier functions in discrete time. | [abstract] |
| Singletary et al. (2023), *Safe control for nonlinear systems with stochastic uncertainty via risk control barrier functions*, IEEE L-CSS. [10.1109/LCSYS.2022.3187458](https://doi.org/10.1109/LCSYS.2022.3187458) | Risk CBFs. | title and venue only |
| Daş et al. (2026), *Safe navigation under state uncertainty: online adaptation for robust CBFs*, IEEE RA-L (arXiv 2508.19159). | Robust CBFs under imperfect state estimates are often too conservative (infeasibility, high control effort). Proposes online adaptation of the robust parameters to reduce this conservatism, tested on a tracked vehicle among obstacles. **Closest competitor for the uncertainty direction.** | [abstract] (journal DOI not yet confirmed on Crossref) |

## 8. Evaluation methodology for RL navigation

| Ref | Content relevant here | Tag |
|---|---|---|
| Henderson et al. (2018), *Deep reinforcement learning that matters*, AAAI. [10.1609/aaai.v32i1.11694](https://doi.org/10.1609/aaai.v32i1.11694) | Seed variance, sensitivity to implementation details, the need for many seeds and significance tests. | [known] |
| Agarwal et al. (2021), *Deep RL at the edge of the statistical precipice*, NeurIPS (no DOI; arXiv 2108.13264). | Stratified bootstrap intervals, interquartile mean, performance profiles for few-run studies (`rliable`). | [abstract] |
| Perille et al. (2020), *Benchmarking metric ground navigation* (BARN), SSRR. [10.1109/SSRR50563.2020.9292572](https://doi.org/10.1109/SSRR50563.2020.9292572) | Benchmark of 300 navigation environments with difficulty metrics; a candidate held-out test set. | [abstract] |
| Kästner et al. (2021), *Arena-Rosnav …*, IROS. [10.1109/IROS51168.2021.9636226](https://doi.org/10.1109/IROS51168.2021.9636226); Kästner et al. (2023), *Arena-Rosnav 2.0 …*, IROS. [10.1109/IROS55552.2023.10342152](https://doi.org/10.1109/IROS55552.2023.10342152) | ROS-based platform that compares DRL planners with classical planners (DWA, TEB, MPC) in dynamic scenes. | [abstract] |
| Francis et al. (2025), *Principles and guidelines for evaluating social robot navigation algorithms*, ACM THRI. [10.1145/3700599](https://doi.org/10.1145/3700599) | Community evaluation guidelines (metrics, scenarios, reporting). | [abstract] |
| Anderson et al. (2018), *On evaluation of embodied navigation agents*, arXiv 1807.06757 (not peer-reviewed). | SPL metric (used in the repository). | [known] |

---

## Summary of the state of the art relevant to SafeNav-RL (verified vs. hypothesis)

**Verified from the sources above:**
1. CBF safety filters for unicycle navigation, discrete-time CBF conditions and their one-step invariance proofs are textbook material (§1). The proof in `docs/SAFETY.md` is correct (see the audit) but is not a contribution.
2. Training PPO with a safety filter in the loop, storing the policy's own action, and adding a correction penalty has been studied (Pizarro Bejarano et al. 2025; Krasowski et al. 2023; Alshiekh et al. 2018 for discrete shields). A theory exists for when filtered training loses no asymptotic performance (Oh et al. 2026: a sufficiently permissive filter).
3. CBF-QP filters are known to create undesired stable equilibria (deadlocks) on the boundary of the safe set for model-based nominal controllers (Reis et al. 2021; Tan & Dimarogonas 2024). Multi-robot deadlock resolution with CBFs is an active topic (Grover; Jankovic; Chandra).
4. Robust, measurement-robust and stochastic discrete-time CBF variants exist (§7), including a 2026 navigation paper that targets their conservatism (Daş et al.).

**Hypotheses (not established by this review; each must be checked by a systematic search before writing):**
- H-L1: no peer-reviewed work analyses filter-induced equilibria of a *learned* navigation policy, or compares training interventions (filter in the loop, correction penalty, observing the speed limit, heading-aware filters, stochastic deployment) by their effect on those equilibria.
- H-L2: no CBF+RL navigation paper reports timeouts/deadlocks against a geometric feasibility oracle, as SafeNav-RL does. In the papers inspected, timeouts were either not reported (Guo et al. 2025) or not separated from feasibility.
- H-L3: the trade-off between estimation-error margins (robust or conformal) and deadlock rate has not been quantified for learned navigation policies with sparse range sensing.
