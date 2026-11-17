<!-- Draft cover letter for Robotics and Autonomous Systems (primary target). Items in [brackets] must be completed or confirmed by the author before sending. Nothing has been submitted. -->

[Date]

The Editor-in-Chief
*Robotics and Autonomous Systems*

Dear Editor,

Please consider the enclosed manuscript, "Deadlocks of Learned Navigation Policies Behind a Control-Barrier-Function Safety Filter: Mechanisms and Planning-Based Recovery", for publication as a research article in *Robotics and Autonomous Systems*.

Safety filters based on control barrier functions are widely used to make learned robot controllers collision-free. The manuscript studies what they cost in task completion. A filtered robot that is never unsafe can still stall in front of obstacles and fail to reach its goal. We analyse this problem for mobile robot navigation with reinforcement-learning policies, in a simple, fully reproducible simulation.

The main findings are:

1. **Two deadlock mechanisms.** Learned policies deadlock behind the filter in two different ways, depending on how they were trained. Policies trained without the filter are held at the boundary by it; policies trained with it mostly stop on their own before the boundary. A geometric oracle confirms that every test scenario was solvable.
2. **Training changes do not remove the deadlock.** Three pre-registered training interventions removed most self-stops, but the stalls moved to the filter boundary and total deadlock barely changed. One of them is a published correction penalty, whose primary endpoint was null.
3. **Recovery planning removes almost all of it.** A recovery layer that leaves the filter unchanged and plans multi-step manoeuvres at stalls reduced deadlock from 53–59 % to below 1 % in two pre-registered rounds, with no collisions.
4. **The limits are stated as clearly as the gains.** The recovery result depends on a correct global map. About 14 % of episodes still time out. The same planner without the learned policy, given the exact map, performs better.

We think the work fits the journal's interest in robot control and learning for autonomous systems. Its contribution is the diagnosis and the evidence; we do not claim a new filter or a new planner. All results come from simulation, and the manuscript does not claim real-world safety or real-time guarantees.

All protocols and analysis code were committed to a public repository before each confirmatory run. The repository also contains the per-episode results, the trained agents and independent audits of the earlier rounds: https://github.com/pouyapd/SafeNav-RL [archived release DOI to be added].

The manuscript is original, has not been published, and is not under consideration elsewhere. [Confirm: a preprint has / has not been posted.] [Confirm competing interests and funding as stated in the manuscript's declarations.]

Thank you for considering this submission.

Yours sincerely,

Pouya Bathaei Pourmand
[Affiliation, postal address, e-mail, ORCID]
