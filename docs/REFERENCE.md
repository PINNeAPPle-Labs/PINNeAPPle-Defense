# Physics AI for Defense: An Open, Unclassified Reference

*A survey of physics-informed / scientific machine learning applied to physical simulation in the defense domain — academic research, public DoD/DARPA programs, defense-industrial products, numerical methods, benchmark geometries, datasets, standards, and open challenges. Compiled from publicly available, unclassified sources only.*

> **Scope note:** This document intentionally excludes anything classified, export-controlled (ITAR/EAR-restricted technical data), or not independently verifiable in open sources. Where a claim could not be confirmed, it is flagged as such rather than asserted. This is version 0.1, intended to seed an open-source reference repository.

---

## Table of Contents

1. [Introduction: What "Physics AI" Means in a Defense Context](#1-introduction)
2. [Academic & Public Research Landscape](#2-academic--public-research-landscape)
3. [Model Architectures by Application Domain](#3-model-architectures-by-application-domain)
4. [Commercial & Defense-Industrial Products](#4-commercial--defense-industrial-products)
5. [Numerical Methods & Solvers (the Physics AI Learns From/Against)](#5-numerical-methods--solvers)
6. [Geometry & Mesh Resources](#6-geometry--mesh-resources)
7. [Datasets & Data-Generation Approaches](#7-datasets--data-generation-approaches)
8. [Standards, V&V, and Governance Context](#8-standards-vv-and-governance-context)
9. [Open Challenges & Gaps](#9-open-challenges--gaps)
10. [Roadmap: Building the Open Reference Hub](#10-roadmap-building-the-open-reference-hub)
11. [Sources](#11-sources)

---

## 1. Introduction

"Physics AI" (used interchangeably in the literature with *physics-informed machine learning*, *scientific machine learning* (SciML), and *neural operators*) refers to machine learning models that are constrained, regularized, or architected around governing physical laws — PDEs, conservation laws, boundary conditions — rather than being pure black-box statistical fits. In defense engineering, the pitch is straightforward: high-fidelity physics simulations (CFD for hypersonics, full-wave electromagnetics for radar signature, finite-element blast/structural codes, parabolic-equation acoustics) are extremely expensive — hours to weeks per run on HPC clusters — which throttles design-space exploration, uncertainty quantification, and real-time decision support. Physics AI promises orders-of-magnitude speedups (seconds instead of hours) while retaining physical fidelity and interpretability that pure data-driven surrogates lack, which matters enormously for safety-critical, lethal, and acquisition-audited systems.

This reference is scoped strictly to what defense organizations, contractors, and academic/FFRDC labs have said **publicly**. It does not speculate about classified programs. It treats named public programs (e.g., DARPA AIRA, DARPA CLARA, DARPA ML2P), named public products (e.g., Luminary Cloud's SHIFT models, NVIDIA PhysicsNeMo, Ansys Government Initiatives), and peer-reviewed/preprint literature as the evidentiary base.

---

## 2. Academic & Public Research Landscape

### 2.1 Federal laboratories and FFRDCs

- **AFRL (Air Force Research Laboratory)** — co-sponsor with the Australian Defence Science and Technology Group of the **HIFiRE** (Hypersonic International Flight Research Experimentation) program, whose flight geometries (HIFiRE-1 through HIFiRE-8) are among the most heavily reused public benchmarks in hypersonic ML/surrogate research. AFRL-adjacent academic work includes turbulence-model calibration for HIFiRE-1 using data-driven, machine-learned closure models (arXiv:2112.09681).
- **DEVCOM Army Research Laboratory (ARL)** — organized around eleven core research competencies including *Electromagnetic Spectrum Sciences*, *Sciences of Extreme Materials*, *Terminal Effects*, and *Weapons Sciences*; runs short-term AI/ML research initiatives (e.g., ARL-Central STRIAM openings) and is reviewed periodically by National Academies "Assessment of the DEVCOM Army Research Laboratory" reports (2023, 2024), which discuss AI/ML integration across the lab's physical-science portfolios.
- **Naval Research Laboratory (NRL)** — publishes on parabolic-equation underwater acoustic propagation modeling (a long NRL/Navy legacy area) and electromagnetic-spectrum sciences; specific NRL physics-ML papers are less publicly indexed than AFRL/ARL but the lab is a co-author base for Navy-relevant acoustic PE-model verification literature (e.g., arXiv:1904.05314, "On parabolic-equation model verification for underwater acoustics").
- **Sandia National Laboratories** and **Los Alamos National Laboratory** — joint and independent work on machine-learning interatomic potentials (MLIAP) for materials under extreme pressure/density conditions, developed with the Center for Integrated Nanotechnologies (a shared Sandia/LANL DOE user facility). Sandia has also published on ML-accelerated materials-process simulation (reported ~40,000x speedup for certain alloy/process simulations) and on physics-informed operator regression for extracting data-driven continuum models (arXiv:2009.11992) and physics-informed neural networks for constitutive-model calibration from full-field data (arXiv:2203.16577). These are DOE-lab (not DoD-lab) efforts but are foundational to the extreme-conditions materials modeling that feeds into warhead-physics and hypersonic thermal-protection research in the open literature.
- **MIT Lincoln Laboratory** — an FFRDC sponsored by the Department of the Air Force/DoD; publicly describes generating training data from physics-based simulations and augmenting it for realistic operational variation, and explicitly encoding physical constraints (e.g., energy conservation) into ML architectures to improve data efficiency and robustness "to close the technology gap between research and operational needs in missile defense applications." This is one of the more direct public FFRDC statements connecting physics-informed ML to a missile-defense mission area.

### 2.2 Landmark and representative public papers

- Field predictions of hypersonic cones using physics-informed neural networks (physics-embedding CNN/GNN architectures — "PE-CNN"/"PE-GNN") — ResearchGate/AIAA literature.
- "Calibrating hypersonic turbulence flow models with the HIFiRE-1 experiment using data-driven machine-learned models" (arXiv:2112.09681).
- "Fusion-DeepONet: A data-efficient neural operator for geometry-dependent hypersonic and supersonic flows" (arXiv:2501.01934; ScienceDirect/Journal of Computational Physics) — a DeepONet variant explicitly targeting 3D waverider geometries, inferring shear-stress and heat-flux distributions at unseen angles of attack.
- "Machine Learning based reduced models for the aerothermodynamic and aerodynamic wall quantities in hypersonic rarefied conditions" (ScienceDirect, 2022) — Kriging and ANN surrogates for stagnation-point pressure/heat-flux coefficients.
- "Surrogate Models for Hypersonic Aerothermodynamics and Aerodynamics using Gaussian Process Regression" (AIAA SciTech 2024 paper, arc.aiaa.org).
- "Hypernetwork Based Surrogate Modeling of Hypersonic Glide Vehicle Aerothermodynamics" (AIAA Aviation/ASCEND 2024) — explicitly targets glide-vehicle (i.e., boost-glide weapon-adjacent) aerothermodynamics in the open literature.
- "Deep operator learning-based surrogate models for aerothermodynamic analysis of AEDC hypersonic waverider" (arXiv:2405.13234) — trained against Arnold Engineering Development Center (AEDC) Hypervelocity Wind Tunnel 9 data.
- Scattering-center-guided RCS prediction ("SC-RNet"), wideband RCS via Method-of-Moments-trained networks ("WBRCS-Net"), and CNN-LSTM/Transformer architectures for flexible/deforming-target RCS prediction — recent (2024–2025) peer-reviewed EM/radar literature (PubMed, MDPI Electronics, Applied Sciences).
- "RCSnet" — 1D CNN using RCS time series to discriminate warhead vs. decoy targets of similar shape in midcourse flight (IntechOpen) — a rare open-literature example of ML applied to a discrimination problem with direct missile-defense relevance.
- Physics-informed deep operator networks for 2D time-domain EM wave propagation (PMC/PubMed, 2025) and symmetry-informed deep learning for EM scattering (arXiv:2607.12810) — both claim substantial (order-of-magnitude) data-efficiency gains from embedding physical/symmetry priors.
- "A parabolic-equation-based physics-informed machine learning method for underwater sound propagation modeling" (Journal of the Acoustical Society of America, JASA) and "Research on intelligent predicting method of underwater acoustic field based on physics-informed neural network" (Frontiers in Marine Science, 2025).
- "SPECTRA: A Physics-informed Digital Twin for Real-time Structural Anomaly Inference under Operational Variability" (arXiv:2607.03446) and the broader "Machine Learning for Structural Health Monitoring of Aerospace Structures: A Review" (PMC/MDPI Sensors, 2025) — both directly relevant to aircraft airframe sustainment.
- NASA/USAF joint work on airframe digital twins: a "cradle-to-grave" airframe digital twin concept using Bayesian updating of service/fleet experience data to reduce structural uncertainty (referenced across SHM/digital-twin review literature).

### 2.3 Key venues

- **AIAA SciTech Forum** (January, Orlando) — the dominant U.S. venue for hypersonics + scientific-ML crossover work; SciTech 2026 (Jan 12–16, Orlando) featured ~3,000 technical presentations across 65+ topic tracks including hypersonics and intelligent/autonomous systems, and was the venue where Luminary Cloud announced its defense-focused physics-AI models (see §4).
- **AIAA Aviation Forum**, **AIAA ASCEND**, **Journal of Thermophysics and Heat Transfer**, **AIAA Journal** — recurring venues for hypersonic surrogate-modeling papers (e.g., "Surrogate Modeling of Aerodynamic Simulations for Multiple Operating Conditions Using Machine Learning," AIAA Journal).
- **Journal of the Acoustical Society of America (JASA)** — underwater-acoustics PINN work.
- **IET Microwaves, Antennas & Propagation** and **IEEE Antennas and Propagation Magazine** — computational-EM/RCS benchmark and ML-surrogate literature; the canonical RCS benchmark-target reference paper ("Benchmark Radar Targets for the Validation of Computational Electromagnetics Programs," IEEE AP Magazine, vol. 35, no. 1, Feb. 1993) still anchors the field.
- **Physics of Fluids (AIP)** special topic issue: "Scientific Machine Learning and Physics-Informed AI for Multiscale and Multiphysics Fluid Mechanics."

### 2.4 DARPA programs relevant to physics AI / scientific ML

- **AIRA (Artificial Intelligence Research Associate)** — DARPA program (Defense Sciences Office lineage) explicitly aimed at "Third Wave AI": systems robust to sparse/adversarial data that incorporate domain (physics/math/context) knowledge. Objectives: (1) new AI architectures to discover physical laws/models governing complex phenomena, and (2) methods to detect when data are too sparse/noisy to build predictive models, generate testable hypotheses, identify high-value experiments, and quantify out-of-distribution confidence. This is the most directly "physics-discovery-oriented" DARPA AI program publicly described.
- **ML2P (Mapping Machine Learning to Physics)** — newer (2025–2026) DARPA program mapping ML model performance to physical/electrical characteristics (energy use in joules), aimed at energy-aware ML for battlefield/edge deployment; ~$5.9M program budget across performers, Phase 1 ~$3.5M.
- **CLARA (Compositional Learning-And-Reasoning for AI Complex Systems Engineering)** — DARPA Defense Sciences Office program (solicitation DARPA-PA-25-07-02) fusing machine learning with automated reasoning to produce "high-assurance AI" for complex systems-of-systems engineering; up to $2M/award over 24 months across two technical areas (new ML/AR composition methods; a validated open-source software composition library). All software deliverables must use permissive open-source licenses — directly relevant to an open-reference-repo effort, since CLARA-funded tooling may itself become citable open source.
- DARPA's Defense Sciences Office (DSO) office-wide BAA explicitly lists "physics-informed machine learning" as a technical interest area under its "Math, Computation, and Processing" and "Complex, Dynamic, and Intelligent Systems" thrusts, alongside SABER (adversarial robustness for battlefield AI) and MATHBAC (multi-agent coordination theory) — both adjacent but not physics-simulation-specific.

---

## 3. Model Architectures by Application Domain

### 3.1 Hypersonic aerothermodynamics
- **Physics-informed neural networks (PINNs)**: embed compressible Navier–Stokes / Euler residuals, shock-capturing terms, and boundary conditions directly in the loss function; used for field prediction around hypersonic cones and for inverse problems (e.g., inferring freestream conditions from sparse measurements) in supersonic/hypersonic flow (arXiv:2202.11821).
- **Neural operators (DeepONet, Fourier Neural Operator/FNO)**: learn mappings between function spaces (geometry/boundary-condition → flow field) rather than single input-output points, enabling generalization across a family of shapes/conditions without retraining. **Fusion-DeepONet** is the most concrete public example targeting geometry-dependent hypersonic/supersonic flow on arbitrary grids, including 3D waverider surfaces.
- **Gaussian Process Regression (GPR)/Kriging surrogates**: still widely used for hypersonic aerothermodynamic and aero-thermo-servo-elastic surrogate modeling where uncertainty quantification (predictive variance) is a first-class requirement, e.g., in guidance/control-loop-relevant hypersonic flight dynamic analysis.
- **Hypernetworks**: recent (2024) approach generating CFD-surrogate network weights conditioned on freestream parameters (Mach, AoA, altitude) for hypersonic glide-vehicle aerothermodynamics.
- **Physics-constrained neural collision operators**: used as surrogates inside Direct Simulation Monte Carlo (DSMC) codes for rarefied hypersonic flow, replacing expensive hard-sphere collision physics with learned, physically constrained operators.

### 3.2 Radar cross-section (RCS) / electromagnetic scattering
- **Scattering-center-guided networks** (e.g., SC-RNet): decouple geometric features into coarse structural shape and fine surface detail, informed by classical scattering-center theory rather than treating RCS prediction as pure image regression.
- **Method-of-Moments-trained surrogate networks** (e.g., WBRCS-Net): trained to converge toward MoM-formulation solutions across wide frequency bands and incidence angles, effectively distilling a classical full-wave solver into a fast network.
- **CNN-LSTM/Transformer hybrids**: used for RCS prediction of flexible/deforming targets, separating static geometric features from dynamic deformation time series.
- **Physics-informed deep operator networks (PI-DeepONet)** and **symmetry-informed architectures**: embed Maxwell's-equations residuals or physical symmetries (rather than raw MoM data-fitting) directly into training, reported to achieve <1% relative error vs. FDTD references and up to an order-of-magnitude improvement in data efficiency.
- **Transformer/GNN EM-dynamics surrogates**: trained on FDTD simulations of plane-wave scattering, reported ~14x inference speedup vs. FDTD in one published study.
- **RCSnet**: 1D CNN discriminating same-shape warhead vs. decoy targets from RCS time series — a rare open example of ML for missile-defense discrimination logic (mid-course target discrimination), published in the open literature (IntechOpen).

### 3.3 Underwater acoustics / sonar
- **Parabolic-equation (PE) physics-informed ML**: hybridizes the classical PE approach to the Helmholtz/wave equation (the workhorse of Navy sound-propagation modeling) with neural-network approximators, published in JASA (2025) and Frontiers in Marine Science (2025).
- **ML for sound-speed field estimation**: combining satellite remote sensing with ML to predict 3D underwater sound-speed fields at ocean-basin scale (Nature Communications Engineering, 2025) — relevant to sonar performance prediction and ASW mission planning.
- **Target recognition/classification**: CNN, RNN, deep Kalman filter, and transformer-based classifiers for passive-sonar target detection/ID/localization (reviewed in Frontiers in Physics 2022 and Journal of Field Robotics 2026 systematic reviews).
- **Physics-informed synthetic sonar image generation**: a framework for statistically validated, physics-informed synthetic sonar imagery (arXiv:2605.19712) — directly relevant to the "synthetic-data-for-classified-adjacent-domains" theme (see §9).

### 3.4 Blast/warhead physics and structural dynamics (open literature only)
- Explainable ML-based prediction of blast loads on structural surfaces (2D spatial coordinates), ScienceDirect 2025 — surrogate replacing costly blast-load CFD/FE mapping.
- Surrogate models for dynamic load factor and ductility ratio in single-degree-of-freedom (SDOF) blast-response analysis (Journal of Vibration Engineering & Technologies, 2023).
- LS-DYNA remains the reference high-fidelity solver (Multi-Material Arbitrary Lagrangian-Eulerian / MM-ALE and Load Blast Enhanced methods) that these ML surrogates are trained against or benchmarked to; no open-source, DoD-published "warhead effects" ML dataset was identified — this area is unsurprisingly thin in open literature relative to hypersonics or EM, consistent with its sensitivity (see §9 on classified-data constraints).
- Automotive crash-dynamics ML acceleration (arXiv:2510.15201) is cited by analogy in some defense-adjacent structural-ML literature as a dual-use methodology transfer path (crash/impact FE surrogate methods generalize to blast/impact surrogate methods without touching sensitive lethality data).

### 3.5 Digital twins for platform sustainment / structural health monitoring
- **SPECTRA** (arXiv:2607.03446): physics-informed digital twin for real-time structural anomaly inference under operational variability — architecture combines a physics-informed model core with a "virtual twin of the sensing network" to monitor and quantify measurement reliability in real time.
- NASA/USAF airframe digital-twin concept: Bayesian-updated, "cradle-to-grave" digital twins that reduce structural life-prediction uncertainty by continuously assimilating fleet service data — widely cited as the seminal USAF/NASA digital-twin-for-airframes concept in SHM review literature.
- General pattern across the 2025 MDPI Sensors / PMC review of aerospace SHM: "gray-box" physics-informed ML (hybridizing physics-based models with data-driven learning) is favored specifically because experimental damage data are scarce for airframes — a direct data-scarcity parallel to the classified-data problem discussed in §9.

---

## 4. Commercial & Defense-Industrial Products

*(All entries below reflect only what each organization has stated publicly; capability claims are the companies' own marketing/press language unless otherwise noted, and should be read accordingly.)*

- **Lockheed Martin** — **Astris AI** is Lockheed's newly branded (2025) secure, end-to-end "AI Factory" platform (built on NVIDIA DGX SuperPOD infrastructure) spanning edge, on-prem, and cloud deployment; reportedly used by ~7,000 Lockheed engineers/developers, producing large numbers of automated pipelines and API calls weekly. It hosts the **Lockheed Martin AI Center (LAIC)** and Genesis/Navigator agent-development frameworks. Separately, Lockheed's **ARISE** is a toolkit family for building system-level weapon-simulation "digital twins" (used across Missiles and Fire Control programs), and Lockheed has publicly discussed an AI knowledge agent integrated with the **Joint Simulation Environment (JSE)** (a joint USAF/Navy high-fidelity virtual-battlefield environment for evaluating new fighter capabilities) that "understands the simulation's physics" and can spin up test scenarios on demand. Lockheed has also announced (with NVIDIA) an AI-driven digital-twin demonstration aimed at predictive forecasting, and discusses structural digital twins for F-35 manufacturing optimization and a conceptual "e-Pilot" human digital twin.
- **Ansys Government Initiatives (AGI)** — the U.S. national-security subsidiary of Ansys (defense-community partner since 1989); publicly states it is developing "physics-informed AI/ML models trained on high-fidelity synthetic datasets" for mission-critical systems in extreme environments, spanning high-speed/hypersonics, avionics, autonomous systems, and electronic warfare. AGI holds an $82.1M sole-source USAF contract (Edwards AFB, FY26–FY31) for modeling/simulation software and engineering services, and was selected to join the Microelectronics Commons national-security consortium (Nov. 2024).
- **NVIDIA** — **PhysicsNeMo** (formerly "Modulus"), NVIDIA's open-source physics-ML framework, is publicly available on GitHub (`NVIDIA/physicsnemo`) and supports FNO, DeepONet, graph-network, and MLP architectures across fluid dynamics, structural mechanics, electromagnetics, and acoustics domains, explicitly targeting "scientific digital twins" with near-real-time inference. NVIDIA's defense-specific applications are less directly documented in public sources beyond the Lockheed DGX SuperPOD AI Factory partnership noted above; PhysicsNeMo itself is dual-use, general-purpose infrastructure rather than a defense-labeled product.
- **Luminary Cloud** — the clearest and most recent (January 2026, at AIAA SciTech) public example of a company explicitly branding "Physics AI" for defense. Announced three defense-focused **SHIFT** models: **SHIFT-CCA** (Collaborative Combat Aircraft), **SHIFT-Submarine**, and **SHIFT-Pump**, produced via its "Physics AI Factory" — described as a secure, U.S.-based, sovereign pipeline unifying GPU-native physics simulation, large-scale synthetic-data generation, AI model training, and deployment/inference. Luminary states these models return CFD-class answers in "less than a second" versus hours for traditional simulation, positioned as complementary to (not a replacement for) simulation. CTO/co-founder Juan Alonso is the named public spokesperson. This is a directly quotable, current (2026) example of commercial "Physics AI for defense" branding.
- **Northrop Grumman / RTX (Raytheon)** — publicly discuss digital engineering and Model-Based Systems Engineering (MBSE) collaboration; a September 2024 joint Northrop/Raytheon demonstration of a "wired end-burning" solid rocket motor for hypersonics explicitly used combined digital-engineering/MBSE workflows and generated data to validate design maturity via digital modeling and simulation prior to hardware testing.
- **Boeing Defense** — publicly a partner (with University of Queensland and AFRL/DSTG) on the HIFiRE hypersonic flight-research program (§2.1, §6); specific public statements on physics-AI/ML tooling for Boeing Defense programs were not identified in this pass and should be flagged as a research gap for future updates.
- **Anduril Industries** — publicly describes its **Lattice** AI operating system as the unifying software layer across its autonomous air/land/sea systems, focused on real-time sensor fusion and decision support at the tactical edge; public materials emphasize autonomy/C2 software rather than physics-simulation-specific ML, though Anduril's Fury aircraft has been reported to integrate Shield AI's Hivemind autonomy stack (see below).
- **Shield AI** — states its **Hivemind** autonomous-flight system is trained in "synthetic, physics-modeled environments" combined with human-machine tactics/behavior data; publicly partnered with Palantir (Hivemind integrated with Palantir's **Gaia** environment for simulate/train/deploy-at-scale workflows) and reportedly integrated into Anduril's Fury platform.
- **Palantir** — positioned primarily as a data-fusion/decision-support (AIP, Gaia, Project Maven integration) company rather than a physics-simulation company; its publicly stated defense role is compute/data orchestration and human-AI decision workflows layered atop partners' physics-modeled simulation environments (e.g., the Shield AI Hivemind/Gaia integration) rather than first-party physics-ML model development.
- **Hermeus** — developing the **Quarterhorse** hypersonic/high-speed test-aircraft series (Mk 1 flown; Mk 2.1 roughly F-16-sized, Pratt & Whitney F100-powered) as a stepping stone to **Darkhorse**, a reusable Mach 5+ hypersonic aircraft aimed at defense/national-security missions; publicly emphasizes rapid iterative flight-test-driven development (Quarterhorse Mk 1: ~19 months drawing-board-to-flight) rather than a named "physics-AI" toolchain, though this iteration speed is widely cited industry-wide as a digital-engineering success story.
- **Stratolaunch** — operates **Talon-A**, the only U.S. autonomous, reusable, air-launched hypersonic flight-test platform, with 10+ successful hypersonic flights reported as of mid-2025 and DoD interest in reusing the testbed for faster iterative testing; public materials describe post-flight data analysis against pre-flight predictions but do not detail specific AI/ML methods used, which should be flagged as unconfirmed rather than assumed.
- **DoD Chief Digital and AI Office (CDAO)** — in August 2025, CDAO was realigned under the Under Secretary of Defense for Research and Engineering (USD(R&E)), reportedly to implement the White House AI Action Plan's DoD-adoption directive; CDAO's public mission historically spans enterprise AI adoption, data governance, and (via its predecessor JAIC) the DoD AI Ethical Principles (see §8).

---

## 5. Numerical Methods & Solvers

These are the classical, high-fidelity tools that physics-AI surrogates are trained against, benchmarked to, or hybridized with — essential context because every ML surrogate paper in this space defines its own accuracy relative to one of these.

### 5.1 Hypersonic CFD
- **US3D** — parallel, unstructured-mesh, implicit Navier–Stokes solver (University of Minnesota origin), widely used for hypersonic reentry/flight simulation; shown in NASA-published comparisons to agree well with the structured-grid codes DPLR and LAURA.
- **DPLR (Data-Parallel Line Relaxation)** — NASA Ames structured-grid hypersonic nonequilibrium aerothermodynamics code; supports Baldwin-Lomax and Shear-Stress-Transport (SST) turbulence models.
- **LAURA (Langley Aerothermodynamic Upwind Relaxation Algorithm)** — NASA Langley structured-grid code, uses the Cebeci-Smith algebraic turbulence model; both DPLR and LAURA are validated extensively for thermal-protection-system design via Apollo-era flight-test cases like FIRE II.
- These three codes are the standard public cross-validation triad for hypersonic nonequilibrium aerothermodynamics code verification (see the "FIRE II calculations..." comparison study).

### 5.2 Computational electromagnetics (CEM)
- **Method of Moments (MoM)** — the default/reference full-wave solver approach in tools like FEKO; discretizes geometry into elements (typically triangles), applies basis functions, and solves the resulting linear system — the classical baseline that most ML RCS surrogates target or are validated against.
- **Multilevel Fast Multipole Method (MLFMM)**, **Finite Element Method (FEM)**, **FDTD**, **Physical Optics/Large-Element PO**, **Ray-Launching Geometrical Optics (RL-GO)**, **UTD** — additional solver methods bundled in commercial CEM suites (FEKO, CST Studio Suite) spanning the low-to-high-frequency / low-to-high-fidelity tradeoff space for RCS, antenna, EMC, and radome problems.
- **FEKO** (Altair) and **CST Studio Suite** (Dassault Systèmes) are the two dominant commercial CEM packages referenced across the RCS/EM-surrogate literature as ground-truth generators for training data.

### 5.3 Underwater acoustics
- **Parabolic Equation (PE) methods** — the dominant approach for range-dependent ocean sound propagation, approximating the wave/Helmholtz equation as a one-way parabolic PDE; the U.S. Navy operates a range-dependent PE acoustic propagation model integrated with Navy/NOAA environmental databases. PE models handle range-dependent bathymetry/sound-speed profiles well but are generally limited to lower frequencies as computational cost rises with frequency — the exact tradeoff physics-informed ML approaches (§3.3) are now targeting.
- Other classical model families referenced in the open literature: ray models, normal-mode models, and coupled-mode models — all typically approximating solutions to the Helmholtz equation under varying boundary-condition assumptions.

### 5.4 Structural dynamics / blast & impact
- **LS-DYNA** (Ansys/Livermore Software Technology origin) — the standard explicit FE solver for blast and impact structural dynamics in the open literature; blast loading is commonly modeled via the **Multi-Material Arbitrary Lagrangian-Eulerian (MM-ALE)** method or the computationally cheaper **Load Blast Enhanced (LBE)** empirical/semi-empirical method, both referenced repeatedly as the ground truth for ML blast-load surrogate papers.

---

## 6. Geometry & Mesh Resources

- **HIFiRE geometries** — HIFiRE-1 (a well-documented, simple axisymmetric cone-cylinder-flare geometry: 1721.7 mm overall length, 409.2 mm diameter, 7° half-angle conical forebody, 2.5 mm nose bluntness radius, 400 mm cylindrical midbody, 33° flare aftbody) is the most widely reused public hypersonic benchmark geometry in ML surrogate papers because its dimensions and flight-test data are fully published. HIFiRE-4 used an osculating-cone waverider design for its fuselage, flight-tested at Mach 8 (July 2017). HIFiRE-5 geometry and flight-test-results papers are also public via NASA NTRS and AIAA.
- **Waverider shapes generally** — the "osculating cone" design method is the standard public technique for generating hypersonic waverider forebody geometries (shock-attached leading edges maximizing lift-to-drag via captured compression); used as the geometric basis in Fusion-DeepONet and related neural-operator surrogate papers.
- **NASA almond** — the canonical benchmark shape for evaluating high-frequency EM/RCS solvers; used ubiquitously to validate MoM, MLFMM, shooting-bouncing-ray (SBR), and now ML-surrogate RCS codes against each other.
- **Ogive, double-ogive, cone-sphere, cone-sphere-with-gap, sphere, squat cylinder, squat dihedral** — the standard set of "simple shapes" for RCS benchmarking, with both simulated and empirically measured (700 MHz–16 GHz) RCS data available; the canonical reference is "Benchmark Radar Targets for the Validation of Computational Electromagnetics Programs," *IEEE Antennas and Propagation Magazine*, vol. 35, no. 1, Feb. 1993 — still the field's anchor citation.
- **AEDC waverider geometry** — a public hypersonic waverider geometry tested at AEDC Hypervelocity Wind Tunnel 9 has been used directly as a neural-operator surrogate-modeling case study (arXiv:2405.13234), making it a rehydrated public benchmark alongside HIFiRE shapes.

---

## 7. Datasets & Data-Generation Approaches

- **NASA/AEDC wind-tunnel data**: AEDC's **Hypervelocity Wind Tunnel 9** (operated for the U.S. Air Force by National Aerospace Solutions) generates high-Mach, high-Reynolds-number hypersonic ground-test data; specific experimental datasets from Tunnel 9 have been used to train neural-operator surrogates for flow-field inference around tested geometries, though a single, centralized, openly downloadable AEDC dataset repository was **not** identified in this pass — data reuse in the literature appears to happen paper-by-paper via research collaborations rather than through a standing open data portal. This should be flagged as a gap/opportunity (see §10).
- **NASA NTRS (NASA Technical Reports Server)** — a genuinely open, searchable archive of NASA technical reports including HIFiRE design/flight-test documentation, CFD-code verification studies (DPLR/LAURA/US3D comparisons), and historical RCS measurement/simulation reports (e.g., NASA TM-4613, "Radar Cross-Section Measurements and Simulation"). This is one of the most valuable existing open resources for this domain and should be a first-class linked resource in the target repo.
- **DTIC (Defense Technical Information Center)** — hosts historical unclassified/public reports such as AEDC-TR-94-6 ("Hypersonic Wind Tunnel Test Techniques"); a key source of legacy public hypersonic ground-test methodology documentation.
- **Synthetic data generation as the field's primary workaround for data scarcity**: across nearly every domain surveyed (hypersonics, RCS, underwater acoustics, blast/structural), the literature converges on the same pattern — because real flight-test, measured-RCS, or classified-adjacent data are scarce/restricted, teams generate large synthetic datasets from high-fidelity physics solvers (CFD, MoM/FDTD, PE-acoustics, LS-DYNA) and train ML surrogates or PINNs against that synthetic corpus, often with physics constraints embedded specifically to reduce the *amount* of synthetic data needed (an explicit stated motivation in the symmetry-informed EM scattering work and the PI-DeepONet EM papers, both reporting order-of-magnitude data-efficiency gains from physics priors). Luminary Cloud's "Physics AI Factory" and MIT Lincoln Laboratory's stated approach (generating training data from physics-based simulations, then augmenting for operational realism, explicitly for missile-defense-relevant ML) are the two most concrete public statements of this synthetic-data-first strategy as deliberate defense-sector practice.
- **Physics-informed synthetic sonar image generation with statistical validation** (arXiv:2605.19712) is a directly on-point recent example of the field formalizing "how do we validate that our synthetic data is statistically representative of the real, restricted-access domain" — a methodological template other physics-AI-for-defense subfields could adopt.

---

## 8. Standards, V&V, and Governance Context

### 8.1 Verification & Validation standards
- **ASME V&V 10** — "Standard for Verification and Validation in Computational Solid Mechanics"; provides a common conceptual framework and language for model verification, validation, and uncertainty quantification (VVUQ) in solid-mechanics simulation, referencing DoD Directive 5000.61 (DoD Modeling & Simulation Verification, Validation, and Accreditation).
- **ASME V&V 20** — "Standard for Verification and Validation in Computational Fluid Dynamics and Heat Transfer"; specifies an approach for quantifying accuracy by comparing computed solutions to experimental data, incorporating experimental-uncertainty-analysis concepts for both simulation and measurement error. Both standards were developed with DoD, DOE, and FAA participation, making them the natural common language for certifying physics-AI surrogates that stand in for classical CFD/FE solvers in defense acquisition contexts.
- **DoD Directive 5000.61** — DoD's own Modeling and Simulation Verification, Validation, and Accreditation (VV&A) policy, cross-referenced by ASME V&V 10; the accreditation step (as opposed to just verification/validation) is the DoD-specific addition — an authority-to-use decision made by a program/system owner, distinct from the purely technical V&V process. This VV&A-vs-V&V distinction is an important nuance for any physics-AI-for-defense repo to get right, since a technically "validated" surrogate model still requires programmatic accreditation before use in an acquisition decision.
- The DoD's Modeling and Simulation Coordination Office (M&S CO) historically issued cross-service guidance on M&S policy and VV&A; current organizational status/URL should be re-verified in a future pass, as its function may now sit under the digital-engineering/CDAO/USD(R&E) reorganization described below.

### 8.2 Digital Engineering Strategy and acquisition policy
- **2018 DoD Digital Engineering Strategy** — the foundational document defining a digital twin as "an integrated, multi-physics, multi-scale, probabilistic simulation of an as-built system, enabled by a digital thread."
- **DoD Instruction 5000.97, "Digital Engineering"** (effective Dec. 21, 2023, approved by USD(R&E) Heidi Shyu) — the current binding policy: directs new acquisition programs to incorporate digital engineering (digital models, digital threads, digital twins) unless the program's decision authority grants an exception, and directs existing programs to adopt it "to the maximum extent possible, when practical, beneficial, and affordable." This is the operative regulatory hook that makes physics-AI/digital-twin adoption a compliance-relevant (not just performance-relevant) topic for DoD program offices.
- Publicly cited highest-value defense digital-twin use cases: platform sustainment/predictive maintenance, system design/engineering trade studies, mission rehearsal/training, supply-chain/logistics modeling, cyber-resilience testing, and intelligence/operational planning.
- **CDAO realignment (August 2025)** — Deputy Secretary of Defense memo (Aug. 14, 2025) placed the Chief Digital and AI Office under USD(R&E), explicitly linked to implementing the White House AI Action Plan's DoD AI-adoption directive — a live organizational signal that AI and digital-engineering governance are being consolidated under the same acquisition-technical authority, relevant to where a "physics AI for defense" community should expect policy leadership to sit going forward.

### 8.3 Responsible/ethical AI
- **DoD AI Ethical Principles** (adopted 2020, originating from the Joint Artificial Intelligence Center, JAIC, stood up 2018): five principles — **Responsible**, **Equitable**, **Traceable**, **Reliable**, **Governable**. "Traceable" is the principle most directly relevant to physics-AI surrogates: it requires that relevant personnel possess an appropriate understanding of the technology, development process, and operational methods, "including transparent and auditable methodologies, data sources, and design procedures and documentation" — effectively a documentation/provenance requirement that maps naturally onto V&V/accreditation packages for physics-AI surrogate models.
- Public reporting (FedScoop and others) has noted that DoD AI ethics principles have historically lacked detailed implementation guidance for contractors and program offices, an open governance gap independent of the specific physics-AI use case.

---

## 9. Open Challenges & Gaps

1. **Classified/restricted data driving a synthetic-data-first field.** Nearly every subdomain surveyed (hypersonics, RCS, blast, underwater acoustics) shows the same adaptation: because real flight-test, measured-signature, or lethality data are scarce, export-controlled, or classified, the open research base trains almost entirely on synthetic data generated from classical high-fidelity solvers (CFD, MoM/FDTD, PE-acoustics, LS-DYNA). This is efficient for methodology development but creates a structural sim-to-real gap between what's published openly and what's actually fielded, and makes public benchmark diversity (HIFiRE, NASA almond, canonical RCS shapes) disproportionately important as the *only* shared ground truth the open community has.
2. **Verification & Validation for safety-critical/lethal systems.** ASME V&V 10/20 give a technical VVUQ framework, but DoD's VV&A **accreditation** step is a separate, program-specific authority-to-use decision. There is no publicly documented, standardized VV&A pathway specifically for *learned* (ML/neural-operator) surrogate models as opposed to classical solvers — accrediting a black-box (or even physics-informed) neural network for use in a targeting-adjacent or flight-critical decision loop is an open methodological and policy problem, not just an engineering one.
3. **Sim-to-real gap.** Broadly documented in the ML literature as having two components — an "appearance gap" (synthetic vs. real data distributions differ) and a "content gap" (synthetic data lacks real-world diversity). Multi-fidelity approaches (e.g., "SIM-FSVGD"-style methods that regularize neural-network training using low-fidelity physics priors) are an active mitigation direction but remain early-stage in the defense-specific literature reviewed here.
4. **Multi-fidelity fusion.** Combining cheap/low-fidelity simulation (empirical/engineering-level models), medium-fidelity CFD/CEM, and scarce high-fidelity/experimental data into a single coherent surrogate remains a hard, actively researched problem — relevant across hypersonics (rarefied-flow vs. continuum regimes), RCS (near-field vs. far-field transforms), and structural health monitoring (physics model vs. sparse sensor data).
5. **Explainability for military decision support.** The DoD's "Traceable" AI ethics principle explicitly requires auditable methodology and documented data provenance, which is in tension with the "black-box-but-fast" nature of many neural-operator/PINN surrogates; physics-informed architectures are partly motivated by this tension (a network constrained to satisfy Maxwell's equations or Navier-Stokes residuals is inherently more auditable than a pure data-fit), but formal explainability requirements specific to physics-AI surrogates (as opposed to LLM-based decision-support tools, which dominate current "responsible AI" discourse) are not yet clearly codified in public DoD guidance.
6. **Data/benchmark fragmentation.** Unlike, say, ImageNet in computer vision, there is no single, actively maintained, open benchmark suite for defense-relevant physics-AI tasks. HIFiRE geometries and the canonical RCS shape set function as de facto benchmarks purely because they were published decades ago and are geometrically simple enough to be unclassified — not because a governing body curated them for ML benchmarking purposes. This is arguably the clearest concrete gap an open-source reference repo could help close (see §10).
7. **Warhead/blast-physics ML remains the thinnest open-literature area** among the topics surveyed, consistent with its inherent sensitivity — dual-use methodology transfer from automotive crash-dynamics ML is the most visible workaround in open sources, but there is comparatively little defense-labeled open blast-surrogate research relative to hypersonics or EM/RCS.

---

## 10. Roadmap: Building the Open Reference Hub

**Guiding constraint:** every deliverable below stays unclassified, open-research, and dual-use-responsible. The repo should explicitly document what it will *not* host (classified data, export-controlled technical data, weapon-effectiveness/lethality data, anything requiring ITAR/EAR review) as a standing policy, not an afterthought.

### Phase 1 (0–3 months): Foundation
- Publish this reference document as the repo's `README`/wiki root, with the Sources section converted into a structured, filterable bibliography (BibTeX + linked markdown).
- Stand up a curated **benchmark-geometry index**: HIFiRE-1 through -8 (with public dimensions/flight-test citations), NASA almond, ogive/double-ogive/cone-sphere RCS shapes, and the AEDC public waverider case — each with links to the primary NASA NTRS / AIAA / IEEE source and, where license permits, redistributable mesh files or generation scripts (e.g., a script to regenerate the HIFiRE-1 axisymmetric geometry from its published dimensions).
- Stand up a **program/organization tracker**: a maintained table of public DARPA programs (AIRA, ML2P, CLARA, DSO BAA thrusts), DoD digital-engineering policy documents (5000.97, the 2018 Digital Engineering Strategy), and named commercial physics-AI-for-defense products (Astris AI, AGI, PhysicsNeMo, Luminary SHIFT models), each with a one-paragraph unclassified summary and primary-source link, updated on a quarterly cadence.

### Phase 2 (3–9 months): Depth and community
- Recruit contributors from the open academic base already publishing in this space (authors of the arXiv/AIAA papers cited above are a natural initial outreach list) to write domain-specific deep-dive pages: hypersonic neural operators, RCS/EM surrogates, underwater-acoustics PINNs, SHM/digital-twin architectures.
- Build a **reproducible-baseline** track: for at least one benchmark per domain (e.g., HIFiRE-1 heat-flux prediction; NASA-almond RCS via open-source MoM), publish an open-source reference implementation (PyTorch/JAX, using open frameworks like NVIDIA PhysicsNeMo or DeepXDE) that anyone can run end-to-end on public data, so the repo has at least one fully reproducible "hello world" per subdomain rather than just literature summaries.
- Add a **V&V/standards explainer track**: plain-language guides to ASME V&V 10/20, DoDI 5000.97, and DoD VV&A, specifically addressing "what does this mean for accrediting a learned surrogate model" — an area with essentially no existing public plain-language treatment, and a genuine value-add opportunity rather than a restatement of existing resources.
- Establish a lightweight editorial/contribution policy explicitly modeled on the "public, cited, unclassified only" constraint, including a checklist contributors must apply before submitting anything (public source only; no ITAR/EAR technical data; no unpublished internal program details even if "not technically classified").

### Phase 3 (9–36 months): Becoming the reference
- Pursue light-touch relationships (not funding dependence) with organizations already active in the open space — e.g., citing/tracking DARPA CLARA's mandatory open-source deliverables as they land, linking NASA NTRS updates, and inviting FFRDC-adjacent academic authors (AFRL/DEVCOM/NRL-sponsored university labs, MIT Lincoln Lab public-facing researchers) to co-author survey updates.
- Run or co-sponsor a small annual "state of physics AI for defense" survey update (a living document, versioned like this one), timed to follow AIAA SciTech each January so the repo captures that year's new public program/product announcements while they're fresh (as this document did with the January 2026 Luminary Cloud SHIFT announcement).
- Expand the benchmark track toward a genuine shared leaderboard/dataset-card system (dataset cards documenting provenance, synthetic-vs-real fidelity, and known sim-to-real limitations for each public dataset used) — directly addressing the benchmark-fragmentation gap identified in §9.
- Maintain explicit, visible scope discipline: as the repo gains visibility, the highest risk is scope creep into classified-adjacent or export-controlled territory chasing relevance; a standing, publicly documented editorial policy and (as the project matures) an informal advisory review step for borderline submissions should be treated as core infrastructure, not bureaucratic overhead.

---

## 11. Sources

**Academic / preprint literature**
- Field Predictions of Hypersonic Cones using Physics-Informed Neural Networks — https://www.researchgate.net/publication/363332780_Field_Predictions_of_Hypersonic_Cones_using_Physics-Informed_Neural_Networks
- Physics-informed neural networks for inverse problems in supersonic flows — https://arxiv.org/pdf/2202.11821
- Calibrating hypersonic turbulence flow models with the HIFiRE-1 experiment using data-driven machine-learned models — https://arxiv.org/pdf/2112.09681
- Fusion DeepONet: A Data-Efficient Neural Operator for Geometry-Dependent Hypersonic Flows on Arbitrary Grids — https://arxiv.org/html/2501.01934v1
- Fusion-DeepONet (ScienceDirect / J. Comp. Physics) — https://sciencedirect.com/science/article/abs/pii/S0021999125007144
- Deep operator learning-based surrogate models for aerothermodynamic analysis of AEDC hypersonic waverider — https://arxiv.org/html/2405.13234v1
- Physics constrained neural collision operators for hard sphere surrogates... (DSMC) — https://pubs.aip.org/aip/pof/article/38/5/057123/3391727
- Machine Learning based reduced models for aerothermodynamic and aerodynamic wall quantities in hypersonic rarefied conditions — https://www.sciencedirect.com/science/article/abs/pii/S0094576522007111
- Surrogate Models for Hypersonic Aerothermodynamics and Aerodynamics using Gaussian Process Regression (AIAA SciTech 2024) — https://arc.aiaa.org/doi/abs/10.2514/6.2024-0461
- Hypernetwork Based Surrogate Modeling of Hypersonic Glide Vehicle Aerothermodynamics — https://arc.aiaa.org/doi/10.2514/6.2024-4107
- Neural Network Approach for Wideband RCS Computation with Wide Incident Angles via Method of Moments — https://doi.org/10.3390/app16052518
- Scattering center guided mono-static radar cross section prediction — https://pubmed.ncbi.nlm.nih.gov/42184466/
- RCS Prediction for Flexible Targets with Uncertain Shape Based on CNN-LSTM — https://www.mdpi.com/2079-9292/14/23/4668
- Deep Learning Network for Classifying Target of Same Shape using RCS Time Series (RCSnet) — https://www.intechopen.com/chapters/78183
- Symmetry-Informed Deep Learning for Electromagnetic Scattering — https://arxiv.org/html/2607.12810
- Physics‐informed surrogates for electromagnetic dynamics using Transformers and graph neural networks — https://ietresearch.onlinelibrary.wiley.com/doi/abs/10.1049/mia2.12463
- Physic-informed deep operator networks for modeling 2D time-domain electromagnetic wave propagation — https://pmc.ncbi.nlm.nih.gov/articles/PMC12964304/
- A parabolic equation-based physics-informed machine learning method for underwater sound propagation modeling (JASA) — https://pubs.aip.org/asa/jasa/article/159/1/906/3378304
- Research on intelligent predicting method of underwater acoustic field based on PINN — https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2025.1665305/full
- Physics-informed simulation framework for realistic sonar image generation and statistical validation — https://arxiv.org/pdf/2605.19712
- Leveraging satellite observations and machine learning for underwater sound speed estimation (Nature Comms Eng.) — https://www.nature.com/articles/s44172-025-00459-6
- Challenges and Advances in Underwater Sonar Systems and AI-Driven Signal Processing for Modern Naval Operations (J. Field Robotics 2026) — https://onlinelibrary.wiley.com/doi/10.1002/rob.70077
- SPECTRA: A Physics-informed Digital Twin for Real-time Structural Anomaly Inference — https://arxiv.org/pdf/2607.03446
- Machine Learning for Structural Health Monitoring of Aerospace Structures: A Review — https://pmc.ncbi.nlm.nih.gov/articles/PMC12526691/ and https://www.mdpi.com/1424-8220/25/19/6136
- Explainable machine learning-based prediction of blast loads on structural surfaces — https://www.sciencedirect.com/science/article/pii/S2590123025010552
- Surrogate models for dynamic load factor and ductility ratio in blast response — https://link.springer.com/article/10.1007/s42417-023-01086-1
- On parabolic-equation model verification for underwater acoustics — https://arxiv.org/pdf/1904.05314
- FIRE II calculations for hypersonic nonequilibrium aerothermodynamics code verification: DPLR, LAURA, and US3D — https://www.researchgate.net/publication/269204875
- Development of the US3D Code for Advanced... (NASA NTRS) — https://ntrs.nasa.gov/api/citations/20180002133/downloads/20180002133.pdf
- On Digital Twins in Defence: Overview and Applications — https://arxiv.org/pdf/2508.05717

**DARPA / DoD program pages**
- AIRA | DARPA — https://www.darpa.mil/research/programs/artificial-intelligence-research-associate
- ML2P | DARPA — https://www.darpa.mil/research/programs/mapping-machine-learning-physics
- Energy-aware machine learning | DARPA — https://www.darpa.mil/news/2025/energy-aware-machine-learning
- CLARA | DARPA — https://www.darpa.mil/research/programs/clara
- CLARA program presentation (PDF) — https://www.darpa.mil/sites/default/files/attachment/2026-02/program-clara-presentation.pdf
- DoD Instruction 5000.97 summary (Digital Engineering) — https://www.cto.mil/wp-content/uploads/2024/05/info-dodi-500097-de-summary.pdf
- DoD Digital Engineering Strategy — https://sebokwiki.org/wiki/DoD_Digital_Engineering_Strategy
- Realignment of DOD's Chief Digital and AI Officer (CDAO) — Congress.gov CRS — https://www.congress.gov/crs-product/IN12615
- DOD Adopts 5 Principles of Artificial Intelligence Ethics — https://www.war.gov/News/News-Stories/Article/Article/2094085/dod-adopts-5-principles-of-artificial-intelligence-ethics/
- Department of Defense Joint Artificial Intelligence Center — AI Ethics Principles & Autonomy (NPS) — https://nps.edu/documents/115559645/125926484/2021+Dist+A+JAIC+AI+Ethics+Principles+%26+Autonomy+(Patel).pdf

**Standards**
- ASME: Verification, Validation and Uncertainty Quantification (VVUQ) — https://www.asme.org/codes-standards/publications-information/verification-validation-uncertainty
- ASME V&V 10 (Computational Solid Mechanics) — https://www.asme.org/codes-standards/find-codes-standards/standard-for-verification-and-validation-in-computational-solid-mechanics
- ASME V&V 20 (CFD and Heat Transfer) — https://www.asme.org/codes-standards/find-codes-standards/standard-for-verification-and-validation-in-computational-fluid-dynamics-and-heat-transfer

**Commercial / defense-industrial**
- Astris AI: Bringing Lockheed Martin's Trusted Innovation to New Industries — https://www.lockheedmartin.com/en-us/news/features/2025/astris-ai-lockheed-martin-trusted-innovation-new-industries.html
- Lockheed Martin's AI Factory: Powered by NVIDIA DGX SuperPOD — https://www.nvidia.com/en-us/case-studies/lockheed-martin-ai-factory-with-dgx-superpod/
- Lockheed's AI Factory: Turning a "Digital Thread" Into a Battle-Ready Advantage — https://aerospaceamerica.aiaa.org/institute/lockheeds-ai-factory-turning-a-digital-thread-into-a-battle-ready-advantage/
- Digital Engineering Simulation Environment (ARISE) | Lockheed Martin — https://www.lockheedmartin.com/en-us/products/arise-digital-engineering-simulation-environment.html
- Redefining Modeling and Simulation for the Military | Lockheed Martin — https://www.lockheedmartin.com/en-us/news/features/2025/redefining-modeling-and-simulation-for-the-military.html
- The Power of Digital Twins | Lockheed Martin — https://www.lockheedmartin.com/en-us/news/features/2025/the-power-of-digital-twins.html
- Ansys Government Initiatives — https://www.agi.com/about
- Accelerating Advanced Defense Technology with Simulation | Ansys — https://www.ansys.com/industries/defense
- Ansys Government Initiatives Selected to Join Microelectronics Commons — https://www.ansys.com/news-center/press-releases/11-21-24-ansys-joins-me-commons
- NVIDIA Announces PhysicsNeMo (Modulus) — https://developer.nvidia.com/blog/nvidia-announces-modulus-a-framework-for-developing-physics-ml-models-for-digital-twins/
- Physics-Informed Machine Learning Platform NVIDIA PhysicsNeMo Is Now Open Source — https://developer.nvidia.com/blog/physics-ml-platform-physicsnemo-is-now-open-source/
- GitHub - NVIDIA/physicsnemo — https://github.com/NVIDIA/physicsnemo
- Advancing the Field: Luminary Cloud Announces New Physics AI Models at AIAA SciTech Forum — https://aerospaceamerica.aiaa.org/institute/advancing-the-field-luminary-cloud-announces-new-physics-ai-models-at-aiaa-scitech-forum/
- Luminary Launches New Secure Advanced Physics AI Models to Power U.S. Department of War — https://www.globenewswire.com/news-release/2026/01/12/3216955/0/en/luminary-launches-new-secure-advanced-physics-ai-models-to-power-u-s-department-of-war-by-air-land-and-sea.html
- Luminary Cloud expands physics AI toolkit with defense-focused models — Washington Technology — https://www.washingtontechnology.com/companies/2026/01/luminary-cloud-expands-physics-ai-toolkit-defense-focused-models/410679/
- RTX's Raytheon, Northrop Grumman successfully test solid rocket motor — https://www.rtx.com/news/news-center/2024/09/12/rtxs-raytheon-northrop-grumman-successfully-test-solid-rocket-motor
- Shield AI + Palantir: Mission Autonomy and C2 Working as One — https://shield.ai/shield-ai-palantir-mission-autonomy-and-c2-working-as-one/
- DARKHORSE | Hermeus — https://www.hermeus.com/darkhorse
- Hermeus Quarterhorse Mk 1 takes flight — https://www.sandboxx.us/news/hermeus-takes-flight-and-a-big-step-toward-their-first-hypersonic-jet/
- Stratolaunch Achieves Major Milestone: 10+ Successful Hypersonic Flights — https://www.stratolaunch.com/news/stratolaunch-achieves-major-milestone-10-successful-hypersonic-flights-with-reusable-fully-autonomous-talon-a-aircraft/
- Stratolaunch hits milestone with fully reusable hypersonic testbed — Defense News — https://www.defensenews.com/pentagon/2025/05/05/stratolaunch-hits-milestone-with-fully-reusable-hypersonic-testbed/

**Federal labs / FFRDCs**
- 2023 Assessment of the DEVCOM Army Research Laboratory (National Academies) — https://nap.nationalacademies.org/catalog/27503/2023-assessment-of-the-devcom-army-research-laboratory
- 2024 Assessment of the DEVCOM Army Research Laboratory (National Academies) — https://nap.nationalacademies.org/catalog/28878/2024-assessment-of-the-devcom-army-research-laboratory
- DEVCOM Army Research Laboratory — https://arl.devcom.army.mil/
- Advanced materials in a snap - Sandia Labs News Releases — https://newsreleases.sandia.gov/advancing_materials/
- Applications of physics-informed scientific machine learning in subsurface science: A survey — https://arxiv.org/pdf/2104.04764
- A physics-informed operator regression framework for extracting data-driven continuum models — https://arxiv.org/pdf/2009.11992
- Calibrating constitutive models with full-field data via physics informed neural networks — https://arxiv.org/pdf/2203.16577
- Machine Learning in Resource-Constrained Environments | MIT Lincoln Laboratory — https://www.ll.mit.edu/r-d/projects/machine-learning-resource-constrained-environments
- MIT Lincoln Laboratory Research and Development — https://www.ll.mit.edu/r-d

**Geometry / historical references**
- HIFiRE-5 Flight Vehicle Design (NASA NTRS) — https://ntrs.nasa.gov/api/citations/20100026402/downloads/20100026402.pdf
- HIFiRE 4: A low-Cost Aerodynamics, Stability, and Control Hypersonic Flight Experiment — https://www.researchgate.net/publication/271366885
- NASA Technical Memorandum 4613, Radar Cross-Section Measurements and Simulation — https://ntrs.nasa.gov/api/citations/19950013365/downloads/19950013365.pdf
- Radar Cross Section Benchmark for Simple Shapes — EMCoS — https://emcos.com/?application-examples=radar-cross-section-benchmark-for-simple-shapes
- AEDC-TR-94-6, Hypersonic Wind Tunnel Test Techniques (DTIC) — https://apps.dtic.mil/sti/tr/pdf/ADA284057.pdf
- Hypervelocity Wind Tunnel 9 — https://en.wikipedia.org/wiki/Hypervelocity_Wind_Tunnel_9
- Feko Overview — Altair — https://2025.help.altair.com/2025/feko/topics/feko/user_guide/introduction/overview_feko_c.htm

---

*Compiled September 2026. This is a living document (v0.1) — flagged gaps (Boeing Defense-specific physics-AI statements, a centralized AEDC open dataset portal, current M&S CO organizational status, formal explainability requirements for physics-AI surrogates specifically) should be revisited in the next update cycle.*
