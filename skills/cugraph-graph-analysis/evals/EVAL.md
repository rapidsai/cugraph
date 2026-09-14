# cuGraph graph-analysis evaluation guidance

Assessment evidence is produced with [NVIDIA SkillEvaluator](https://github.com/NVIDIA/SkillEvaluator); CPU/static checks remain distinct from cuGraph/GPU execution evidence.

## Questions
- Include coding-agent tasks for directed reachability, PageRank convergence/weight semantics, and weak-versus-strong component construction.
- Include multiplicity, composite-key renumbering, symmetrization/metadata compatibility, method-selection, and negative table-only cases.

## Behaviors
- Reward explicit direction, weights, parallel-edge and isolate policy, stable IDs, and algorithm-specific validation.
- Require agents to distinguish NetworkX CPU reference evidence from cuGraph/GPU execution.
- Penalize unnecessary graphs, silent algorithm substitution, and causal interpretation of structural output.

## Notes
- Retain explicit coding, implicit, contextual, and negative buckets; add repo-specific cases when a strong baseline saturates the basic set.
- Baseline and with-skill arms receive identical selected fixtures.
- The default sandbox need not contain a GPU; an honest runtime blocker is preferable to a fabricated cuGraph run.
