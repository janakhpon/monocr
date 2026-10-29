# Changing what the package reads: A/B first, ship only if better

A change that can alter the text this package returns ships only with an A/B
comparison, and only if it is better. That covers the segmenter or its
constants, tiling, preprocessing, decoding, and a new model revision.

- **Two arms, same inputs.** A is the current code, B is the change. Nothing
  else differs between them: same images, same model revision, same settings.
- **Decide what counts as better before running.** Name the primary metric
  (usually CER, or for segmentation, lines found, missed and merged) and the
  guard metrics, and state the result that would ship B.
- **Ship only if B wins the primary metric and loses no guard metric.** A tie or
  an unclear result does not ship. If the change is still worth keeping, it goes
  behind an option that is off by default.
- **Put the result in the pull request:** the inputs, both arms' numbers, any
  per-bucket split (for example by line width), the exact command, and what the
  comparison doesn't cover.
