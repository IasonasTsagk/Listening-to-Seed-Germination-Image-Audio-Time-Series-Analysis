# Phase D exploratory findings

10 independently reviewed visual milestones form 3 temporal groups. Only Phase B crack-onset intervals are available so far.
0 groups have good exact-interval audio coverage under the predeclared >=50% criterion.
The search reused 10327 event-window and 9729 control candidate peaks from the existing find_peaks table.
Preliminary classification: **insufficient_audio_coverage**. Blinded listening remains pending.

The reference is an unexplained transient of unknown source. Its self-similarity is 1 by construction; this is not independent replication.
High feature similarity does not identify a physical source, and gaps without WAVs are unknown rather than silent.
One microphone observes all seeds; overlapping visual intervals were grouped before acoustic comparison.
A dramatic visual change may be gradual, too faint, substrate-borne, masked, or occur during an audio gap.

Similarity method: July 14 peak population IQR per feature; equal feature weights; missing values imputed by population median; 1/(1+sqrt(mean(((candidate-reference)/IQR)^2)))
Controls: Same day, fixed clock offsets, no overlap with any visual ±5-minute window, coverage matched without using peak activity; two requested per group.

Next: independently review later visual milestones in notebook 14 and label neutral-ID WAVs in notebook 15 before opening the mapping.
