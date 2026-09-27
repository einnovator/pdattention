# vLLM/CUDA Pi N=3 request-81 mechanism gate

This directory records the largest exact frozen Pi fixture admitted by the
RTX 5060 8 GB host at Paper 4.5 commit `ec26528a`.  vLLM 0.28 consumes the
receipt-free N=3 request-81 plan through scheduler page aliases:

- 37,784 logical source tokens become 37,792 resident page tokens;
- 13,418 selected tokens become 13,440 page tokens (+22 rounding tokens);
- resident K/V omission is 64.45% and total-visible saving is 60.99%;
- selected-history re-encoding, selected-K/V copy, and H2D are zero;
- two executions emit the same token IDs `[151667, 198]`;
- two alias installs/releases and final cleanup pass.

The result qualifies page aliasing and the runner's bounded lifecycle scope.
It does not yet qualify same-subset logit/dense-oracle parity or consumer
scratch/temporary bytes.  The separate 48-token scheduler lifecycle smoke has
broader cancellation, stale-generation, error, and eviction coverage; it is
not conflated with this frozen request.

