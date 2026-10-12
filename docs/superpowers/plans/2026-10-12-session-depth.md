# Session Depth Implementation Plan

> For agentic workers: use superpowers:executing-plans task by task.

Spec: ../specs/2026-10-12-session-depth-design.md. User approved the spec and
explicitly requested implementation on the checked-out refactor branch.

## Task 1: Session-owned profiling

- [ ] Add tests/test_session_profile.py: construct a CPU session with a real
  random-weight torch UNet stand-in, tokenizer and text encoder; assert image,
  exclusive measurements, forward count, and hook/method restoration on success
  and failure through profile(seed).
- [ ] Run new tests red; add ProfileResult and ModelSession.profile(seed).
  Use existing TimeLedger, instrument_pipelines and profile_preprocessing;
  preserve benchmark generation_s as generation exclusive time plus VAE time.
- [ ] Replace benchmark manual instrumentation with profile(seed); retain native
  timers, reset placement, image saving, pixel comparison and result schemas.
- [ ] Run profiling and benchmark regression tests green.

## Task 2: Internal generation context

- [ ] Add a CPU session test exercising the shared context through generate(seed).
  Run red, then add GenerationContext in models/session.py and migrate the
  archived, SDXL and Turbo _generate methods and legacy delegation together.
- [ ] Change the old direct-private-method forwarding test to generate(seed),
  preserving assertions on actual notebook arguments and diagnostics.
- [ ] Run adapter/generation regressions and full CPU suite; compile Python and
  notebook cells, assert seven notebook hashes unchanged.

## Task 3: Review and handoff

- [ ] Run one fresh independent review and address reproduced defects.
- [ ] Append REVIEW.md with measured CPU evidence, update README session contract,
  and commit code/tests/docs. Preserve main and existing remote branch history.
- [ ] Report outcome and limitations. Do not perform GPU generation or fabricate
  timing results. A push uses the existing user-authorized refactor branch.
