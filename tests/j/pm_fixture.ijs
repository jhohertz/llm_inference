NB. ================================================================
NB. jpm fixture — Performance Monitor tracing for all test suites
NB. Include: load '.../tests/j/pm_fixture.ijs'
NB. Provides: pm_start (start tracing), pm_report (echo showtotal)
NB.
NB. Always-on by design: we trace EVERYTHING (model load + inference)
NB. so we can measure changes over time and root out dumb load code.
NB. Overhead is accepted.
NB.
NB. jpm lives in system (system/util/pm.ijs, locale 'jpm') but is NOT
NB. auto-loaded in a fresh console — the fixture loads it once per session.
NB. Console API: start_jpm_ <size>, showtotal_jpm_ '' (summary table),
NB. showdetail_jpm_ '' (per-line detail). The old GUI viewtotal_jpm_
NB. is gone in J9.7.
NB. ================================================================
coclass 'inference'
load 'jpm'

pm_start =: 3 : 0
  NB. Default SIZE = 1e9 on 64-bit (pm.ijs). The model suites trace load
  NB. + inference with hundreds of thousands of lines; a small buffer
  NB. overflows and corrupts the profile, so use the full-size buffer.
  start_jpm_ ''
  ''
)

pm_report =: 3 : 0
  echo showtotal_jpm_ ''
  echo ''
  ''
)
