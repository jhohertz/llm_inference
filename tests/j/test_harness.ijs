NB. ================================================================
NB. Test Harness — centralized reporting for all test suites
NB. Include: load './tests/j/test_harness.ijs'
NB. Provides: pass, fail, assert_test, section_header, show_summary, section_start, section_time
NB. Uses global counters G_tc/G_pc/G_fc/G_fl and timer G_ts
NB. NOTE: uses =: (capital) inside verbs to write globals
NB. ================================================================

coclass 'inference'
init_counters =: 3 : 0
  G_tc =: 0
  G_pc =: 0
  G_fc =: 0
  G_fl =: ''
  G_ts =: 6!:1 ''   NB. session time at start
)

pass =: 3 : 0
  G_tc =: G_tc + 1
  G_pc =: G_pc + 1
  echo '  PASS: ' , y
)

fail =: 3 : 0
  G_tc =: G_tc + 1
  G_fc =: G_fc + 1
  G_fl =: G_fl , y , LF
  echo '  FAIL: ' , y
)

assert_test =: 3 : 0
  NB. assert_test (<cond ; msg)
  NB. pass/fail already increment G_tc — do NOT count here too (the old
  NB. extra G_tc increment made every assertion count twice, e.g. Chat
  NB. Template reported 6/3 instead of the real 3/3).
  cond =. > 0 { y
  msg =. > 1 { y
  if. cond do.
    pass msg
  else.
    fail msg
  end.
)

section_header =: 3 : 0
  echo ''
  echo '--- ' , y , ' ---'
)

section_start =: 3 : 0
  G_ts =: 6!:1 ''   NB. reset timer at section start
  ''
)

section_time =: 3 : 0
  NB. Report elapsed time for current section
  NB. Usage: echo '  Time: ' , ": section_time ''
  elapsed =. 6!:1 '' - G_ts
  elapsed
)

show_summary =: 3 : 0
  NB. y = 1 to include timing; default 0
  show_time =. 0
  if. 0 < # y do.
    show_time =. > 0 { y
  end.
  
  echo ''
  echo '========== TEST SUMMARY =========='
  echo '  Total: ' , ": G_tc
  echo '  Pass:  ' , ": G_pc
  echo '  Fail:  ' , ": G_fc
  if. show_time do.
    elapsed =. section_time 0
    s =. ": elapsed
    echo '  Time:  ' , s , 's'
  end.
  if. 0 < # G_fl do.
    echo ''
    echo 'Failed tests:'
    echo G_fl
  end.
  echo ''
  if. 0 = G_fc do.
    echo 'All tests passed!'
  else.
    echo 'Some tests failed.'
  end.
  echo ''
)
