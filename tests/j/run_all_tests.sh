#!/usr/bin/env bash
# Master test runner - fresh J process per suite, unified report with timing
set -euo pipefail

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
JINSTALL="$( "$BASE/scripts/jfind.sh" )"
# Runner uses the raw binary (NOT jconsole.sh, which cds to the J install dir)
# so J's working directory stays at the checkout root — test files use
# ./-relative paths.
J="${J:-$JINSTALL/bin/jconsole}"

# Run suites from the checkout root so ./ paths in test files resolve.
cd "$BASE"

TOTAL_TC=0 TOTAL_PC=0 TOTAL_FC=0
TOTAL_TIME=0
declare -A RESULTS
declare -A TIMINGS

run_suite() {
  local name="$1" file="$2"
  local output tc pc fc suite_time
  local start_time=$(date +%s%N)
  
  output=$($J <<EOFILE 2>&1
load '$file'
EOFILE
  )
  local end_time=$(date +%s%N)
  suite_time=$(( (end_time - start_time) / 1000000 ))  # ms → ms
  
  # Parse summary - handles both harness format and inline format
  tc=$(echo "$output" | grep -oP '(Total|total):\s+\K[0-9]+' | tail -1 || echo 0)
  pc=$(echo "$output" | grep -oP '(Pass|Passed):\s+\K[0-9]+' | tail -1 || echo 0)
  fc=$(echo "$output" | grep -oP '(Fail|Failed):\s+\K[0-9]+' | tail -1 || echo 0)
  tc=${tc:-0}; pc=${pc:-0}; fc=${fc:-0}

  echo "$output"

  # Display timing
  if [ "$suite_time" -ge 1000 ]; then
    time_str="$(echo "scale=1; $suite_time / 1000" | bc)s"
  else
    time_str="${suite_time}ms"
  fi
  
  echo "  ⏱  ${time_str}"
  echo ""

  # A suite that ran 0 tests almost certainly crashed before its summary
  # (error output has no Total line) — treat it as a failure, not "OK 0/0".
  if [ "$tc" -eq 0 ]; then
    fc=1
    echo "  WARNING $name: no tests executed (possible crash)"
  elif [ "$fc" -gt 0 ]; then
    echo "  WARNING $name: $fc failure(s)"
  else
    echo "  OK $name: all passed"
  fi
  echo ""

  RESULTS[$name]="$tc:$pc:$fc"
  TIMINGS[$name]="$time_str"
  TOTAL_TC=$((TOTAL_TC + tc))
  TOTAL_PC=$((TOTAL_PC + pc))
  TOTAL_FC=$((TOTAL_FC + fc))
  TOTAL_TIME=$((TOTAL_TIME + suite_time))
}

echo "################################################################"
echo "#          LLM-IN-J - FULL TEST SUITE                        #"
echo "################################################################"
echo ""
echo "################################################################"
echo "#  !!!! WARNING: THIS SUITE DOWNLOADS MANY MANY GB !!!!      #"
echo "#  Model tests resolve catalog ids / HF specs via model_path #"
echo "#  and fetch any GGUF not already cached in ~user/models.   #"
echo "#  First run can pull tens of GB of weights.                #"
echo "#  Ensure disk space + bandwidth before running.            #"
echo "################################################################"
echo ""

run_suite "Kernels"         "$BASE/tests/j/test_kernels.ijs"
run_suite "KV Cache"        "$BASE/tests/j/test_kv_cache.ijs"
run_suite "Tokenizer"       "$BASE/tests/j/test_tokenizer_llama3.ijs"
run_suite "GGUF Parser"     "$BASE/tests/j/test_gguf.ijs"
run_suite "Model Catalog"   "$BASE/tests/j/test_models.ijs"
run_suite "Chat Template"   "$BASE/tests/j/test_chat.ijs"
run_suite "Chat Session"    "$BASE/tests/j/test_chat_session.ijs"
run_suite "Sampler"         "$BASE/tests/j/test_sampler.ijs"
run_suite "Multi-Arch"      "$BASE/tests/j/test_multiach.ijs"
run_suite "Gemma3 Model"    "$BASE/tests/j/test_gemma3_all.ijs"
run_suite "SWA Boundary"    "$BASE/tests/j/test_swa.ijs"
run_suite "Llama Arch"      "$BASE/tests/j/test_llama.ijs"
run_suite "Llama-3.2 1B"    "$BASE/tests/j/test_llama32.ijs"
run_suite "Qwen2 Coder"     "$BASE/tests/j/test_qwen2.ijs"
run_suite "Qwen3 0.6B"      "$BASE/tests/j/test_qwen3.ijs"
run_suite "Qwen3.5 0.8B"    "$BASE/tests/j/test_qwen35.ijs"
run_suite "Granite 350M"    "$BASE/tests/j/test_granite.ijs"
run_suite "ERNIE 0.3B"      "$BASE/tests/j/test_ernie.ijs"
run_suite "LFM2 350M"       "$BASE/tests/j/test_lfm2.ijs"
run_suite "LFM2.5 230M"     "$BASE/tests/j/test_lfm25.ijs"
run_suite "LFM2 700M"       "$BASE/tests/j/test_lfm2700.ijs"
run_suite "Batched Decode"  "$BASE/tests/j/test_batched.ijs"

echo "################################################################"
echo "#              MASTER TEST SUMMARY                           #"
echo "################################################################"
echo ""
printf "  %-25s %6s %6s %6s %8s\n" "Suite" "Total" "Pass" "Fail" "Time"
printf "  %-25s %6s %6s %6s %8s\n" "-------------------------" "------" "------" "------" "--------"
# Build sorted list of suite names, preserving keys with spaces
mapfile -t names < <(printf '%s\n' "${!RESULTS[@]}" | sort)
for name in "${names[@]}"; do
  IFS=: read -r tc pc fc <<< "${RESULTS[$name]}"
  printf "  %-25s %6s %6s %6s %8s\n" "$name" "$tc" "$pc" "$fc" "${TIMINGS[$name]}"
done
printf "  %-25s %6s %6s %6s %8s\n" "-------------------------" "------" "------" "------" "--------"
if [ "$TOTAL_TIME" -ge 1000 ]; then
  total_time_str="$(echo "scale=1; $TOTAL_TIME / 1000" | bc)s"
else
  total_time_str="${TOTAL_TIME}ms"
fi
printf "  %-25s %6s %6s %6s %8s\n" "TOTAL" "$TOTAL_TC" "$TOTAL_PC" "$TOTAL_FC" "$total_time_str"
echo ""

if [ "$TOTAL_FC" -gt 0 ]; then
  echo "SOME TEST SUITES FAILED."
  for name in "${names[@]}"; do
    IFS=: read -r tc pc fc <<< "${RESULTS[$name]}"
    if [ "$fc" -gt 0 ]; then
      echo "  - $name (${TIMINGS[$name]}, $fc failures)"
    fi
  done
fi
echo ""
echo "################################################################"
echo "#              J LINT CHECK (jlinter / debug/lint)          #"
echo "#  Headless lint over the checkout's runtime .ijs files.   #"
echo "#  debug/lint \"undefined name\" findings are expected for a #"
echo "#  multi-file addon (each file checked standalone); the    #"
echo "#  LOAD-PROBE is the gate — a runtime file that fails to   #"
echo "#  load (real syntax/load regression) fails this step.     #"
echo "#  Requires addons/tmcguire/jlinter installed.             #"
echo "################################################################"
echo ""
LINT_EXIT=0
if "$BASE/scripts/lint.sh"; then
  echo ""
  echo "  OK lint: all runtime files load"
else
  LINT_EXIT=1
  echo ""
  echo "  WARNING lint: a runtime file failed to load"
fi
echo ""

echo "################################################################"
echo "#              MASTER SUMMARY                                 #"
echo "################################################################"
echo ""
if [ "$TOTAL_FC" -gt 0 ] || [ "$LINT_EXIT" -gt 0 ]; then
  echo "SOME CHECKS FAILED."
  if [ "$TOTAL_FC" -gt 0 ]; then echo "  - test suites: $TOTAL_FC failure(s)"; fi
  if [ "$LINT_EXIT" -gt 0 ]; then echo "  - lint: a runtime file failed to load"; fi
else
  echo "ALL TESTS + LINT PASSED."
fi
echo ""
echo "################################################################"
