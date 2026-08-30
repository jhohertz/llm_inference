NB. ================================================================
NB. SWA Boundary Regression — gemma3 sliding-window attention at pos >= 512
NB. Guards the two J right-to-left precedence gotchas fixed in commit 8933009:
NB.   - gem3_attention single-token mask: 'mask_1d =. swa_l > 0 *. mask_1d'
NB.     parsed as swa_l > (0 *. mask_1d) -> all keys masked for SWA layers ->
NB.     uniform attention -> the CLI generation bug: long-context answers
NB.     degenerate into '**' repetition past the 512 window.
NB.   - gem3_attention_b batched mask: 'mask_swa =. ((i. L) - swa_l + 1) >/ (i. L)'
NB.     parsed as (t - (swa_l+1)) -> 514-key window instead of 512.
NB. Reference: llama-cpp-python 0.3.34 logits for the SAME 600-token list
NB. (gemma chat prompt + filler). Our J engine matches llama_cpp to ~0.01
NB. maxdiff (F16 precision); the buggy code diverged ~5.96 at pos 599.
NB. Checks:
NB.   - single-token path == batched path at boundary positions (< 1e-6)
NB.   - batched top-5 token ids == reference at 513/520/550/599
NB.   - batched top-5 logits within 0.05 of reference
NB. ================================================================

coclass 'inference'
load './inference.ijs'
load './tests/j/test_harness.ijs'
load './tests/j/pm_fixture.ijs'

NB. 600-token context: gemma chat prompt + 'The capital of France is Paris. '
NB. filler (tokenized by our J tokenizer; llama_cpp eval'd the SAME list, so the
NB. pin is self-consistent — it guards the forward-pass SWA masks, not the
NB. tokenizer).
swa_toks =: 2 105 2364 107 3689 563 45556 28243 236881 106 107 105 4368 107 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 818 818 818 818 818 818 818 818 818 818 818 818 818 818 818 818 818 818 818 818 818 818 818 818 818 818 818 818 818 818 818 818 818 818 818 818 818 818 818 818 818 818 818 818 818 818 818 818 818 818 818 818 818 818 818 818 818 818 818 818 818 818 818 818 818 818 5279 5279 5279 5279 5279 5279 5279 5279 5279 5279 5279 5279 5279 5279 5279 5279 5279 5279 5279 5279 5279 5279 5279 5279 5279 5279 5279 5279 5279 5279 5279 5279 5279 5279 5279 5279 5279 5279 5279 5279 5279 5279 5279 5279 5279 5279 5279 5279 5279 5279 5279 5279 5279 5279 5279 5279 5279 5279 5279 5279 5279 5279 5279 5279 5279 5279 529 529 529 529 529 529 529 529 529 529 529 529 529 529 529 529 529 529 529 529 529 529 529 529 529 529 529 529 529 529 529 529 529 529 529 529 529 529 529 529 529 529 529 529 529 529 529 529 529 529 529 529 529 529 529 529 529 529 529 529 529 529 529 529 529 529 7001 7001 7001 7001 7001 7001 7001 7001 7001 7001 7001 7001 7001 7001 7001 7001 7001 7001 7001 7001 7001 7001 7001 7001 7001 7001 7001 7001 7001 7001 7001 7001 7001 7001 7001 7001 7001 7001 7001 7001 7001 7001 7001 7001 7001 7001 7001 7001 7001 7001 7001 7001 7001 7001 7001 7001 7001 7001 7001 7001 7001 7001 7001 7001 7001 7001 563 563 563 563 563 563 563 563 563 563 563 563 563 563 563 563 563 563 563 563 563 563 563 563 563 563 563 563 563 563 563 563 563 563 563 563 563 563 563 563 563 563 563 563 563 563 563 563 563 563 563 563 563 563 563 563 563 563 563 563 563 563 563 563 563 563 9079 9079 9079 9079 9079 9079 9079 9079 9079 9079 9079 9079 9079 9079 9079 9079 9079 9079 9079 9079 9079 9079 9079 9079 9079 9079 9079 9079 9079 9079 9079 9079 9079 9079 9079 9079 9079 9079 9079 9079 9079 9079 9079 9079 9079 9079 9079 9079 9079 9079 9079 9079 9079 9079 9079 9079 9079 9079 9079 9079 9079 9079 9079 9079 9079 9079 236761 236761 236761 236761 236761 236761 236761 236761 236761 236761 236761 236761 236761 236761 236761 236761 236761 236761 236761 236761 236761 236761 236761 236761 236761 236761 236761 236761 236761 236761 236761 236761 236761 236761 236761 236761 236761 236761 236761 236761 236761 236761 236761 236761 236761 236761 236761 236761 236761 236761 236761 236761 236761 236761 236761 236761 236761 236761 236761 236761 236761 236761 236761 236761 236761 236761 236743 236743 236743 236743 236743 236743 236743 236743 236743 236743 236743 236743 236743 236743 236743 236743 236743 236743 236743 236743 236743 236743 236743 236743 236743 236743 236743 236743 236743 236743 236743 236743 236743 236743 236743 236743 236743 236743 236743 236743 236743 236743 236743 236743 236743 236743 236743 236743 236743 236743 236743 236743 236743 236743 236743 236743 236743 236743

NB. Reference (llama-cpp-python 0.3.34, same token list): top-5 token ids + top-5
NB. logits at the window boundary. Positions: 513 520 550 599.
NB. Boxed: <pos ; top5toks ; top5vals>
swa_ref =: <((<513) , (<108 107 669 138 9079) , (<20.3122 20.1472 19.5 19.1371 19.0256))
swa_ref =: swa_ref , <((<520) , (<108 107 669 138 9079) , (<20.5429 20.4143 19.9514 19.3571 19.1892))
swa_ref =: swa_ref , <((<550) , (<236770 236778 108 107 236800) , (<22.5765 22.5616 21.626 21.045 20.9932))
swa_ref =: swa_ref , <((<599) , (<236770 236778 236800 236810 236812) , (<23.9056 23.2644 22.9461 22.3263 22.239))

test_swa =: 3 : 0
  init_counters ''
  echo '========================================'
  echo 'SWA Boundary Regression (gemma3 pos >= 512)'
  echo '========================================'
  echo ''

  gemma_path =. 'gemma-3-270m-it'
  echo '  (loading gemma 270M)...'
  llm =. load_gguf_to_llm gemma_path

  NB. ---- Build embeddings + batched prefill of the 600-token context ----
  mi =. llm_mi llm
  emb_len =. mi_emb_len mi
  emb_w =. 'token_embd.weight' get_tensor_cached_d llm
  scale =. %: emb_len
  emb_all =. scale * |: (swa_toks {"1 emb_w)
  result_b =. emb_all gem3_run_blocks_b ((<llm) , <0)
  h_b =. > 0 { result_b
  output_norm_w =. 'output_norm.weight' get_tensor_cached_d llm
  eps =. mi_rms_eps mi
  emb_final =. 'token_embd.weight' get_tensor_cached_d llm

  NB. ---- Batched logits at the 4 boundary positions (vectorized) ----
  rows =. (513 520 550 599 { h_b)
  norm_rows =. rms_norm_rows ((<eps) , (<output_norm_w) , <rows)
  lg_mat =. norm_rows (+/ .*) emb_final   NB. (4, vocab)

  NB. ---- Single-token path at the 4 boundary positions (resume from KV) ----
  sgl =. ''
  for_xyz. 513 520 550 599 do.
    p =. xyz
    hidden =. scale * |: ((p { swa_toks) {"1 emb_w)
    h_s =. > 0 { (hidden gem3_run_blocks (<llm) , <p)
    lg_s =. output_head ((<eps) , (<output_norm_w) , (<emb_final) , <h_s)
    sgl =. sgl , <lg_s
  end.

  NB. ---- Checks ----
  tol =. 0.05
  for_xyz. swa_ref do.
    item =. > 0 { xyz
    pos =. > 0 { item
    ref_toks =. > 1 { item
    ref_vals =. > 2 { item
    row_idx =. 513 520 550 599 i. pos
    batched_lg =. row_idx { lg_mat
    single_lg =. > row_idx { sgl
    bt =. 5 {. \: batched_lg
    st =. 5 {. \: single_lg
    assert_test (1e_6 > >./ | single_lg - batched_lg) ; ('pos ' , (": pos) , ': single == batched (SWA single-token mask)')
    assert_test (bt -: ref_toks) ; ('pos ' , (": pos) , ': batched top-5 tokens == reference (SWA batched mask)')
    assert_test (st -: ref_toks) ; ('pos ' , (": pos) , ': single top-5 tokens == reference')
    assert_test (tol > >./ | (bt { batched_lg) - ref_vals) ; ('pos ' , (": pos) , ': batched top-5 logits within ' , (": tol) , ' of reference')
  end.

  echo ''
  show_summary 1
  ''
)

pm_start 1e8

test_swa 0

pm_report ''
